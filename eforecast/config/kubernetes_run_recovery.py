# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Recover only from a positively terminated, exactly bound Pod.

A missing/unreachable Pod is never death. The Pod has no host PID namespace, privileged
mode, Docker socket, sidecars or API token. Its terminated worker container therefore bounds
its native descendants. Foreign namespaces and external containers remain unknown.
"""
from dataclasses import replace
import json
import time
from types import SimpleNamespace

from .executors import LocalExecutor
from .run_model import Outcome, pinned
from . import run_cleanup, run_reporting
from .run_coordinator import control_allocator, require
from eforecast.training.command_line.allocation_contract import Token


def recover_terminated(store, run, bound, observed):
    from .runs import RunManager
    evidence = json.loads(observed.evidence)
    if (observed.state != 'finished' or not evidence.get('pod_uid') or
            (bound['pod_uid'] and evidence['pod_uid'] != bound['pod_uid'])):
        raise ValueError('termination does not name the assigned Pod')
    reason = f"assigned Pod {evidence['pod_uid']} terminated before managed cleanup"
    if run.cleanup == 'not_required':
        run = store.compare_and_set(replace(run, cleanup='pending'), run.revision)
    try:
        # Retire the exact controller first. A late writer then fails its ordinary A08 guard.
        if run.controller_json:
            controller = json.loads(run.controller_json)
            if controller['owner'].get('container_id') != bound['pod_uid']:
                raise ValueError('controller is not owned by the terminated Pod')
            ledger = control_allocator(store, json.loads(bound['handle'].identity_json)['directory'])
            token = Token(**controller['token'])
            settlement = require(ledger.retire_stopped(token, lambda entry: 'dead' if entry.process and
                entry.process.container_id == bound['pod_uid'] and entry.request.run_id == run.id
                else 'unknown', reason))
            # Controller reservations have no model settlement target.
            if settlement is not None:
                require(ledger.finish_settlement(token))
        run = store.get(run.owner_id, run.case_id, run.id)
        handle = run.tasks[0].attempts[-1].executor
        intent, spec = LocalExecutor._spec(handle) if handle else (None, None)
        if spec:
            if spec['location'] != bound['agent']:
                raise ValueError('root execution namespace differs from the attached Pod')
            context = {'run_id': run.id, 'task_id': run.tasks[0].id,
                       'attempt_id': run.tasks[0].attempts[-1].id, 'executor_id': handle.id}
            events, offset, at_end, partial = run_reporting.read_batch(intent['report_directory'], run.report_offset, context)
            if events:
                run = store.collect_reports(run, events, offset)
            if not at_end:
                return run  # bounded report collection before settlement
            if partial:
                raise ValueError('terminated Pod left an incomplete report journal')
        plan = json.loads(run.cleanup_json) if run.cleanup_json else {}
        page = store.children(run.owner_id, run.case_id, run.id, after=plan.get('kube_cursor', ''), limit=64)
        if any(child['kind'] == 'container' for child in page['items']):
            raise ValueError('external container cleanup requires its own liveness evidence')
        # The run's own workers were processes of the Pod and ended with it; nothing of theirs
        # is recorded anywhere that would have to be given back.
        if store.child_summary(run)['unrooted']:
            raise ValueError('unrooted child journal needs explicit inspection')
        if run.state != 'terminal':
            run = RunManager._execution_progress(run, 'terminal', outcome=Outcome(
                'cancelled' if run.cancel else 'failure', observed.finished_at, observed.returncode, reason))
        complete = not page['next']
        plan.update(kube_cursor=page['next'] or '', phase='complete' if complete else 'running',
                    pod_uid=evidence['pod_uid'], reason=reason, observed_at=time.time())
        updates = {child['node_id']: {'state':'complete', 'detail':reason, 'observed_at':time.time()}
                   for child in page['items']}
        run = store.collect_cleanup(replace(run, cleanup='complete' if complete else 'running',
            cleanup_json=pinned(plan), report_error=None), updates)
        return run
    except (ValueError, OSError, KeyError, TypeError) as error:
        latest = store.get(run.owner_id, run.case_id, run.id)
        return store.compare_and_set(replace(latest, cleanup='unknown', report_error=f'Kubernetes recovery: {error}'), latest.revision)
