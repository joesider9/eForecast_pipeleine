# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Restartable, bounded cancellation/cleanup. A signal is never completion evidence."""
from dataclasses import asdict, replace
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
import subprocess
import time

from eforecast.config.executors import LocalExecutor, atomic_json, observe_process, same_location
from eforecast.config.run_model import Outcome, pinned
from eforecast.config.run_store import RunConflict, StoreUnavailable
from eforecast.config.run_processes import signal_owned

_docker_environment = ContextVar('cleanup_docker_environment', default=None)
DOCKER_ENV = ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_CONFIG', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH')


def docker(*arguments):
    environment = _docker_environment.get()
    result = subprocess.run(['docker', *arguments], capture_output=True, text=True, timeout=3, env=environment)
    if result.returncode:
        if any(text in result.stderr.casefold() for text in ('no such object:', 'no such container:')):
            return None
        raise ValueError('Docker cleanup is unreachable or refused')
    return result.stdout


def container(reference, expected, *, force, evidence=None):
    """Resolve names only through exact labels; signal/remove the verified full ID."""
    raw = docker('inspect', reference)
    if raw is None:
        return 'complete'
    item, = json.loads(raw)
    cid = item['Id']
    if not re.fullmatch('[0-9a-f]{64}', cid):
        raise ValueError('Docker did not return a full container identity')
    labels = item['Config'].get('Labels') or {}
    if not expected or any(labels.get(key) != str(value) for key, value in expected.items()):
        raise ValueError('container identity/labels disagree; no signal sent')
    if re.fullmatch('[0-9a-f]{64}', reference) and cid != reference:
        raise ValueError('container identity changed; no signal sent')
    if evidence is not None:
        evidence['container_id'] = cid
    state = item['State']['Status']
    if (item.get('HostConfig', {}).get('RestartPolicy') or {}).get('Name') not in (None, '', 'no'):
        docker('update', '--restart=no', cid)
        return 'running'  # confirm the policy/state on a fresh inspect before declaring cleanup
    if state in {'exited', 'dead'}:
        return 'complete'
    if state == 'paused':
        docker('unpause', cid)
        return 'running'
    if state == 'created':
        docker('rm', cid)  # prevents a delayed start from bringing this owned intent to life
    else:
        docker('kill', '--signal', 'KILL' if force else 'TERM', cid)
    return 'running'  # re-inspect on the next pass; sending a signal is not proof


def docker_inventory(run, handle, *, force, evidence):
    labels = {'eforecast.managed_run': run.id, 'eforecast.executor': handle.id}
    command = ['ps', '-aq', '--no-trunc']
    for state in ('created', 'running', 'restarting', 'paused', 'removing'):
        command += ['--filter', f'status={state}']
    for key, value in labels.items():
        command += ['--filter', f'label={key}={value}']
    ids = sorted(set((docker(*command) or '').split()) | set(evidence))
    if len(ids) > 256:
        raise ValueError('Docker inventory exceeds one cleanup batch; ownership retained')
    states = []
    for cid in ids:
        state = container(cid, labels, force=force)
        evidence[cid] = state
        states.append(state)
    return all(state == 'complete' for state in states)


def step(manager, run, *, grace=10):
    token = None
    try:
        handle = manager._executor_handle(run)
        if handle is not None and handle.adapter == LocalExecutor.adapter:
            _, spec = manager.executor._spec(handle)
            environment = {key: value for key, value in os.environ.items() if key not in DOCKER_ENV}
            environment.update({key: spec['environment'][key] for key in DOCKER_ENV if key in spec['environment']})
            token = _docker_environment.set(environment)
        return _step(manager, run, grace=grace)
    except (RunConflict, StoreUnavailable):
        raise
    except Exception as error:
        latest = manager.store.get(run.owner_id, run.case_id, run.id)
        plan = json.loads(latest.cleanup_json) if latest.cleanup_json else {}
        plan.update(phase='unknown', issues=[f'{type(error).__name__}: {error}'], observed_at=time.time(),
                    next_retry_at=time.time() + 2)
        return manager.store.compare_and_set(replace(latest, cleanup='unknown', cleanup_json=pinned(plan)), latest.revision)
    finally:
        if token is not None:
            _docker_environment.reset(token)


def _step(manager, run, *, grace=10):
    """One nonblocking pass. Persist escalation before effects; resume the same plan on restart."""
    if run.cleanup == 'complete':
        return run
    handle = manager._executor_handle(run)
    if handle is None:
        if run.state in {'pending', 'preparing', 'queued'} and run.cancel:
            ended = manager._execution_progress(run, 'terminal', outcome=Outcome('cancelled', time.time(), None,
                                                                      'cancelled before launch'))
            return manager.store.compare_and_set(replace(ended, cleanup='complete',
                cleanup_json=pinned({'phase': 'complete', 'reason': 'no executor was committed'})), run.revision)
        return manager.store.compare_and_set(replace(run, cleanup='unknown',
            cleanup_json=pinned({'phase': 'unknown', 'issues': ['execution ownership is unverified']})), run.revision)
    if handle.adapter != LocalExecutor.adapter:
        raise ValueError('cleanup requires a supported owned executor')
    intent, spec = manager.executor._spec(handle)
    if not same_location(intent):
        raise ValueError('cleanup execution namespace is not visible')
    now = time.time()
    plan = {'started_at': now, 'term_deadline': (run.cancel.requested_at if run.cancel else now) + grace,
            'monotonic_deadline': time.monotonic() + max(0, (run.cancel.requested_at if run.cancel else now) + grace - now),
            'passes': 0, 'cursor': '', 'page_issues': 0,
            **(json.loads(run.cleanup_json) if run.cleanup_json else {})}
    if now < plan.get('next_retry_at', 0):
        return run
    force = plan.get('force', False) or now >= plan['term_deadline'] or time.monotonic() >= plan['monotonic_deadline']
    plan.update(phase='kill' if force else 'term', force=force, passes=plan['passes'] + 1, observed_at=now)
    run = manager.store.compare_and_set(replace(run, cleanup='running', cleanup_json=pinned(plan)), run.revision)
    check = getattr(manager.store, 'check_ownership', lambda: None)
    check()
    # Also stop late descendant dispatch after a normal/abnormal root exit. The root outcome
    # stays unchanged; this marker closes fitting entry points in the shared worker namespace.
    atomic_json(Path(intent['report_directory']) / 'cancel.json', {'executor_id': handle.id})
    if run.cancel is not None and run.state != 'terminal':
        manager.executor.cancel(handle, force=force)
        if manager.executor.discover(handle).state == 'absent':
            check()
            manager.executor.submit(handle)  # durable cancel marker prevents starting the fitter
    run = manager._observe_executor(run)
    folder = Path(intent['directory'])
    issues = []
    unknown = False
    try:
        receipt = manager.executor._receipt(folder, 'tree_cleanup.json', handle, intent)
        tree_complete = bool(receipt and receipt.get('complete') is True)
        if not tree_complete:
            tree = manager.executor._receipt(folder, 'tree.json', handle, intent)
            for process in (tree or {}).get('processes', []):
                check()
                signal_owned(process, force=force)
            issues.append('local descendant tree has not confirmed exit/reap')
            started = manager.executor._receipt(folder, 'started.json', handle, intent)
            unknown = not started or observe_process(started['process']) != 'alive'
        page = manager.store.children(run.owner_id, run.case_id, run.id, after=plan.get('cursor') or '', limit=64)
        children = page['items']
        summary = manager.store.child_summary(run)
        needs_docker = bool(spec['environment'].get('EFORECAST_WORKER_IMAGE') or
                            any(group['kind'] == 'container' for group in summary['groups']))
        inventory = plan.setdefault('docker_inventory', {})
        docker_complete = not needs_docker or docker_inventory(run, handle, force=force, evidence=inventory)
        if not docker_complete:
            issues.append('owned Docker executions have not confirmed exit')
        updates = {}
        for child in children:
            detail, status = 'root descendant tree confirmed reaped', 'complete' if tree_complete else 'running'
            container_evidence = {}
            try:
                execution = child['execution'] or {}
                if child['kind'] == 'container':
                    identity = child['identity']
                    expected = ({'eforecast.run': identity['orchestrator_run_id'],
                                 'eforecast.task': identity['task'], 'eforecast.attempt': identity['attempt']}
                                if all(key in identity for key in ('orchestrator_run_id', 'task', 'attempt'))
                                else {'eforecast.managed_run': run.id, 'eforecast.executor': handle.id})
                    reference = execution.get('container_id') or (child.get('cleanup') or {}).get('container_id') or identity['name']
                    check()
                    status = container(reference, expected, force=force, evidence=container_evidence)
                    detail = 'exact labelled Docker inspection'
                    if (status == 'complete' and not container_evidence and not re.fullmatch('[0-9a-f]{64}', reference)
                            # 'legacy_labelled_name' is how runs of earlier versions recorded the same thing.
                            and (execution.get('create_requested')
                                 or identity.get('provenance') in {'labelled_name', 'legacy_labelled_name'})
                            and (child['outcome'] or {}).get('returncode') != 0):
                        status, detail = 'unknown', 'Docker create/start was requested but its exact identity is unresolved'
                else:
                    for field in ('process', 'worker'):
                        process = execution.get(field)
                        if process:
                            if (field == 'worker' and needs_docker and docker_complete and tree_complete
                                    and process.get('host') == intent['location']['host']
                                    and process.get('boot_id') == intent['location']['boot_id']
                                    and process.get('pid_namespace') != intent['location']['pid_namespace']):
                                detail = 'owned Docker inventory is stopped; container PID was not signalled as a host PID'
                                continue
                            check()
                            verdict = signal_owned(process, force=force)
                            if verdict != 'dead':
                                status = 'unknown' if verdict == 'unknown' else 'running'
                                detail = f'{field} liveness {verdict}'
                if status != 'complete':
                    issues.append(f'{child["node_id"]}: {detail}')
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                status, detail = 'unknown', str(error)
                issues.append(f'{child["node_id"]}: {detail}')
            updates[child['node_id']] = {'state': status, 'detail': detail, 'observed_at': now, **container_evidence}
            unknown |= status == 'unknown'
        if summary['unrooted'] or run.report_error:
            issues.append('child journal is incomplete or unrooted; explicit recovery required')
            unknown = True
        plan['page_issues'] = plan.get('page_issues', 0) + len(issues)
        complete = run.state == 'terminal' and not page['next'] and plan['page_issues'] == 0
        plan.update(cursor=page['next'] or '', issues=issues[:16], phase='complete' if complete else plan['phase'])
        if not page['next']:
            plan['page_issues'] = 0
        return manager.store.collect_cleanup(replace(run, cleanup='complete' if complete else 'unknown' if unknown
            else 'running', cleanup_json=pinned(plan)), updates)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        plan.update(phase='unknown', issues=[str(error)], next_retry_at=now + 2)
        return manager.store.compare_and_set(replace(run, cleanup='unknown', cleanup_json=pinned(plan)), run.revision)
