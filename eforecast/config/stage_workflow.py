# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Bind catalogue operations to durable tasks and collect their own observations.

The first task remains the supervised entry-point envelope. Stage attempts describe work
inside that owned process; A08 study/trial children retain their own execution identities.
"""
from dataclasses import replace
import hashlib
import json
import re
import time

from .run_model import Attempt, InvalidRun, Outcome, Task, pinned


def _id(run, key):
    return hashlib.sha256(f'{run.id}/stage/{key}'.encode()).hexdigest()


def stage_ids(run_id, key, number=0):
    return (hashlib.sha256(f'{run_id}/stage/{key}'.encode()).hexdigest(),
            hashlib.sha256(f'{run_id}/stage/{key}/attempt/{number}'.encode()).hexdigest())


def validate_binding(run):
    if not run.snapshot_json:
        return
    snapshot = json.loads(run.snapshot_json)
    plan = snapshot.get('stage_plan')
    if plan is None:
        return  # pre-catalogue history remains unchanged/readable
    if 'completion_version' in plan and (type(plan['completion_version']) is not int or plan['completion_version'] != 1):
        raise InvalidRun('unknown stage completion contract version')
    from .stage_catalogue import VERSION, ordered, plan_identity
    if plan['version'] != VERSION or plan['identity'] != plan_identity(plan) or plan['order'] != ordered(plan['nodes']):
        raise InvalidRun('invalid pinned stage catalogue')
    accepted = json.loads(run.request_json)
    if accepted.get('script') is not None and accepted['script'] != plan['script']:
        raise InvalidRun('stage catalogue differs from the accepted entry point')
    if 'effective_options' in accepted:
        from .run_inputs import decode
        if decode(accepted['effective_options']) != plan['options']:
            raise InvalidRun('stage catalogue differs from the accepted options')
    selected = set(plan['execution_order']) if not plan['dry_run'] else set()
    nodes = {node['key']: node for node in plan['nodes'] if node['selected']}
    if not plan['dry_run'] and selected != set(nodes):
        raise InvalidRun('stage execution order disagrees with its catalogue')
    tasks = {_id(run, key): key for key in selected}
    if {task.id for task in run.tasks[1:]} != set(tasks):
        raise InvalidRun('durable tasks disagree with the pinned stage catalogue')
    by_id = {task.id: task for task in run.tasks}
    for task in run.tasks[1:]:
        key = tasks[task.id]
        node = nodes[key]
        request = json.loads(task.request_json)
        expected = {'catalogue_version': plan['version'], 'plan_id': plan['identity'], 'key': key,
                    'contract': node, 'effective_revision': snapshot.get('effective_revision')}
        if request != expected or (task.stage, task.scope) != (key, node['scope']):
            raise InvalidRun('stage task changed its pinned contract')
        dependencies = tuple(_id(run, dep) for dep in node['dependencies'] if dep in selected)
        if task.dependencies != dependencies:
            raise InvalidRun('stage task dependencies disagree with the catalogue')
        if not task.attempts or any(attempt.id != stage_ids(run.id, key, attempt.number)[1] for attempt in task.attempts):
            raise InvalidRun('stage attempt identity differs from its declared task')
        if (plan.get('completion_version') == 1 and node['enabled'] and task.outcome is not None
                and task.outcome.kind in {'success', 'nothing_to_do'}):
            evidence = json.loads(task.attempts[-1].completion_json or '{}').get('stage_completion')
            root = run.tasks[0]
            producer = evidence.get('producer', {}) if isinstance(evidence, dict) else {}
            expected = {'run_id': run.id, 'task_id': root.id, 'attempt_id': root.attempts[-1].id,
                'executor_id': root.attempts[-1].executor.id if root.attempts[-1].executor else None,
                'stage_task_id': task.id, 'stage_attempt_id': task.attempts[-1].id}
            reused = isinstance(evidence, dict) and evidence.get('status') == 'reused'
            if (not isinstance(evidence, dict) or evidence.get('version') != 1 or
                    evidence.get('status') not in {'verified', 'legacy', 'reused'} or
                    not isinstance(evidence.get('id'), str) or not re.fullmatch('[0-9a-f]{64}', evidence['id']) or
                    not isinstance(producer, dict) or
                    # A reused artifact was published by an earlier run, so its producer is an
                    # earlier attempt; the plan this run accepted is what permits it.
                    (not reused and any(producer.get(k) != v for k, v in expected.items())) or
                    (reused and plan.get('resume', {}).get(key) != 'reuse') or
                    type(producer.get('fence')) is not int or producer['fence'] <= 0):
                raise InvalidRun('successful stage requires its own completion receipt')
        if (node['enabled'] and (task.state in {'preparing', 'queued', 'running'} or
                task.outcome is not None and task.outcome.kind in {'success', 'nothing_to_do'})):
            if any(by_id[dep].outcome is None or by_id[dep].outcome.kind not in {'success', 'nothing_to_do'}
                   for dep in dependencies):
                raise InvalidRun('stage execution precedes successful dependencies')
    if run.outcome is not None and run.outcome.kind in {'success', 'nothing_to_do'}:
        # The collector already refuses to call an entry point successful while a declared stage
        # has no successful observation; the aggregate itself refuses to hold such a record, so no
        # other writer can produce one either.
        for task in run.tasks[1:]:
            if task.outcome is None or task.outcome.kind not in {'success', 'nothing_to_do'}:
                raise InvalidRun('a successful run cannot leave a declared stage unobserved')


def bind(run, plan, effective_revision):
    if len(run.tasks) != 1:
        raise InvalidRun('stage tasks can only be bound once to a single execution envelope')
    selected = set(plan['execution_order'])
    tasks = list(run.tasks)
    if plan['dry_run']:
        return run
    for node in plan['nodes']:
        if node['key'] not in selected:
            continue
        key = node['key']
        request = {'catalogue_version': plan['version'], 'plan_id': plan['identity'],
            'key': key, 'contract': node, 'effective_revision': effective_revision}
        outcome = None if node['enabled'] else Outcome('nothing_to_do', time.time(), None, node['skip_reason'])
        state = 'pending' if outcome is None else 'terminal'
        tasks.append(Task(_id(run, key), key, node['scope'], pinned(request),
            dependencies=tuple(_id(run, dep) for dep in node['dependencies'] if dep in selected),
            state=state, outcome=outcome,
            attempts=(Attempt(_id(run, key + '/attempt/0'), 0, state=state, outcome=outcome),)))
    return replace(run, tasks=tuple(tasks))


def execution_progress(run, state, *, outcome=None, executor=None, execution=None):
    """Only the supervised envelope follows process state; never stamp stage outcomes alike."""
    if not run.tasks:
        return replace(run, state=state, outcome=outcome)
    root = run.tasks[0]
    root = replace(root, state=state, outcome=outcome, attempts=tuple(
        replace(attempt, state=state, outcome=outcome,
                executor=executor if executor is not None else attempt.executor,
                execution_json=execution if execution is not None else attempt.execution_json)
        if index == len(root.attempts) - 1 else attempt for index, attempt in enumerate(root.attempts)))
    remaining = []
    for task in run.tasks[1:]:
        if state == 'terminal' and task.state in {'preparing', 'queued', 'running'}:
            # No stage receipt: preserve uncertainty, never infer its parent's success/failure.
            task = replace(task, state='unknown', attempts=tuple(
                replace(attempt, state='unknown') if attempt.state in {'preparing', 'queued', 'running'} else attempt
                for attempt in task.attempts))
        remaining.append(task)
    return replace(run, state=state, outcome=outcome, tasks=(root, *remaining))


def merge(run, event):
    matches = [task for task in run.tasks[1:] if json.loads(task.request_json).get('key') == event['key']]
    if len(matches) != 1:
        raise InvalidRun('stage report does not name a declared stage task')
    task = matches[0]
    request = json.loads(task.request_json)
    if event['plan_id'] != request['plan_id']:
        raise InvalidRun('stage report belongs to a different pinned plan')
    if event['stage_task_id'] != task.id or event['stage_attempt_id'] != task.attempts[-1].id:
        raise InvalidRun('stage report belongs to a different task or attempt')
    if task.state == 'terminal':
        if event['phase'] == 'finished':
            supplied = event['outcome']
            # The decision is what cannot change. A producer that reports the same decision again
            # -- a retried publication, a stage entered twice in one process -- carries a later
            # timestamp and a different event id, and must not fail the whole collection for it.
            if (supplied['kind'], supplied['detail']) != (task.outcome.kind, task.outcome.detail):
                raise InvalidRun('stage already has a different observed outcome')
            evidence = json.loads(task.attempts[-1].completion_json or '{}').get('stage_completion')
            if evidence is not None and event.get('completion') != {k: v for k, v in evidence.items() if k != 'version'}:
                raise InvalidRun('stage already has a different completion receipt')
        return run
    if run.state == 'terminal' and event['phase'] == 'started':
        raise InvalidRun('stage cannot start after its execution envelope finishes')
    by_id = {item.id: item for item in run.tasks}
    if any(by_id[dep].outcome is None or by_id[dep].outcome.kind not in {'success', 'nothing_to_do'}
           for dep in task.dependencies):
        raise InvalidRun('stage report precedes successful dependency observations')
    outcome = None
    state = 'running'
    if event['phase'] == 'finished':
        supplied = event['outcome']
        outcome = Outcome(supplied['kind'], event['observed_at'], None, supplied['detail'])
        state = 'terminal'
    completion = task.attempts[-1].completion_json
    if (state == 'terminal' and outcome.kind in {'success', 'nothing_to_do'} and
            json.loads(run.snapshot_json)['stage_plan'].get('completion_version') == 1):
        from .stage_artifacts import IncompleteStage, verify_completion
        try:
            completion = pinned({'stage_completion': verify_completion(run, task, event)})
        except (IncompleteStage, OSError, KeyError, TypeError, ValueError) as error:
            raise InvalidRun(f'invalid stage completion: {error}') from error
    task = replace(task, state=state, outcome=outcome, attempts=(*task.attempts[:-1],
        replace(task.attempts[-1], state=state, outcome=outcome, completion_json=completion)))
    run = replace(run, tasks=tuple(task if old.id == task.id else old for old in run.tasks))
    return execution_progress(run, 'running') if run.state in {'queued', 'preparing', 'unknown'} else run


def project(run):
    by_id = {task.id: task for task in run.tasks}
    out = []
    for task in run.tasks[1:]:
        request = json.loads(task.request_json)
        if 'catalogue_version' not in request:
            continue
        blocked = request['contract'].get('blocked', [])
        if not blocked and task.state == 'pending':
            blocked = [{'reason': f'waiting for stage {by_id[dep].stage}', 'task_id': dep}
                for dep in task.dependencies if by_id[dep].outcome is None or
                by_id[dep].outcome.kind not in {'success', 'nothing_to_do'}]
        evidence = json.loads(task.attempts[-1].completion_json or '{}').get('stage_completion') if task.attempts else None
        out.append({'id': task.id, 'stage': task.stage, 'scope': task.scope, 'state': task.state,
            'dependencies': list(task.dependencies), 'options': request['contract']['options'],
            'outcome': None if task.outcome is None else {'kind': task.outcome.kind,
                'detail': task.outcome.detail, 'observed_at': task.outcome.observed_at},
            'blocked': blocked, 'readiness': evidence or {'status': 'unverified'},
            'attempt_id': task.attempts[-1].id if task.attempts else None})
    return out
