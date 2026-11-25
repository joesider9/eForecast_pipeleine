# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Versioned durable run aggregates. No scheduling or process-liveness policy lives here.

One revision protects the run and all its tasks/attempts in a single transaction. Pinned JSON
is stored as canonical text so frozen dataclasses cannot conceal mutable request dictionaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import uuid

from eforecast.config.service import ServiceError


class InvalidRun(ServiceError):
    pass


def identity() -> str:
    return uuid.uuid4().hex


def pinned(value: dict) -> str:
    if not isinstance(value, dict):
        raise InvalidRun('pinned run input must be a JSON object')
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise InvalidRun('run input must contain finite JSON values') from error


@dataclass(frozen=True)
class Outcome:
    kind: str                   # an observation; cancellation intent is independent
    observed_at: float
    returncode: int | None = None
    detail: str = ''


@dataclass(frozen=True)
class CancelRequest:
    id: str
    requested_at: float
    requested_by: str
    reason: str = ''


@dataclass(frozen=True)
class ExecutorHandle:
    id: str
    adapter: str
    identity_json: str           # exact executor evidence, interpreted by A09-03 adapters


@dataclass(frozen=True)
class Attempt:
    id: str
    number: int
    state: str = 'pending'
    executor: ExecutorHandle | None = None
    outcome: Outcome | None = None
    cleanup: str = 'not_required'
    execution_json: str | None = None  # verified launch observation, pinned once independently of intent
    completion_json: str | None = None  # logical stage content receipt; needs no separate executor


@dataclass(frozen=True)
class Task:
    id: str
    stage: str
    scope: str
    request_json: str
    dependencies: tuple[str, ...] = ()
    state: str = 'pending'
    attempts: tuple[Attempt, ...] = ()
    outcome: Outcome | None = None
    cleanup: str = 'not_required'


@dataclass(frozen=True)
class Run:
    id: str
    owner_id: str
    case_id: str
    request_json: str
    created_at: float
    snapshot_json: str | None = None   # may be pinned once after durable submission
    metadata_json: str = '{}'
    state: str = 'pending'
    revision: int = 0
    tasks: tuple[Task, ...] = ()
    outcome: Outcome | None = None
    cancel: CancelRequest | None = None
    cleanup: str = 'not_required'
    deleted: bool = False             # history removal is a revisioned tombstone
    preparation_json: str | None = None
    version: int = 1
    controller_json: str | None = None
    report_offset: int = 0
    report_error: str | None = None
    result_json: str | None = None
    collection_json: str | None = None
    cleanup_json: str | None = None  # resumable observations/deadlines; independent of outcome
    dispatch_json: str | None = None  # of runs accepted when runs were queued for slots; unused since

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'), allow_nan=False)

    @classmethod
    def from_json(cls, document: str) -> Run:
        try:
            raw = json.loads(document)
            tasks = []
            for task in raw.pop('tasks'):
                attempts = []
                for attempt in task.pop('attempts'):
                    if attempt['executor'] is not None:
                        attempt['executor'] = ExecutorHandle(**attempt['executor'])
                    if attempt['outcome'] is not None:
                        attempt['outcome'] = Outcome(**attempt['outcome'])
                    attempts.append(Attempt(**attempt))
                task['attempts'] = tuple(attempts)
                task['dependencies'] = tuple(task['dependencies'])
                if task['outcome'] is not None:
                    task['outcome'] = Outcome(**task['outcome'])
                tasks.append(Task(**task))
            raw['tasks'] = tuple(tasks)
            if raw['outcome'] is not None:
                raw['outcome'] = Outcome(**raw['outcome'])
            if raw['cancel'] is not None:
                raw['cancel'] = CancelRequest(**raw['cancel'])
            run = cls(**raw)
            validate(run)
            return run
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise InvalidRun('invalid or unsupported durable run record') from error


TRANSITIONS = {
    'pending': {'preparing', 'queued', 'running', 'terminal', 'unknown'},
    'preparing': {'queued', 'running', 'terminal', 'unknown'},
    'queued': {'running', 'terminal', 'unknown'},
    'running': {'terminal', 'unknown'},
    'unknown': {'running', 'terminal'},
    'terminal': set(),
}
CLEANUP = {'not_required': {'pending'}, 'pending': {'running', 'unknown', 'complete'},
           'running': {'pending', 'unknown', 'complete'}, 'unknown': {'pending', 'running', 'complete'},
           'complete': set()}


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise InvalidRun('run identities and labels must be nonempty strings')


def _time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise InvalidRun('run timestamps must be finite nonnegative numbers')


def _json(value):
    if not isinstance(value, str) or pinned(json.loads(value)) != value:
        raise InvalidRun('run inputs must be canonical JSON objects')


def _execution(item):
    if item.state not in TRANSITIONS or item.cleanup not in CLEANUP:
        raise InvalidRun('unknown execution or cleanup state')
    if (item.state == 'terminal') != (item.outcome is not None):
        raise InvalidRun('terminal state requires an outcome; other states cannot assert one')
    if item.outcome is not None:
        if item.outcome.kind not in {'success', 'failure', 'cancelled', 'partial', 'nothing_to_do'}:
            raise InvalidRun('unknown run outcome')
        _time(item.outcome.observed_at)
        if not isinstance(item.outcome.detail, str):
            raise InvalidRun('outcome detail must be text')
        code = item.outcome.returncode
        if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
            raise InvalidRun('returncode must be an observed integer or null')


def _terminal_children(parent, children):
    # Unknown remains legal: execution outcome and unresolved cleanup are separate facts.
    if parent.state == 'terminal' and any(child.state in {'preparing', 'queued', 'running'} for child in children):
        raise InvalidRun('terminal run/task cannot contain an active child execution')


def validate(run: Run):
    try:
        if type(run.version) is not int or run.version != 1:
            raise InvalidRun('unsupported run record version')
        if type(run.revision) is not int or run.revision < 0 or type(run.deleted) is not bool:
            raise InvalidRun('invalid run revision or tombstone')
        for value in (run.id, run.owner_id, run.case_id):
            _text(value)
        _time(run.created_at)
        for value in (run.request_json, run.metadata_json):
            _json(value)
        if run.snapshot_json is not None:
            _json(run.snapshot_json)
        if run.preparation_json is not None:
            _json(run.preparation_json)
        for value in (run.controller_json, run.result_json, run.collection_json, run.cleanup_json, run.dispatch_json):
            if value is not None:
                _json(value)
        if run.cleanup_json is not None and not isinstance(json.loads(run.cleanup_json), dict):
            raise InvalidRun('cleanup observations must be a mapping')
        if type(run.report_offset) is not int or run.report_offset < 0:
            raise InvalidRun('report offset must be a nonnegative integer')
        if run.report_error is not None and not isinstance(run.report_error, str):
            raise InvalidRun('report error must be text or null')
        _execution(run)
        if run.cancel is not None:
            for value in (run.cancel.id, run.cancel.requested_by):
                _text(value)
            _time(run.cancel.requested_at)
            if not isinstance(run.cancel.reason, str):
                raise InvalidRun('cancel reason must be text')
        if not isinstance(run.tasks, tuple):
            raise InvalidRun('tasks must be immutable tuples')
        task_ids, attempt_ids, handles = set(), set(), set()
        for task in run.tasks:
            for value in (task.id, task.stage, task.scope):
                _text(value)
            if task.id in task_ids:
                raise InvalidRun('duplicate task identity')
            task_ids.add(task.id)
            _json(task.request_json)
            _execution(task)
            if not isinstance(task.dependencies, tuple) or not isinstance(task.attempts, tuple):
                raise InvalidRun('dependencies and attempts must be immutable tuples')
            numbers = set()
            for attempt in task.attempts:
                _text(attempt.id)
                if attempt.id in attempt_ids or type(attempt.number) is not int or attempt.number < 0 or attempt.number in numbers:
                    raise InvalidRun('duplicate or invalid attempt identity/number')
                attempt_ids.add(attempt.id)
                numbers.add(attempt.number)
                _execution(attempt)
                if attempt.execution_json is not None:
                    _json(attempt.execution_json)
                    if attempt.executor is None:
                        raise InvalidRun('execution observation requires an executor handle')
                if attempt.completion_json is not None:
                    _json(attempt.completion_json)
                if attempt.executor is not None:
                    handle = attempt.executor
                    _text(handle.id)
                    _text(handle.adapter)
                    _json(handle.identity_json)
                    if handle.id in handles:
                        raise InvalidRun('executor handle belongs to more than one attempt')
                    handles.add(handle.id)
            _terminal_children(task, task.attempts)
        _terminal_children(run, run.tasks)
        _terminal_children(run, (attempt for task in run.tasks for attempt in task.attempts))
        from .stage_workflow import validate_binding
        validate_binding(run)
        graph = {task.id: task.dependencies for task in run.tasks}
        remaining = set(graph)
        while remaining:
            ready = {key for key in remaining if all(dep in graph and dep not in remaining for dep in graph[key])}
            if not ready:
                raise InvalidRun('missing or cyclic task dependencies')
            remaining -= ready
    except (TypeError, AttributeError, ValueError) as error:
        if isinstance(error, InvalidRun):
            raise
        raise InvalidRun('invalid durable run value') from error


def _unchanged(old, new, fields):
    if any(getattr(old, key) != getattr(new, key) for key in fields):
        raise InvalidRun('immutable run identity or pinned input changed')


def _advance(old, new):
    if old.state != new.state and new.state not in TRANSITIONS[old.state]:
        raise InvalidRun(f'invalid execution transition {old.state} -> {new.state}')
    if old.cleanup != new.cleanup and new.cleanup not in CLEANUP[old.cleanup]:
        raise InvalidRun(f'invalid cleanup transition {old.cleanup} -> {new.cleanup}')
    if old.outcome is not None and old.outcome != new.outcome:
        raise InvalidRun('observed outcome is immutable')


def transition(old: Run, new: Run) -> Run:
    """Validate a proposed aggregate; the store alone advances its revision."""
    validate(new)
    _unchanged(old, new, ('id', 'owner_id', 'case_id', 'request_json', 'created_at', 'metadata_json', 'version'))
    if old.snapshot_json is not None and old.snapshot_json != new.snapshot_json:
        raise InvalidRun('snapshot is already pinned')
    if old.dispatch_json is not None and old.dispatch_json != new.dispatch_json:
        raise InvalidRun('dispatch ownership is already pinned')
    if new.report_offset < old.report_offset:
        raise InvalidRun('report collection cannot move backwards')
    if old.cleanup_json is not None and new.cleanup_json is None:
        raise InvalidRun('cleanup observations cannot be forgotten')
    if old.cleanup_json is not None and new.cleanup_json is not None:
        before, after = json.loads(old.cleanup_json), json.loads(new.cleanup_json)
        for key in ('started_at', 'term_deadline', 'monotonic_deadline'):
            if key in before and after.get(key) != before[key]:
                raise InvalidRun('cleanup escalation deadline is already pinned')
        if before.get('force') and not after.get('force'):
            raise InvalidRun('cleanup escalation cannot move backwards')
    for key in ('result_json', 'collection_json'):
        if getattr(old, key) is not None and getattr(old, key) != getattr(new, key):
            raise InvalidRun(f'{key} is already pinned')
    if old.cancel is not None and old.cancel != new.cancel:
        raise InvalidRun('cancel request is immutable')
    if old.deleted and not new.deleted:
        raise InvalidRun('a deleted run cannot be restored implicitly')
    if old.preparation_json != new.preparation_json:
        if new.state != 'preparing' or any(attempt.executor is not None or attempt.state == 'unknown'
                for task in old.tasks for attempt in task.attempts):
            raise InvalidRun('preparation ownership can only change before launch')
    _advance(old, new)
    new_tasks = {task.id: task for task in new.tasks}
    if old.state == 'terminal':
        if set(new_tasks) - {task.id for task in old.tasks}:
            raise InvalidRun('tasks cannot be added to a terminal run')
        recorded_attempts = {attempt.id for task in old.tasks for attempt in task.attempts}
        if any(attempt.id not in recorded_attempts for task in new.tasks for attempt in task.attempts):
            raise InvalidRun('attempts cannot be added to a terminal run')
    for task in old.tasks:
        if task.id not in new_tasks:
            raise InvalidRun('recorded tasks cannot be removed')
        after = new_tasks[task.id]
        _unchanged(task, after, ('stage', 'scope', 'request_json', 'dependencies'))
        _advance(task, after)
        new_attempts = {attempt.id: attempt for attempt in after.attempts}
        for attempt in task.attempts:
            if attempt.id not in new_attempts:
                raise InvalidRun('recorded attempts cannot be removed')
            successor = new_attempts[attempt.id]
            _unchanged(attempt, successor, ('number',))
            if attempt.executor is not None and attempt.executor != successor.executor:
                raise InvalidRun('executor handle is already pinned')
            if attempt.execution_json is not None and attempt.execution_json != successor.execution_json:
                raise InvalidRun('execution identity is already pinned')
            if attempt.completion_json is not None and attempt.completion_json != successor.completion_json:
                raise InvalidRun('completion receipt is already pinned')
            _advance(attempt, successor)
    return replace(new, revision=old.revision + 1)
