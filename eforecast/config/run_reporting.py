# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Small append-only managed-run journal shared by CLI, CPU and Docker workers.

Unmanaged CLI calls are no-ops. Readers checkpoint offsets with their database changes;
event identities and child merges make duplicate/out-of-order delivery harmless.
"""
from contextlib import contextmanager
import hashlib
import json
import logging
import os
import time
import math
from pathlib import Path

from eforecast.config.run_model import pinned
from eforecast.folder_lock import folder_lock as file_critical_section

# One definition of the variable names, in a module the neutral side can reach without this
# one's A09 imports. See eforecast/managed_identity.py.
from eforecast.managed_identity import CONTEXT_ENV, context        # noqa: F401,E402
KINDS = {'success', 'partial', 'failure', 'cancelled', 'nothing_to_do'}
MAX_EVENT = 64 * 1024

log = logging.getLogger(__name__)


def check_cancelled():
    """Stop new managed dispatch/fitting after the durable request reaches this worker."""
    ctx = context()
    if ctx is not None:
        path = Path(os.environ['EFORECAST_RUN_REPORT_DIR']) / 'cancel.json'
        if path.exists() and json.loads(path.read_text()).get('executor_id') == ctx['executor_id']:
            raise KeyboardInterrupt('managed run cancellation requested')


def event_digest(value):
    return hashlib.sha256(pinned(value).encode()).hexdigest()


def node_id(kind, key, parent=None):
    ctx = context()
    return event_digest({'attempt': ctx['attempt_id'], 'parent': parent or ctx['attempt_id'],
                         'kind': kind, 'key': str(key)}) if ctx else None


def append(value, *, context_override=None, directory=None):
    # Host-side supervised children can report to their pinned parent even after the
    # submitting process exits. Worker targets continue using their own environment.
    ctx = context_override if context_override is not None else context()
    if ctx is None:
        return
    body = {**value, 'context': ctx, 'version': 1}
    event = {**body, 'event_id': event_digest(body)}
    encoded = (pinned(event) + '\n').encode()
    if len(encoded) > MAX_EVENT:
        raise ValueError('managed run event exceeds the publication limit')
    folder = Path(directory or os.environ['EFORECAST_RUN_REPORT_DIR'])
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with file_critical_section(folder, timeout=2):
        path = folder / 'events.jsonl'
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            size = os.lseek(descriptor, 0, os.SEEK_END)
            if size:
                os.lseek(descriptor, max(0, size - MAX_EVENT), os.SEEK_SET)
                tail = os.read(descriptor, MAX_EVENT)
                if not tail.endswith(b'\n'):
                    last = tail.rfind(b'\n')
                    if last < 0 and size >= MAX_EVENT:
                        raise ValueError('unverifiable journal tail; explicit recovery required')
                    # Only an unfinished append to this private journal is discarded, never
                    # a completed event, case value, or historical artifact.
                    os.ftruncate(descriptor, max(0, size - MAX_EVENT) + last + 1)
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if not written:
                    raise OSError('managed journal write made no progress')
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def child(kind, key, *, parent=None, identity=None, execution=None, outcome=None):
    ctx = context()
    if ctx is None:
        return None
    ident = node_id(kind, key, parent)
    append({'type': 'child', 'node_id': ident, 'parent_id': parent or ctx['attempt_id'],
            'kind': kind, 'identity': identity or {}, 'execution': execution, 'outcome': outcome})
    return ident


def result(kind, detail='', **counts):
    if kind not in KINDS:
        raise ValueError('unknown managed outcome')
    append({'type': 'result', 'result': {'kind': kind, 'detail': detail, 'counts': counts}})


@contextmanager
def stage(plan, key, static_data, completed):
    """Observe the operation and require its own content receipt before reporting success."""
    from .stage_catalogue import require_inputs
    node = next(node for node in plan['nodes'] if node['key'] == key)
    if not node['enabled']:
        yield False
        return
    check_cancelled()
    require_inputs(static_data, plan, key, completed)
    expected = os.environ.get('EFORECAST_STAGE_PLAN_ID') if context() is not None else None
    if expected and expected != plan['identity']:
        raise ValueError('CLI stage catalogue differs from the accepted plan')
    def publish(phase, outcome=None, completion=None):
        if expected:
            from .stage_workflow import stage_ids
            task_id, attempt_id = stage_ids(context()['run_id'], key)
            append({'type': 'stage', 'key': key, 'plan_id': expected, 'phase': phase,
                    'stage_task_id': task_id, 'stage_attempt_id': attempt_id,
                    'observed_at': time.time(), 'outcome': outcome,
                    **({'completion': completion} if completion else {})})
    from .stage_resume import permitted
    if plan.get('completion_version') == 1 and permitted(plan, key):
        # Permission is not proof: the artifact is hashed here, now, and a payload that moved
        # since the plan was pinned is rebuilt like any other.
        from .stage_artifacts import inspect_stage
        evidence = inspect_stage(static_data, key)
        # A payload with no receipt names no producer and no content: there is nothing to reuse,
        # so it takes the ordinary route and publishes its own inspected observation.
        if evidence['status'] in {'verified', 'legacy'} and evidence.get('id') and evidence.get('producer'):
            publish('started')
            publish('finished', {'kind': 'success', 'detail': f'reused the published {key} artifact; '
                                 f'its content and inputs verify and it was not recomputed'},
                    {'id': evidence['id'], 'producer': evidence['producer'], 'status': 'reused'})
            completed.add(key)
            _keep_configuration(static_data, plan, key)
            yield False
            return
        log.warning('%s was planned for reuse but its artifact no longer verifies (%s); rebuilding',
                    key, evidence['reason'])
    try:
        if plan.get('completion_version') == 1 or plan.get('script') == 'datasets':
            from .direct_stage import DirectStage       # one writer, in place
            with DirectStage(static_data, plan, key) as summary:
                publish('started')
                yield summary
            if plan.get('completion_version') != 1:
                # The worker may normalize execution into the safe private stage
                # boundary without upgrading an already accepted observation contract.
                summary.pop('completion', None)
                summary['detail'] = 'legacy operation observation; artifact readiness is unverified'
        else:
            from types import SimpleNamespace
            from .stage_workspace import Observation
            summary = Observation(SimpleNamespace(static=static_data, rebase=lambda value, **kw: value))
            summary['detail'] = 'legacy operation observation; artifact readiness is unverified'
            publish('started')
            yield summary
    except BaseException as error:
        publish('finished', {'kind': 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failure',
                             'detail': f'{type(error).__name__}: {error}'})
        raise
    else:
        publish('finished', {k: v for k, v in summary.items() if k != 'completion'}, summary.get('completion'))
        completed.add(key)
        if summary.get('kind') in {'success', 'nothing_to_do', 'partial'}:
            _keep_configuration(static_data, plan, key)


def _keep_configuration(static_data, plan, key):
    """The model now holds this stage as this run's configuration describes it: keep that configuration.

    One copy, in the model's own folder (``model_configuration``). Never a reason for a stage
    that has finished to be reported as anything else.
    """
    try:
        from . import model_configuration
        model_configuration.record(static_data, plan.get('script'), key)
    except Exception as error:                                   # noqa: BLE001
        log.debug('the configuration of the model was not recorded after %s (%r)', key, error)


@contextmanager
def operation(script):
    """Managed entrypoints publish a failure on exception and success unless overridden."""
    summary = {'kind': 'success', 'detail': script, 'counts': {}}
    try:
        check_cancelled()
        yield summary
    except BaseException as error:
        result('cancelled' if isinstance(error, KeyboardInterrupt) else 'failure',
               f'{script}: {type(error).__name__}: {error}')
        raise
    else:
        result(summary['kind'], summary['detail'], **summary['counts'])


def read_batch(directory, offset, expected_context, *, limit=64):
    """Bounded locked read; returns complete events, next offset, EOF, unfinished tail."""
    if not 1 <= limit <= 256:
        raise ValueError('report batch limit must be 1–256')
    folder = Path(directory)
    path = folder / 'events.jsonl'
    if not path.exists():
        if offset:
            raise ValueError('previously collected journal is missing')
        return [], 0, True, False
    events = []
    with file_critical_section(folder, timeout=2), path.open('rb') as stream:
        if os.fstat(stream.fileno()).st_size < offset:
            raise ValueError('journal is shorter than its committed offset')
        os.fsync(stream.fileno())
        stream.seek(offset)
        for _ in range(limit):
            line = stream.readline(MAX_EVENT + 1)
            if not line:
                return events, offset, True, False
            if len(line) > MAX_EVENT:
                raise ValueError('managed event exceeds read limit')
            if not line.endswith(b'\n'):
                return events, offset, True, True
            event = json.loads(line)
            validate_event(event, expected_context)
            events.append(event)
            offset = stream.tell()
        return events, offset, stream.read(1) == b'', False


def validate_event(event, expected_context):
    body = {key: value for key, value in event.items() if key != 'event_id'}
    if event['event_id'] != event_digest(body) or event['context'] != expected_context or event['version'] != 1:
        raise ValueError('foreign or corrupt managed event')
    if event['type'] == 'result':
        value = event['result']
        if value['kind'] not in KINDS or not isinstance(value['detail'], str) or not isinstance(value['counts'], dict):
            raise ValueError('invalid structured result')
    elif event['type'] == 'stage':
        if (not isinstance(event.get('key'), str) or not event['key'] or
                not isinstance(event.get('plan_id'), str) or not event['plan_id'] or
                event.get('phase') not in {'started', 'finished'}):
            raise ValueError('invalid stage observation')
        if not all(isinstance(event.get(key), str) and event[key] for key in ('stage_task_id', 'stage_attempt_id')):
            raise ValueError('missing stage task/attempt identity')
        observed = event.get('observed_at')
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed) or observed < 0:
            raise ValueError('invalid stage observation time')
        value = event.get('outcome')
        completion = event.get('completion')
        if completion is not None and (event['phase'] != 'finished' or not isinstance(completion, dict) or
                completion.get('status') not in {'verified', 'legacy', 'reused'} or not isinstance(completion.get('id'), str)
                or not isinstance(completion.get('producer'), dict)):
            raise ValueError('invalid stage completion reference')
        if event['phase'] == 'finished':
            if not isinstance(value, dict) or value.get('kind') not in KINDS or not isinstance(value.get('detail'), str):
                raise ValueError('invalid stage outcome')
        elif value is not None:
            raise ValueError('started stage cannot assert an outcome')
    elif event['type'] == 'child':
        for key in ('node_id', 'parent_id', 'kind'):
            if not isinstance(event[key], str) or not event[key]:
                raise ValueError('missing child identity')
        if event['kind'] not in {'orchestrator', 'study', 'trial', 'container', 'compute'}:
            raise ValueError('unknown child kind')
        if not isinstance(event['identity'], dict):
            raise ValueError('child identity must be a mapping')
        if event['execution'] is not None and not isinstance(event['execution'], dict):
            raise ValueError('child execution must be a mapping')
        outcome = event['outcome']
        if outcome is not None and (not isinstance(outcome, dict) or outcome.get('kind') not in KINDS):
            raise ValueError('invalid child outcome')
    else:
        raise ValueError('unknown managed event type')


def merge_child(previous, event):
    incoming = {key: event[key] for key in ('node_id', 'parent_id', 'kind', 'identity', 'execution', 'outcome')}
    if previous is None:
        return incoming
    if 'cleanup' in previous:
        incoming['cleanup'] = previous['cleanup']
    if any(previous[key] != incoming[key] for key in ('node_id', 'parent_id', 'kind')):
        raise ValueError('child identity changed')
    if any(previous['identity'][key] != value for key, value in incoming['identity'].items()
           if key in previous['identity']):
        raise ValueError('child identity changed')
    incoming['identity'] = {**previous['identity'], **incoming['identity']}
    for key in ('execution', 'outcome'):
        if key == 'execution' and previous[key] is not None and incoming[key] is not None:
            if any(previous[key][field] != value for field, value in incoming[key].items() if field in previous[key]):
                raise ValueError('child execution changed')
            incoming[key] = {**previous[key], **incoming[key]}
            continue
        if key == 'outcome' and previous[key] is not None and incoming[key] is not None and previous[key] != incoming[key]:
            raise ValueError('child outcome changed')
        incoming[key] = previous[key] if previous[key] is not None else incoming[key]
    return incoming


def process():
    if context() is None:
        return None
    from eforecast.config.executors import process_evidence
    return {'process': process_evidence(os.getpid())}


def worker_started():
    """The trial runner publishes its own birth/namespace, including inside Docker.

    The parent records the launcher and exact container; their observations merge by
    distinct immutable keys, so a Docker client PID is never mistaken for its fitter.
    """
    check_cancelled()
    key, parent = os.environ.get('EFORECAST_MANAGED_TRIAL_KEY'), os.environ.get('EFORECAST_MANAGED_STUDY_NODE')
    if context() is not None and key and parent:
        return child('trial', key, parent=parent, execution={'worker': process()['process']})
    return None
