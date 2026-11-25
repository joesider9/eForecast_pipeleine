# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Work that runs in a process of its own: prepared, started, watched and collected.

A stage hands a worker its input and reads its result when the worker has exited. In between
the worker is a process like any other: the operating system gives it processor time and
memory, and it writes where the work says. Nothing is reserved for it and nothing is asked
before it starts -- no allocation, no lease, no promise of disk.

What a stage gets from running work this way:

* the memory of the work (a network, a study of many fits) goes back to the machine when the
  process exits, whatever the work did with it;
* the work can be stopped without stopping the stage, and it is stopped when the stage itself
  goes away: nobody is waiting for it then, and a worker that kept writing into a case would
  meet the next run of that case. The case's lock is the one guard between two runs, and it is
  held by the stage, so a worker must not outlive it;
* its output goes to a log of its own, which is named when the worker starts.

If the machine does not have what the work needs, the work fails the way any process does --
an allocation error, or the kernel ending the largest process -- and ``failure_reason`` says
which, with the log.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from eforecast.common_utils import file_stamps
from eforecast.compute_worker import WorkerContext, atomic_bytes, check_inputs
from eforecast import execution_policy
from .executors import LocalExecutor, atomic_json, digest, observe_process
from . import run_reporting
from eforecast.training.command_line import resource_reporting

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Work:
    """One start of one task of a run: its names, its threads and, if it has one, its card.

    ``threads`` is what the worker's libraries are told to use. ``device`` is a GPU or MIG
    UUID, and the worker sees that card only. Neither is taken from anybody.
    """
    run_id: str
    task_id: str
    attempt_id: str
    threads: int = 1
    device: str | None = None

    def __post_init__(self):
        for name in ('run_id', 'task_id', 'attempt_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name} must be a nonempty name')
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError('work needs at least one thread')
        if self.device is not None and not str(self.device).startswith(('GPU-', 'MIG-')):
            raise ValueError('a device is named by its GPU/MIG UUID, never by an ordinal')


def killed_by(returncode):
    """What a negative exit status says, for whoever reads the failure."""
    if not isinstance(returncode, int) or returncode >= 0:
        return None
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        name = f'signal {-returncode}'
    if -returncode == signal.SIGKILL:
        return (f'it was killed ({name}); when nobody stopped it, that is the kernel ending '
                'the largest process because the machine ran out of memory')
    return f'it was ended by {name}'


def last_error(log):
    """The last exception a worker wrote to ``log``, or its last line; ``None`` when it cannot be read."""
    try:
        lines = Path(log).read_text(errors='replace').splitlines()
    except (OSError, TypeError):
        return None
    start = max((i for i, line in enumerate(lines) if line.startswith('Traceback (most recent call last)')), default=None)
    said = [line for line in (lines if start is None else lines[start + 1:]) if line.strip()]
    # After the last traceback's frames, which are indented, the first line that is not is the
    # exception. Output the worker had buffered can follow it, so the last line is not it.
    last = (next((line.strip() for line in said if not line.startswith((' ', '\t'))), None) if start is not None
            else said[-1].strip() if said else None)
    return last[:400] if last else 'it wrote nothing'


def failure_reason(result):
    """What a worker that did not succeed said last, and where the rest of it is.

    A worker's traceback goes to its own log, and the stage that waited for it reported
    "failure; attempt log retained": neither the reason nor the log. A clustering stage that
    had nothing to fit on read exactly like a machine that had run out of memory. A worker
    the kernel ended wrote no reason at all, so its exit status is read for one.
    """
    try:
        path = Path(result['context']['result_directory']) / 'worker.log'
    except (KeyError, TypeError):
        path = None
    reason = (result.get('detail') or killed_by(result.get('returncode'))
              or (last_error(path) if path is not None else None))
    if reason is None:
        return f"{result.get('state', 'failure')}; its log could not be read"
    return f"{result.get('state', 'failure')}: {reason} (worker log: {path})"


def report(spec, *, process=None, outcome=None):
    contract = spec['compute']
    ctx = contract['reporting']
    if ctx is None:
        return
    context = contract['context']
    execution = {'launch_id': context['launch_id'], 'executor_id': spec['token']}
    if 'policy' in contract:
        execution['policy'] = contract['policy']
    if process is not None:
        execution['process'] = process
    run_reporting.append({'type': 'child', 'node_id': digest({'attempt': ctx['attempt_id'],
        'parent': ctx['attempt_id'], 'kind': 'compute',
        'key': contract.get('report_key', context['attempt_id'])}),
        'parent_id': ctx['attempt_id'], 'kind': 'compute', 'identity': context,
        'execution': execution, 'outcome': outcome}, context_override=ctx,
        directory=contract['report_directory'])


def stamp(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def artifact_files(root, paths, *, stamps=None):
    """Hash what a worker left under ``artifacts/``; never follow a symlink."""
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError('compute artifacts must be a list of relative file paths')
    records = {}
    for name in paths:
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or len(relative.parts) < 2
                or relative.parts[0] != 'artifacts' or relative.as_posix() != name):
            raise ValueError('compute artifact must be below its private artifacts/ directory')
        path = root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError('compute artifact cannot contain a symlink')
        if not path.is_file():
            raise ValueError('named compute artifact is not a regular file')
        checksum = hashlib.sha256()
        # Stamps, not bare stats: the worker wrote this file a moment ago, and in a synced
        # folder that is when the sync client marks it as uploaded -- its change time moves
        # and its content does not (``file_stamps``).
        with path.open('rb') as stream:
            before = file_stamps.stamp(stream.fileno())
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(block)
            if not file_stamps.same(before, file_stamps.stamp(stream.fileno())):
                raise ValueError('compute artifact changed while being hashed')
        if not file_stamps.same(before, file_stamps.stamp(path, follow=False)):
            raise ValueError('compute artifact was replaced while being hashed')
        records[name] = {'bytes': before[2], 'sha256': checksum.hexdigest()}
        if stamps is not None:
            stamps[name] = tuple(before[:5])
    return records


def followable(task_id):
    """A name for the task that a person reading a terminal recognises.

    Task identifiers are built for machines, not for people: a dataset stage is
    ``data:<scope>:<stage>`` and a CPU study is the path of its method. The part worth
    printing is the last one in either case.
    """
    tail = task_id.rsplit(':', 1)[-1]
    return Path(tail).name if '/' in tail else tail


class ComputeWorkers:
    """Prepare, submit, collect and cancel work that runs in its own process.

    ``prepare`` writes what the worker reads and returns the handle of a process that does
    not exist yet; ``submit`` starts it; ``collect`` says where it is and, once its whole
    process tree has exited, what it returned. Each prepared piece of work has a folder of its
    own under the executor's directory, named by an identifier that no other start shares.
    """
    def __init__(self, *, executor=None):
        self.executor = executor or LocalExecutor()
        self._announced = set()         # launches whose log file has already been named

    @staticmethod
    def request(run_id, task_id, attempt_id, threads=1, *, device=None):
        return Work(str(run_id), str(task_id), str(attempt_id), max(1, int(threads)), device)

    def prepare(self, request, entrypoint, payload, *, execution=None, input_files=None):
        input_files = input_files or {}
        for name, value in input_files.items():
            if not name or Path(name).name != name or name in {'.', '..'} or not isinstance(value, bytes):
                raise ValueError('compute inputs need simple file names and bytes')
        records = {name: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
                   for name, value in input_files.items()}
        if not isinstance(entrypoint, str) or len(entrypoint.split(':')) != 2:
            raise ValueError('compute entrypoint must be module:function')
        policy = execution_policy.resolve(request.threads, request.device, execution)
        launch = uuid.uuid4().hex
        root = self.executor.directory / 'compute' / launch
        root.mkdir(parents=True, mode=0o700)
        if records:
            (root / 'inputs').mkdir()
            for name, value in input_files.items():
                atomic_bytes(root / 'inputs' / name, value)
        context = WorkerContext(request.run_id, request.task_id, request.attempt_id, launch,
                                request.threads, request.device, str(root))
        document = {'version': 1, 'context': asdict(context), 'entrypoint': entrypoint,
                    'payload': payload, 'policy': policy}
        if records:
            document['input_files'] = records
        atomic_json(root / 'input.json', document)
        compute = {'version': 1, 'context': asdict(context), 'input_digest': digest(document),
                   'policy': policy,
                   'report_key': digest({name: getattr(context, name) for name in
                                         ('run_id', 'task_id', 'attempt_id')}),
                   'reporting': run_reporting.context(),
                   'report_directory': os.environ.get('EFORECAST_RUN_REPORT_DIR')}
        environment = dict(os.environ)
        environment.pop('EFORECAST_COMPUTE_ATTACHED', None)
        environment['EFORECAST_RUN_ID'] = request.run_id
        environment.update(execution_policy.environment(policy))
        return self.executor.plan(request.run_id, request.task_id, request.attempt_id,
            [sys.executable, '-m', 'eforecast.compute_worker', str(root / 'input.json')], environment,
            Path(__file__).resolve().parents[2], root / 'worker.log', compute=compute)

    def _report(self, handle, *, outcome=None):
        _, spec = self.executor._spec(handle)
        report(spec, outcome=outcome)

    def submit(self, handle):
        # Journal before launching: whoever cleans up after this run can find the worker even
        # if this process never hears that its supervisor started.
        self._report(handle)
        self._announce(handle)
        # Watched: the worker's supervisor stops its tree when this process goes away.
        return self.executor.submit(handle, watch=True)

    def _announce(self, handle):
        """Name the file the work's own output goes to, once per launch.

        The worker is a separate process whose standard output and error are redirected
        to a private log; this process prints only the lines it writes itself. Somebody
        watching a terminal therefore sees a stage begin and then nothing at all, with no
        way to reach that file except by walking two directories named after hashes. A
        submission can be repeated for an already running tree, so this says it once.
        """
        if handle.id in self._announced:
            return
        self._announced.add(handle.id)
        try:
            _, spec = self.executor._spec(handle)
            log.info('%s runs in its own process; its output goes to %s',
                     followable(spec['task_id']), spec['log'])
        except KeyError:
            pass        # an older specification without a log; the launch still proceeds

    def cancel(self, handle, *, force=False):
        return self.executor.cancel(handle, force=force)

    def pid(self, handle):
        """The worker's process, while it can be observed; ``None`` otherwise."""
        try:
            evidence = json.loads(self.executor.discover(handle).evidence or '{}')
        except (ValueError, TypeError):
            return None
        return evidence.get('pid')

    def root(self, handle):
        """The worker's own folder: its input, its log, its result and its ``artifacts/``."""
        _, spec = self.executor._spec(handle)
        return Path(spec['compute']['context']['result_directory'])

    def discard_result(self, handle):
        """Remove a worker's own folder once its whole process tree has exited.

        The executor's small record of the launch stays; this never deletes anything of a case.
        """
        intent, _ = self.executor._spec(handle)
        cleanup = self.executor._receipt(Path(intent['directory']), 'tree_cleanup.json', handle, intent)
        if not cleanup or not cleanup.get('complete'):
            return False
        root = self.root(handle)
        if root.exists():
            shutil.rmtree(root)
        return True

    def collect(self, handle):
        observed = self.executor.collect(handle)
        intent, spec = self.executor._spec(handle)
        folder = Path(intent['directory'])
        cleanup = self.executor._receipt(folder, 'tree_cleanup.json', handle, intent)
        if observed.state != 'finished' or not cleanup or not cleanup.get('complete'):
            return {'state': observed.state if observed.state != 'finished' else 'cleaning', 'detail': observed.detail}
        receipt = self.executor._receipt(folder, 'compute_completion.json', handle, intent)
        if receipt is None:
            # The supervisor says its tree has exited before it reads the worker's result and
            # writes it down, so for a moment there is one receipt without the other. Read at
            # that moment, a study that had just succeeded was reported as lost. While the
            # supervisor is there it is still writing; once it is gone, what it wrote is final.
            started = self.executor._receipt(folder, 'started.json', handle, intent)
            if started and observe_process(started['process']) == 'alive':
                return {'state': 'cleaning', 'detail': 'the supervisor is recording the result'}
            receipt = self.executor._receipt(folder, 'compute_completion.json', handle, intent)
        if receipt is None:
            # Its supervisor could not write the result down, and said why if it could say
            # anything; otherwise it was ended from outside. What the worker wrote is in its log.
            said = self.executor._receipt(folder, 'compute_error.json', handle, intent)
            return {'state': 'unknown', 'context': spec['compute']['context'],
                    'detail': (said or {}).get('detail') or 'the worker\'s supervisor ended without recording a result'}
        result = receipt['result']
        if result['state'] == 'success' and result.get('files'):
            try:
                if artifact_files(self.root(handle), list(result['files'])) != result['files']:
                    raise ValueError('a compute artifact no longer matches what the worker left')
            except (OSError, ValueError) as error:
                return {'state': 'unknown', 'detail': str(error)}
        self._report(handle, outcome={'kind': result['state']})
        return result

    def wait(self, handle, *, timeout=120):
        deadline = float('inf') if timeout is None else time.monotonic() + timeout
        self.submit(handle)
        while time.monotonic() < deadline:
            result = self.collect(handle)
            if result['state'] in {'success', 'failure', 'cancelled'}:
                process = self.executor.processes.get(handle.id)
                if process is not None:
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        log.warning('compute supervisor %s has not exited after tree cleanup', handle.id)
                    self.executor.collect(handle)
                return result
            try:
                run_reporting.check_cancelled()
            except BaseException:
                self.cancel(handle, force=True)
                raise
            time.sleep(.05)
        self.cancel(handle, force=True)
        raise TimeoutError('compute worker timed out and was told to stop')


def graph_preview(document, tag, method=None):
    """Interactive preview of a network's graph; FastAPI never loads the network itself.

    A fresh process each time, so that no Torch or model cache stays resident in the API.
    """
    from .graph_service import normalize_experiments
    from .service import ServiceError
    if tag not in normalize_experiments(document.get('experiments') or {}):
        raise ServiceError(f'experiment {tag!r} is not defined inline in the case')
    worker = ComputeWorkers()
    ident = uuid.uuid4().hex
    from .model_compute import preview_estimate
    estimate = preview_estimate(document, tag, method)
    request = worker.request('preview:' + ident, 'graph-preview:' + ident, ident)
    handle = worker.prepare(request, 'eforecast.config.graph_service:compute_preview',
                            {'document': document, 'tag': tag, 'method': method, 'estimate': estimate})
    result = worker.wait(handle)
    if result['state'] != 'success':
        raise ServiceError('graph preview worker failed: ' + failure_reason(result))
    # Preview's response contains its whole payload; there are no model artifacts to retain.
    try:
        worker.discard_result(handle)
    except OSError as error:
        log.warning('graph preview %s completed; its folder was not removed: %s', ident, error)
    return result['value']


class Completion:
    """The supervisor's side of one worker: check its input, then record how it ended.

    No estimator or target is imported in the supervisor's process.
    """
    def __init__(self, spec):
        self.spec = spec
        self.contract = spec['compute']
        self.context = self.contract['context']
        self.root = Path(self.context['result_directory'])

    def before_spawn(self):
        document = json.loads((self.root / 'input.json').read_text())
        if digest(document) != self.contract['input_digest'] or document['context'] != self.context:
            raise ValueError('compute input changed after it was prepared')
        check_inputs(self.root, document.get('input_files', {}))
        policy = execution_policy.validate(document['policy'], self.context['threads'], self.context['device_uuid'])
        if self.contract['policy'] != policy:
            raise ValueError('compute policy changed after it was prepared')
        from .executors import process_evidence
        report(self.spec, process=process_evidence(os.getpid(), self.spec['location']))

    def finish(self, returncode, *, cancelled=False, detail=None):
        """Called once the worker's whole process tree has exited.

        ``detail`` is why a worker that never started did not: its input had changed, or the
        interpreter could not be started.
        """
        state = 'cancelled' if cancelled else 'success' if returncode == 0 else 'failure'
        result = {'state': state, 'context': self.context, 'returncode': returncode,
                  'execution_policy': self.contract['policy']}
        if state == 'failure' and detail:
            result['detail'] = f'the worker was not started: {detail}'
        try:
            prerequisite = self.root / 'prerequisite.json'
            if state == 'failure' and prerequisite.is_file():
                if prerequisite.is_symlink() or prerequisite.stat().st_size > 65536:
                    raise ValueError('invalid compute prerequisite record')
                document = json.loads(prerequisite.read_text())
                if (document.get('version') != 1 or document.get('context') != self.context
                        or not isinstance(document.get('prerequisite'), dict)):
                    raise ValueError('compute prerequisite has a foreign identity or shape')
                result['prerequisite'] = document['prerequisite']
            if state == 'success':
                source = self.root / 'worker_result.json'
                if source.stat().st_size > 16 * 1024 ** 2:
                    raise ValueError('compute JSON result exceeds 16 MiB; publish large arrays as artifacts')
                document = json.loads(source.read_text())
                if (document.get('version') != 1 or document.get('context') != self.context
                        or not isinstance(document.get('value'), dict)):
                    raise ValueError('the worker\'s result has a foreign identity or shape')
                result['value'] = document['value']
                result['files'] = artifact_files(self.root, document['value'].get('artifacts', []))
        except (OSError, ValueError, KeyError, TypeError) as error:
            # The worker exited, so there is something definite to say: what it left cannot
            # be read as a result. That is a failure of this attempt, not an unknown.
            result = {'state': 'failure', 'context': self.context, 'returncode': returncode,
                      'execution_policy': self.contract['policy'],
                      'detail': f'the worker\'s result could not be read: {type(error).__name__}: {error}'}
        atomic_json(self.root / 'result.json', result)
        report(self.spec, outcome={'kind': result['state']})
        return result
