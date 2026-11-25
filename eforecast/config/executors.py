# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Recoverable execution, separate from admission and run-store state transitions.

Handles are immutable launch intents. Observations and completion receipts belong to that
handle, never to a bare PID. The local lock lives on local disk and is inherited across spawn;
closing the controller's copy must NOT unlock the child's copy of the open file description.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import hmac
import json
import math
import os
import secrets
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Protocol

from eforecast.config.run_model import ExecutorHandle, identity, pinned
from eforecast.training.command_line import liveness
from eforecast.training.command_line.liveness import ProcessIdentity, process_identity


@dataclass(frozen=True)
class Observation:
    state: str                 # absent, created, running, finished, unknown
    evidence: str = '{}'
    returncode: int | None = None
    finished_at: float | None = None
    detail: str = ''


class Executor(Protocol):
    def submit(self, handle: ExecutorHandle) -> Observation: ...
    def discover(self, handle: ExecutorHandle) -> Observation: ...
    def status(self, handle: ExecutorHandle) -> Observation: ...
    def cancel(self, handle: ExecutorHandle, *, force: bool = False) -> Observation: ...
    def collect(self, handle: ExecutorHandle) -> Observation: ...


def digest(document):
    return hashlib.sha256(pinned(document).encode()).hexdigest()


def atomic_json(path, document):
    """Publish a complete private receipt, including directory durability, or raise."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(pinned(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def locked(directory):
    """Nonblocking: None means an execution/dispatch already owns this intent."""
    descriptor = os.open(Path(directory) / 'execution.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
        else:
            yield descriptor
    finally:
        os.close(descriptor)  # no LOCK_UN: a child may hold the SAME open description


def location():
    boot = liveness.boot_id()
    return {'host': liveness.host_id(boot), 'boot_id': boot,
            'pid_namespace': liveness.pid_namespace()}


def same_location(intent):
    here = location()
    return all(here[key] is not None and here[key] == intent['location'].get(key) for key in here)


def process_evidence(pid, owner_location=None):
    here = owner_location or location()
    evidence = asdict(process_identity(pid, here['host'], here['boot_id']))
    evidence['pgid'] = os.getpgid(pid)
    return evidence


def observe_process(evidence):
    # An executor cannot establish ownership of a process without its birth time.
    if evidence.get('start_ticks') is None:
        return 'unknown'
    process = ProcessIdentity(**{key: value for key, value in evidence.items() if key != 'pgid'})
    verdict = liveness.observe(SimpleNamespace(process=process, request=None))[0]
    if verdict == 'alive':
        try:
            if Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[0] in {'Z', 'X'}:
                return 'dead'
        except OSError:
            return 'unknown'
    return verdict


class LocalExecutor:
    adapter = 'local_supervised'

    #: Set to 1/0 to decide explicitly whether this process's workers stop when it does.
    WATCH = 'EFORECAST_STOP_WITH_CONTROLLER'

    @classmethod
    def watching(cls):
        """Whether a whole run this process starts should stop when this process goes away.

        A stage's own workers are always watched (``ComputeWorkers.submit``): nobody is waiting
        for them once the stage is gone. This is about the other thing an executor starts, a
        whole run. A run started by the service has to outlive a restart of the service and a
        closed terminal, so it is detached unless told otherwise.

        Set ``EFORECAST_STOP_WITH_CONTROLLER=1`` where a stop should mean a stop: a debugger run
        configuration, an interactive shell. A run started by the pipeline itself always has a
        managed identity, and is never watched.
        """
        choice = os.environ.get(cls.WATCH)
        if choice is None:
            return False
        if choice.strip().lower() in ('', '0', 'no', 'off', 'false'):
            return False
        from eforecast.config import run_reporting
        return run_reporting.context() is None

    def __init__(self, directory=None, *, workspace=None):
        # Reuse the store's local-filesystem checks: a synced case folder is not a lock service.
        from eforecast.config.run_store import _local_path
        state = Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local' / 'state'))
        self.directory = _local_path(Path(directory or os.environ.get('EFORECAST_RUN_EXECUTOR_DIR')
                                    or state / 'eforecast' / 'executors'), workspace)
        self.processes = {}  # reaping optimization only; never completion/ownership evidence
        # Write ends of the pipes a watched supervisor reads. Only this process holds one, so
        # the kernel closes it however this process ends, and the supervisor sees EOF.
        self.controller_ends = {}

    def plan(self, run_id, task_id, attempt_id, command, environment, cwd, log, *, compute=None):
        token = identity()
        folder = self.directory / token
        folder.mkdir(parents=True, mode=0o700)
        reports = folder / 'reports'
        reports.mkdir(mode=0o700)
        managed = {'run_id': run_id, 'task_id': task_id, 'attempt_id': attempt_id, 'executor_id': token}
        environment = {**environment, **{'EFORECAST_MANAGED_' + key.upper(): value for key, value in managed.items()},
                       'EFORECAST_RUN_REPORT_DIR': str(reports)}
        where = location()
        spec = {'version': 1, 'token': token, 'run_id': run_id, 'task_id': task_id, 'location': where,
                'attempt_id': attempt_id, 'command': command,
                'environment': {**environment, 'EFORECAST_EXECUTOR_ID': token},
                'cwd': str(cwd), 'log': str(log)}
        if compute is not None:
            spec['compute'] = compute
        atomic_json(folder / 'spec.json', spec)
        # The worker inherits the specification's environment and runs the owner's own experiment
        # code. It must not be able to write a receipt for itself: every receipt the supervisor
        # writes is sealed with this secret, which reaches the supervisor through a pipe and is
        # never written to the specification, the environment or the registry directory.
        return ExecutorHandle(token, self.adapter, pinned({'version': 1, 'directory': str(folder),
            'location': where, 'spec_digest': digest(spec), 'command': command, 'seal': secrets.token_hex(32),
            'run_id': run_id, 'task_id': task_id, 'attempt_id': attempt_id,
            'report_directory': str(reports),
            'cleanup_contract': 'subreaper-v1',
            'result_required': environment.get('EFORECAST_RUN_RESULT_REQUIRED') == '1'}))

    @staticmethod
    def _spec(handle):
        intent = json.loads(handle.identity_json)
        spec = json.loads((Path(intent['directory']) / 'spec.json').read_text())
        if (digest(spec) != intent['spec_digest'] or spec['token'] != handle.id or
                any(spec[key] != intent[key] for key in ('run_id', 'task_id', 'attempt_id'))):
            raise ValueError('execution specification disagrees with pinned launch intent')
        return intent, spec

    @staticmethod
    def sealed(seal, receipt):
        """The supervisor's signature over a receipt: its contents, not merely its name."""
        body = pinned({key: value for key, value in receipt.items() if key != 'seal'})
        return hmac.new(bytes.fromhex(seal), body.encode(), hashlib.sha256).hexdigest()

    @classmethod
    def _receipt(cls, folder, name, handle, intent):
        path = folder / name
        try:
            receipt = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        if receipt['token'] != handle.id or receipt['spec_digest'] != intent['spec_digest']:
            raise ValueError('foreign execution receipt')
        seal = intent.get('seal')
        if seal is not None and not hmac.compare_digest(cls.sealed(seal, receipt), receipt.get('seal') or ''):
            # Written by something other than this launch's supervisor -- the worker itself can
            # reach this directory. Handles pinned before sealing existed carry no seal and are
            # read as before, so older runs keep their evidence.
            raise ValueError('execution receipt is not sealed by its supervisor')
        return receipt

    def _release_watch(self, handle_id, *, only_if_exited=True):
        """Let go of a watched run's pipe once its supervisor is gone.

        Never before: closing it is what tells a *live* supervisor to stop, so an early close
        would end the run this call is merely observing.

        A missing process entry is not a live supervisor. Only this process holds a write end, so
        an entry here means this process launched that supervisor, and the only code that drops
        the process entry is ``collect``, which drops it precisely because it has exited.
        """
        if handle_id not in self.controller_ends:
            return
        proc = self.processes.get(handle_id)
        if only_if_exited and not (proc is None or proc.poll() is not None):
            return
        try:
            os.close(self.controller_ends.pop(handle_id))
        except OSError:
            pass

    def discover(self, handle):
        self._release_watch(handle.id)
        try:
            intent, _ = self._spec(handle)
            # Shared paths or a copied registry do not prove visibility into another host.
            if not same_location(intent):
                return Observation('unknown', detail='execution host/boot/PID namespace is not locally visible')
            folder = Path(intent['directory'])
            completed = self._receipt(folder, 'completion.json', handle, intent)
            started = self._receipt(folder, 'started.json', handle, intent)
            worker = self._receipt(folder, 'worker.json', handle, intent)
            for receipt in (started, worker):
                if receipt is not None:
                    process = receipt['process']
                    ProcessIdentity(**{key: value for key, value in process.items() if key != 'pgid'})
                    if (type(process['pid']) is not int or process['pid'] <= 0 or
                            process['pgid'] != process['pid'] or type(process['start_ticks']) is not int or
                            not process['pid_namespace'] or any(process[key] != intent['location'][key]
                                for key in ('host', 'boot_id', 'pid_namespace'))):
                        raise ValueError('incomplete supervised process identity')
            evidence = {'command': intent['command'], 'supervisor': started['process'] if started else None,
                        'worker': worker['process'] if worker else None,
                        'pid': worker['process']['pid'] if worker else None}
            if completed is not None:
                code, observed = completed['returncode'], completed['finished_at']
                if (code is not None and type(code) is not int or
                        isinstance(observed, bool) or not isinstance(observed, (int, float)) or
                        not math.isfinite(observed) or observed < 0 or
                        completed['kind'] not in ('exit', 'launch_failure', 'cancelled_before_spawn') or
                        (completed['kind'] == 'exit') != (type(code) is int) or started is None or
                        (completed['kind'] == 'exit' and worker is None) or
                        not isinstance(completed.get('detail', ''), str)):
                    raise ValueError('invalid completion receipt')
                return Observation('finished', pinned(evidence), code, observed,
                                   completed.get('detail', completed['kind']))
            with locked(folder) as descriptor:
                if descriptor is not None and started is None and worker is None:
                    # The child can publish and finish between the reads above and this
                    # lock acquisition. Absence must be checked at the locked boundary.
                    if any((folder / name).exists() for name in ('started.json', 'worker.json', 'completion.json')):
                        return Observation('unknown', detail='execution evidence changed during discovery; rediscover')
                    return Observation('absent')  # no supervisor passed its pre-spawn publication
            if started and observe_process(started['process']) == 'alive':
                return Observation('running', pinned(evidence))
            return Observation('unknown', pinned(evidence), detail='launch is locked or supervisor vanished without a receipt')
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
            return Observation('unknown', detail=f'unreadable execution evidence: {error}')

    def submit(self, handle, *, watch=None):
        """Start what ``handle`` plans, unless it has been started already.

        ``watch`` says whether the supervisor stops its tree when this process goes away. A
        stage's own worker is always watched (``ComputeWorkers.submit``): nobody is waiting for
        it once the stage is gone. ``None`` leaves it to ``watching()``, which is how a whole
        run is started.
        """
        before = self.discover(handle)
        if before.state != 'absent':
            return before
        intent, spec = self._spec(handle)
        folder = Path(intent['directory'])
        with locked(folder) as descriptor:
            if descriptor is None:
                observed = self.discover(handle)
                # The contender may have been another discovery reader, rather than a
                # launcher. Losing this nonblocking lock is an unconfirmed submission;
                # a later discovery can establish absence and retry this same intent.
                return observed if observed.state != 'absent' else Observation(
                    'unknown', detail='dispatch lock was contended; discover before retrying')
            # Check again after the lock: another controller may have completed meanwhile.
            if any((folder / file).exists() for file in ('started.json', 'worker.json', 'completion.json')):
                return self.discover(handle)
            reader = writer = watched = watcher = None
            try:
                command = [sys.executable, '-m', 'eforecast.config.run_supervisor', str(folder), str(descriptor),
                           intent['spec_digest']]
                # The seal travels on the supervisor's standard input: /proc/<pid>/cmdline and
                # /proc/<pid>/environ are readable by the worker, a consumed pipe is not.
                reader, writer = os.pipe()
                os.write(writer, (intent.get('seal') or '').encode())
                inherited = (descriptor,)
                if self.watching() if watch is None else watch:
                    # A pipe nobody writes to. Its only message is EOF, which arrives however
                    # this process ends -- Stop in a debugger, Ctrl-C, SIGKILL, a crash -- and
                    # needs no cooperation from it. The supervisor stops its tree when it does.
                    watched, watcher = os.pipe()
                    command.append(str(watched))
                    inherited += (watched,)
                proc = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parents[2]),
                    env=spec['environment'], stdin=reader, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, start_new_session=True, pass_fds=inherited)
                self.processes[handle.id] = proc
                if watcher is not None:
                    self.controller_ends[handle.id] = watcher
                    watcher = None              # held for the run, not closed below
            except (OSError, subprocess.SubprocessError):
                # A lost reply may still have created a child. Discovery/lock decides, not this exception.
                return Observation('unknown', detail='supervisor launch acknowledgement unavailable')
            finally:
                for descriptor_end in (reader, writer, watched, watcher):
                    if descriptor_end is not None:
                        os.close(descriptor_end)   # the supervisor reads to EOF, then closes it
        # Short bounded handshake for compatibility with the synchronous CLI/manager helper.
        observed = Observation('unknown', detail='supervisor handshake has not been observed')
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            observed = self.discover(handle)
            if observed.state == 'finished' or (observed.state == 'running' and json.loads(observed.evidence)['worker']):
                return observed
            if observed.state == 'absent':
                return Observation('unknown', detail='supervisor ended before publishing ownership')
            time.sleep(.01)
        return observed

    def status(self, handle):
        return self.discover(handle)

    def cancel(self, handle, *, force=False):
        observed = self.discover(handle)
        if observed.state == 'finished':
            return observed
        try:
            intent, _ = self._spec(handle)
            if not same_location(intent):
                return Observation('unknown', detail='cannot cancel outside verified execution namespace')
            from eforecast.folder_lock import folder_lock as file_critical_section
            folder = Path(intent['directory'])
            with file_critical_section(folder, timeout=1):
                path = folder / 'cancel.json'
                prior = json.loads(path.read_text()) if path.exists() else {}
                atomic_json(path, {'token': handle.id, 'force': force or bool(prior.get('force'))})
                atomic_json(folder / 'reports' / 'cancel.json', {'executor_id': handle.id})
            evidence = json.loads(observed.evidence).get('supervisor')
            if evidence is None:
                return observed  # supervisor reads the durable request before starting its worker
            # Open FIRST then verify identity; pidfd binds the signal to this kernel process,
            # even if it exits and its numeric PID is reused immediately after verification.
            from eforecast.config.run_processes import pidfd_open, pidfd_signal
            descriptor = pidfd_open(evidence['pid'])
            try:
                if observe_process(evidence) != 'alive':
                    return Observation('unknown', detail='recorded supervisor identity cannot be verified')
                pidfd_signal(descriptor, signal.SIGUSR1 if force else signal.SIGTERM)
            finally:
                os.close(descriptor)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            return Observation('unknown', detail=f'cancellation unconfirmed: {error}')
        return self.discover(handle)

    def collect(self, handle):
        proc = self.processes.get(handle.id)
        if proc is not None and proc.poll() is not None:
            self.processes.pop(handle.id, None)
        self._release_watch(handle.id)
        return self.discover(handle)


class FakeExecutor:
    """Deterministic contract adapter: retries discover the same execution and receipt."""
    adapter = 'fake'

    def __init__(self):
        self.executions = {}
        self.submissions = 0

    def plan(self, run_id, task_id, attempt_id, command, environment, cwd, log):
        return ExecutorHandle(identity(), self.adapter, pinned({'run_id': run_id, 'task_id': task_id,
                              'attempt_id': attempt_id, 'command': command}))

    def discover(self, handle):
        return self.executions.get(handle.id, Observation('absent'))

    def submit(self, handle):
        if self.discover(handle).state == 'absent':
            self.submissions += 1
            self.executions[handle.id] = Observation('running', pinned({'token': handle.id}))
        return self.discover(handle)

    def finish(self, handle, code=0):
        self.executions[handle.id] = Observation('finished', pinned({'token': handle.id}), code, time.time())

    def status(self, handle):
        return self.discover(handle)

    def cancel(self, handle, *, force=False):
        if self.discover(handle).state == 'finished':
            return self.discover(handle)
        self.finish(handle, -signal.SIGKILL if force else -signal.SIGTERM)
        return self.discover(handle)

    def collect(self, handle):
        return self.discover(handle)


class DockerExecutor:
    """Recoverable wrapper for the existing trial argv/labels/create adapter.

    Top-level scripts still run under LocalExecutor; their training scheduler places and
    runs the trials. This adapter exposes that same Docker boundary to managed runs.
    Containers are retained through collection.
    """
    adapter = 'docker_trial'
    LABEL_EXECUTOR = 'eforecast.executor'

    def __init__(self, directory=None):
        self.local = LocalExecutor(directory)

    def plan_trial(self, argv, environment, *, run_id, task_id, attempt_id):
        from eforecast.training.command_line import docker_task
        if argv[:2] != ['docker', 'run']:
            raise ValueError('expected existing docker trial argv')
        daemon = subprocess.run(['docker', 'info', '--format', '{{.ID}}'], env=environment,
                                check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        if not daemon:
            raise ValueError('Docker daemon identity unavailable')
        token = identity()
        name = docker_task.container_name(run_id, 'execution', attempt_id, task_id)
        # Preserve the established mounts, environment and labels. Replace only the
        # name/ownership labels and --rm: exit status must outlive the docker client.
        replaced = {docker_task.LABEL_RUN, docker_task.LABEL_TASK, docker_task.LABEL_ATTEMPT,
                    self.LABEL_EXECUTOR}
        clean, index = [], 2
        while index < len(argv):
            value = argv[index]
            if value == '--rm':
                index += 1
            elif (value == '--name' or (value == '--label' and argv[index + 1].split('=', 1)[0] in replaced)
                  or (value == '-e' and argv[index + 1].split('=', 1)[0] in {
                      'EFORECAST_RUN_ID', 'EFORECAST_TASK_ID', 'EFORECAST_ATTEMPT_ID', 'EFORECAST_EXECUTOR_ID'})):
                index += 2
            else:
                clean.append(value)
                index += 1
        labels = {docker_task.LABEL_RUN: run_id, docker_task.LABEL_TASK: task_id,
                  docker_task.LABEL_ATTEMPT: attempt_id, self.LABEL_EXECUTOR: token}
        command = ['docker', 'run', '--name', name, '-e', f'EFORECAST_RUN_ID={run_id}',
                   '-e', f'EFORECAST_TASK_ID={task_id}', '-e', f'EFORECAST_ATTEMPT_ID={attempt_id}',
                   '-e', f'EFORECAST_EXECUTOR_ID={token}']
        for key, value in labels.items():
            command += ['--label', f'{key}={value}']
        command += clean
        folder = self.local.directory / token
        folder.mkdir(parents=True, mode=0o700)
        atomic_json(folder / 'docker.json', {'environment': environment})
        return ExecutorHandle(token, self.adapter, pinned({'version': 1, 'directory': str(folder),
            'location': location(), 'daemon_id': daemon, 'name': name, 'labels': labels, 'command': command}))

    @staticmethod
    def _environment(intent):
        return json.loads((Path(intent['directory']) / 'docker.json').read_text())['environment']

    def _inspect(self, handle):
        intent = json.loads(handle.identity_json)
        if not same_location(intent):
            return Observation('unknown', detail='Docker daemon belongs to an unverified execution location')
        try:
            environment = self._environment(intent)
            daemon = subprocess.run(['docker', 'info', '--format', '{{.ID}}'], env=environment,
                                    check=True, capture_output=True, text=True, timeout=5).stdout.strip()
            if daemon != intent['daemon_id']:
                return Observation('unknown', detail='Docker daemon identity changed')
            result = subprocess.run(['docker', 'inspect', intent['name']], capture_output=True,
                                    env=environment, text=True, timeout=5)
            if result.returncode:
                if 'no such object:' in result.stderr.lower() or 'no such container:' in result.stderr.lower():
                    return Observation('absent')
                return Observation('unknown', detail='Docker daemon unavailable')
            item, = json.loads(result.stdout)
            labels = item['Config'].get('Labels') or {}
            if any(labels.get(key) != value for key, value in intent['labels'].items()):
                return Observation('unknown', detail='container name belongs to a foreign execution')
            cid = item['Id']
            if len(cid) != 64 or any(char not in '0123456789abcdef' for char in cid):
                raise ValueError('Docker returned an abbreviated or invalid container ID')
            created = Path(intent['directory']) / 'docker_created.json'
            if created.exists() and json.loads(created.read_text()) != {'token': handle.id, 'container_id': cid}:
                return Observation('unknown', detail='container identity disagrees with recorded launch')
            evidence = pinned({'container_id': cid, 'labels': intent['labels'], 'name': intent['name']})
            state = item['State']
            if state['Status'] in ('exited', 'dead'):
                from datetime import datetime
                # A dead container without an observed start is not an executed success.
                if state['StartedAt'].startswith('0001-') or type(state['ExitCode']) is not int:
                    return Observation('unknown', evidence, detail='container has no valid execution receipt')
                observed = datetime.fromisoformat(state['FinishedAt'].replace('Z', '+00:00')).timestamp()
                if not math.isfinite(observed) or observed <= 0:
                    return Observation('unknown', evidence, detail='container completion time is unverified')
                return Observation('finished', evidence, state['ExitCode'], observed, 'Docker exit')
            if state['Status'] == 'created':
                return Observation('created', evidence)
            if state['Status'] in ('running', 'paused', 'restarting'):
                return Observation('running', evidence)
            return Observation('unknown', evidence, detail=f"Docker status {state['Status']}")
        except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, subprocess.SubprocessError) as error:
            return Observation('unknown', detail=f'Docker discovery unavailable: {error}')

    def discover(self, handle):
        intent = json.loads(handle.identity_json)
        try:
            path = Path(intent['directory']) / 'docker_completion.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.pop('token') != handle.id or saved.pop('intent_digest') != digest(intent):
                    raise ValueError('foreign Docker completion receipt')
                observation = Observation(**saved)
                if (observation.state != 'finished' or type(observation.returncode) is not int or
                        type(observation.finished_at) not in (float, int) or
                        not math.isfinite(observation.finished_at) or observation.finished_at <= 0 or
                        not isinstance(observation.detail, str)):
                    raise ValueError('invalid Docker completion receipt')
                evidence = json.loads(observation.evidence)
                cid = evidence['container_id']
                if (evidence['labels'] != intent['labels'] or evidence['name'] != intent['name'] or
                        len(cid) != 64 or any(char not in '0123456789abcdef' for char in cid)):
                    raise ValueError('foreign or malformed Docker completion identity')
                return observation
            return self._inspect(handle)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            return Observation('unknown', detail=f'Docker receipt unavailable: {error}')

    def submit(self, handle):
        from eforecast.training.command_line import docker_task
        intent = json.loads(handle.identity_json)
        if not same_location(intent):
            return Observation('unknown', detail='cannot launch on another execution location')
        try:
            with locked(intent['directory']) as descriptor:
                if descriptor is None:
                    return Observation('unknown', detail='Docker dispatch already in progress')
                observed = self.discover(handle)
                if observed.state == 'absent':
                    # Missing after a known creation is loss, not permission to run again.
                    if (Path(intent['directory']) / 'docker_created.json').exists():
                        return Observation('unknown', detail='previously created container is missing')
                    env = json.loads((Path(intent['directory']) / 'docker.json').read_text())['environment']
                    cid = docker_task.create_trial(intent['command'], env)
                    atomic_json(Path(intent['directory']) / 'docker_created.json', {'token': handle.id, 'container_id': cid})
                    observed = self.discover(handle)
                if observed.state == 'created':
                    cid = json.loads(observed.evidence)['container_id']
                    folder = Path(intent['directory'])
                    atomic_json(folder / 'docker_created.json', {'token': handle.id, 'container_id': cid})
                    # Docker has no conditional start operation. If a prior start is still
                    # unacknowledged, retrying could restart a fast container. Retain unknown
                    # while discovery waits for running/exited evidence instead.
                    if (folder / 'docker_start.json').exists():
                        return Observation('unknown', observed.evidence, detail='Docker start is unconfirmed; not repeated')
                    atomic_json(folder / 'docker_start.json', {'token': handle.id, 'container_id': cid})
                    subprocess.run(['docker', 'start', cid], env=self._environment(intent),
                                   check=True, capture_output=True, timeout=30)
                    observed = self.discover(handle)
                return observed
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            return Observation('unknown', detail=f'Docker launch acknowledgement unavailable: {error}')

    def status(self, handle):
        return self.discover(handle)

    def cancel(self, handle, *, force=False):
        observed = self.discover(handle)
        if observed.state in ('running', 'created'):
            cid = json.loads(observed.evidence)['container_id']
            try:
                subprocess.run(['docker', 'kill', '--signal', 'KILL' if force else 'TERM', cid],
                               env=self._environment(json.loads(handle.identity_json)),
                               check=True, capture_output=True, timeout=10)
            except (OSError, subprocess.SubprocessError):
                return Observation('unknown', observed.evidence, detail='Docker cancellation unconfirmed')
        return self.discover(handle)

    def collect(self, handle):
        observed = self.discover(handle)
        if observed.state == 'finished':
            intent = json.loads(handle.identity_json)
            try:
                atomic_json(Path(intent['directory']) / 'docker_completion.json',
                            {'token': handle.id, 'intent_digest': digest(intent), **asdict(observed)})
            except OSError as error:
                return Observation('unknown', observed.evidence, detail=f'Docker receipt publication failed: {error}')
        return observed
