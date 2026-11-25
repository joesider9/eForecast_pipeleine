# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""One supervised local execution: start the worker, watch its tree, record how it ended."""
import json
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from eforecast.config.executors import LocalExecutor, atomic_json, digest, process_evidence
from eforecast.config.run_processes import descendants, key, signal_owned


def supervise(folder, lock_fd, expected_digest, seal=None, controller_fd=None):
    folder = Path(folder)
    os.fstat(lock_fd)  # inherited lock ownership must survive the controller
    spec = json.loads((folder / 'spec.json').read_text())
    if digest(spec) != expected_digest:
        raise ValueError('execution specification changed after launch intent was pinned')
    base = {'token': spec['token'], 'spec_digest': digest(spec)}

    def receipt(path, payload):
        """Every piece of evidence this supervisor publishes, signed with the controller's seal.

        The worker runs in this directory's registry entry and could otherwise write a completion
        of its own choosing. It never receives the seal: it arrived on standard input, which is
        closed below before the worker is spawned.
        """
        document = {**base, **payload}
        if seal:
            document['seal'] = LocalExecutor.sealed(seal, document)
        atomic_json(path, document)
    # Reparent double-forked / new-session children here, including joblib/dataset workers.
    # ECHILD after the root exits is a kernel-backed empty tree, not a sparse journal guess.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'cannot establish descendant supervision')
    proc = None
    requested = [False, False]
    inventory = {}
    signalled = {}
    census_due = [0.]
    compute = None

    def census():
        if time.monotonic() < census_due[0]:
            return
        census_due[0] = time.monotonic() + .25
        before = len(inventory)
        for process in descendants(os.getpid()):
            inventory[key(process)] = process
        if len(inventory) != before:
            receipt(folder / 'tree.json', {'processes': list(inventory.values())})

    if controller_fd is not None:
        os.set_blocking(controller_fd, False)

    def controller_gone():
        """Has the process that launched this run gone away?

        Only the controller holds the write end of this pipe, and nothing is ever written to it,
        so a read of ``b''`` is EOF and EOF is proof: the kernel closes that end however the
        controller ended -- Stop in a debugger, Ctrl-C, SIGKILL, a crash -- and no cooperation
        from it is required. This is a positive observation, not a timeout.

        A run started by the service is not given this descriptor at all: it survives the
        process that started it, because a service must not lose hours of compute to a restart.
        A stage's own workers are always given one. See ``LocalExecutor.submit``.
        """
        if controller_fd is None:
            return False
        try:
            return os.read(controller_fd, 1) == b''
        except BlockingIOError:
            return False
        except OSError:
            return True             # an unusable descriptor is no longer evidence of a live controller

    def signal_tree(force):
        for process in inventory.values():
            if signalled.get(key(process)) != force:
                signal_owned(process, force=force)
                signalled[key(process)] = force

    def request(signum, _frame):
        requested[0] = True
        requested[1] |= signum == signal.SIGUSR1

    signal.signal(signal.SIGTERM, request)
    signal.signal(signal.SIGINT, request)
    signal.signal(signal.SIGUSR1, request)

    def wait_worker():
        sent = None
        while True:
            census()
            # Do not reap before signalling: the unreaped group leader pins its PID,
            # eliminating a process-group reuse race in this parent/child relationship.
            try:
                cancel = json.loads((folder / 'cancel.json').read_text())
                if cancel['token'] == spec['token']:
                    requested[0] = True
                    requested[1] |= cancel['force']
            except (OSError, ValueError, KeyError, TypeError):
                pass  # direct verified signals still work if the request file is unreadable
            if controller_gone():
                requested[0] = True
            wanted = signal.SIGKILL if requested[1] else signal.SIGTERM if requested[0] else None
            if wanted and wanted != sent:
                try:
                    os.killpg(proc.pid, wanted)
                except ProcessLookupError:
                    pass
                sent = wanted
            if wanted:
                signal_tree(wanted == signal.SIGKILL)
            code = proc.poll()
            if code is not None:
                return code
            time.sleep(.05)

    receipt(folder / 'started.json', {'process': process_evidence(os.getpid(), spec['location'])})
    kind, code, detail = 'launch_failure', None, ''
    try:
        path = Path(spec['log'])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as log:
            log.write((' '.join(spec['command']) + '\n\n').encode())
            log.flush()
            if (folder / 'cancel.json').exists() or requested[0] or controller_gone():
                kind = 'cancelled_before_spawn'
            else:
                if spec.get('compute'):
                    from eforecast.config.compute_workers import Completion
                    compute = Completion(spec)
                    # No target interpreter exists until the input it was prepared with has
                    # been found unchanged.
                    compute.before_spawn()
                # Worker inherits the lock too: killing the supervisor cannot make an
                # unrecorded but live worker look safe to relaunch.
                environment = spec['environment']
                if spec.get('compute'):
                    environment = {**environment, 'EFORECAST_COMPUTE_ATTACHED':
                                   spec['compute']['context']['launch_id']}
                proc = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lock_fd,))
                receipt(folder / 'worker.json', {'process': process_evidence(proc.pid, spec['location'])})
                code = wait_worker()
                kind = 'exit'
    except Exception as error:
        if proc is not None:
            # No false completion: an error after spawn may leave live work. Keep the
            # inherited lock and supervise/reap the worker even if evidence publication failed.
            wait_worker()
            return 1
        detail = f'{type(error).__name__}: {error}'
    receipt(folder / 'completion.json', {'kind': kind, 'returncode': code,
                                         'finished_at': time.time(), 'detail': detail or kind,
                                         **({'compute_cancelled': requested[0] or kind == 'cancelled_before_spawn'}
                                            if spec.get('compute') else {})})
    # Root outcome is published independently. Keep this supervisor and its lock until every
    # local descendant has actually exited and been reaped, even after a normal root exit.
    deadline = time.monotonic() + 10
    while True:
        census()
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            receipt(folder / 'tree_cleanup.json', {'complete': True, 'observed_at': time.time(),
                                                   'processes': list(inventory.values())})
            if spec.get('compute'):
                try:
                    if compute is None:
                        from eforecast.config.compute_workers import Completion
                        compute = Completion(spec)
                    result = compute.finish(code, cancelled=requested[0] or kind == 'cancelled_before_spawn',
                                            detail=detail if kind == 'launch_failure' else None)
                    receipt(folder / 'compute_completion.json', {'result': result})
                except Exception as error:
                    # The result could not be written down (a full disk, a removed folder).
                    # Whoever collects this worker is told so instead of being told nothing.
                    receipt(folder / 'compute_error.json', {'detail': f'{type(error).__name__}: {error}'})
            return 0
        signal_tree(requested[1] or time.monotonic() >= deadline)
        time.sleep(.05)


if __name__ == '__main__':
    # The seal arrives on standard input and goes no further: the descriptor is replaced by
    # /dev/null before any worker exists, so the worker cannot read what it was.
    secret = sys.stdin.buffer.read(64).decode('ascii', 'ignore').strip() or None
    with open(os.devnull, 'rb') as quiet:
        os.dup2(quiet.fileno(), 0)
    # A fourth argument is the controller's pipe: this run stops when that process does.
    sys.exit(supervise(sys.argv[1], int(sys.argv[2]), sys.argv[3], secret,
                       int(sys.argv[4]) if len(sys.argv) > 4 else None))
