# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""NVIDIA MPS for the cards of one training run: trials that share a card run their kernels together.

Without it the processes on a card take turns: the kernels of one run at a time and the card
switches between them. Small networks, which is what a card holds several of, lose most of it to
the switching. Measured on one RTX 2080 Ti with a small dense network (training steps a second,
all processes together):

    processes on the card       1      4      8
    taking turns             1282   4343   5000
    through MPS              1284   4973   8374

and the card read 78% busy with four of them where it had read 100%, so the scheduler, which adds
trials of a kind to a card until the card is busy, finds room it could not see before.

One control daemon a card, started by the run and stopped with it:

- in the foreground, as a child of the scheduler, and told by the kernel to end when the scheduler
  does. A daemon that outlived its run would be found by nobody;
- its pipe directory under the user's runtime directory, because the path of its socket has to be
  short: one of 149 characters made the daemon exit at once, without a word in its log;
- a trial is given the directory in its environment. A process started with a directory nobody
  listens in runs without MPS, so a daemon that could not be started costs nothing.

When a daemon goes, its server goes with it, and every trial on that card fails with "the remote
procedural call between the MPS server and the MPS client failed". The scheduler reads that as the
card's sharing being lost: the card goes on without it and those trials are started again.

``EFORECAST_GPU_MPS=0`` turns it off. Trials in containers do not use it (the pipe directory and
the IPC namespace would have to be shared with them), nor do cards that cannot be named by UUID.
"""
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time

log = logging.getLogger(__name__)

CONTROL = 'nvidia-cuda-mps-control'
SOCKET_PATH_LIMIT = 100                 # sun_path holds 108 bytes; the daemon adds names of its own
STARTS_WITHIN = 5.0
#: What a trial says when the daemon it was using has gone.
LOST = ('MPS server and the MPS client', 'cudaErrorMpsRpcFailure', 'cudaErrorMpsConnectionFailed',
        'cudaErrorMpsServerNotReady', 'cudaErrorMpsMaxClientsReached', 'cudaErrorMpsMaxConnectionsReached')


def wanted():
    return os.environ.get('EFORECAST_GPU_MPS', '1').strip().lower() not in {'0', 'off', 'false', 'no'}


def lost(error):
    """Whether a trial's error says its card's MPS daemon went away under it."""
    text = str(error or '')
    return any(mark in text for mark in LOST)


def _end_with_parent():
    """Run in the child before exec: the kernel ends it (SIGTERM) when the scheduler's thread ends."""
    try:
        import ctypes
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)     # PR_SET_PDEATHSIG
    except Exception:                                           # noqa: BLE001 -- stop() still ends it
        pass


def _others():
    """Process ids of MPS control daemons already running on this machine."""
    found = []
    try:
        for entry in os.scandir('/proc'):
            if entry.name.isdigit():
                try:
                    with open(f'/proc/{entry.name}/comm') as stream:
                        if stream.read().strip() == CONTROL[:15]:
                            found.append(int(entry.name))
                except OSError:
                    continue
    except OSError:
        pass
    return found


class Sharing:
    """The MPS daemons of one run: ``start``, ``environment(gpu)`` for each trial, ``stop``."""

    def __init__(self, cards, control=None, base=None):
        self.cards = dict(cards)                                # gpu ordinal -> its UUID ('GPU-...')
        self.control = control if control is not None else shutil.which(CONTROL)
        self.base = base
        self.daemons = {}                                       # gpu -> {'process', 'folder', 'environment', 'log'}
        self.dropped = {}                                       # gpu -> why it goes on without sharing

    def _base(self):
        if self.base:
            return self.base
        runtime = os.environ.get('XDG_RUNTIME_DIR')
        return runtime if runtime and os.path.isdir(runtime) and os.access(runtime, os.W_OK) else tempfile.gettempdir()

    def start(self):
        """Start a daemon for every card that can have one; returns the cards that are shared."""
        if not self.cards:
            return []
        if not self.control:
            log.info('trials on one card take turns: %s is not installed, so the cards are not shared through MPS', CONTROL)
            return []
        already = _others()
        if already:
            log.info('trials on one card take turns: an MPS control daemon is already running here (pid %s), and '
                     'who shares a card through it is its owner\'s to say', ', '.join(map(str, already)))
            return []
        for gpu, uuid in self.cards.items():
            try:
                self._start(gpu, uuid)
            except Exception as error:                          # noqa: BLE001 -- a run without sharing is still a run
                self.dropped[gpu] = f'its MPS daemon could not be started ({error})'
                log.warning('gpu %s: %s; its trials take turns', gpu, self.dropped[gpu])
        if self.daemons:
            log.info('cards %s are shared through MPS: the trials on one card run their kernels together '
                     '(EFORECAST_GPU_MPS=0 turns this off)', sorted(self.daemons))
        return sorted(self.daemons)

    def _start(self, gpu, uuid):
        folder = tempfile.mkdtemp(prefix='efmps-', dir=self._base())
        pipe, logs = os.path.join(folder, 'pipe'), os.path.join(folder, 'log')
        if len(os.path.join(pipe, 'control')) > SOCKET_PATH_LIMIT:
            shutil.rmtree(folder, ignore_errors=True)
            raise OSError(f'the socket path under {self._base()} is too long for a socket')
        os.mkdir(pipe)
        os.mkdir(logs)
        environment = {'CUDA_MPS_PIPE_DIRECTORY': pipe, 'CUDA_MPS_LOG_DIRECTORY': logs}
        output = open(os.path.join(folder, 'control.out'), 'ab')
        try:
            process = subprocess.Popen([self.control, '-f'], stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, preexec_fn=_end_with_parent,
                                       env={**os.environ, **environment, 'CUDA_VISIBLE_DEVICES': str(uuid)})
        finally:
            output.close()
        deadline = time.monotonic() + STARTS_WITHIN
        while process.poll() is None and not os.path.exists(os.path.join(pipe, 'control')):
            if time.monotonic() > deadline:
                break
            time.sleep(0.05)
        if process.poll() is not None or not os.path.exists(os.path.join(pipe, 'control')):
            said = self._said(folder)
            self._end(process)
            shutil.rmtree(folder, ignore_errors=True)
            raise OSError(f'the daemon did not come up{": " + said if said else ""}')
        self.daemons[gpu] = {'process': process, 'folder': folder, 'environment': environment}

    @staticmethod
    def _said(folder):
        try:
            with open(os.path.join(folder, 'control.out'), errors='replace') as stream:
                lines = [line.strip() for line in stream if line.strip()]
            return lines[-1][:200] if lines else ''
        except OSError:
            return ''

    def serves(self, gpu):
        daemon = self.daemons.get(gpu)
        return daemon is not None and daemon['process'].poll() is None

    def environment(self, gpu):
        """What a trial on ``gpu`` is started with to share the card; empty when the card is not shared."""
        daemon = self.daemons.get(gpu)
        if daemon is None:
            return {}
        if daemon['process'].poll() is not None:
            self.drop(gpu, f'its MPS daemon ended (exit {daemon["process"].returncode})')
            return {}
        return dict(daemon['environment'])

    def drop(self, gpu, reason):
        """Go on without sharing ``gpu``: stop its daemon, say why once. Returns whether it was shared."""
        daemon = self.daemons.pop(gpu, None)
        if daemon is None:
            return False
        self.dropped[gpu] = reason
        log.warning('gpu %s is no longer shared through MPS: %s. The trials on it take turns from now on, and '
                    'the ones that failed with it are started again', gpu, reason)
        self._stop(daemon)
        return True

    def stop(self):
        for gpu in list(self.daemons):
            self._stop(self.daemons.pop(gpu))

    def _stop(self, daemon):
        process = daemon['process']
        if process.poll() is None:
            try:
                subprocess.run([self.control], input=b'quit\n', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env={**os.environ, **daemon['environment']}, timeout=10)
            except Exception:                                   # noqa: BLE001 -- it is ended below either way
                pass
        self._end(process, patience=5)
        shutil.rmtree(daemon['folder'], ignore_errors=True)

    @staticmethod
    def _end(process, patience=0):
        try:
            process.wait(timeout=patience)
        except subprocess.TimeoutExpired:
            process.terminate()                                 # the daemon takes its server with it
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def status(self):
        """``{gpu: 'mps' | reason it is not shared}`` for the run's status file."""
        out = {str(gpu): 'mps' for gpu in self.daemons}
        out.update({str(gpu): reason for gpu, reason in self.dropped.items()})
        return out
