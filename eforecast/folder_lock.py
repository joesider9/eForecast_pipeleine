# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A lock on a folder, held by a process and dropped by the kernel when that process goes away.

This is what keeps two writers out of one folder: a case's artifacts, a trial's folder while its
files are moved in, a run's report journal. It is an ``flock`` on a file inside the folder, so
nothing has to be cleaned up after a holder that was killed, and nothing is recorded anywhere
else. The file is called ``allocator.lock`` because folders on disk already hold one by that
name, and a second lock file under another name would not exclude a writer that still uses it.
"""
import contextlib
import fcntl
import math
import os
import threading
import time

LOCK_FILE = 'allocator.lock'
DEFAULT_TIMEOUT = 30.0


class Busy(RuntimeError):
    """The lock could not be taken in time; the caller waits or says who holds the folder."""


def timeout_seconds():
    try:
        value = float(os.environ.get('EFORECAST_ALLOCATOR_TIMEOUT', DEFAULT_TIMEOUT))
        return max(0., value) if math.isfinite(value) else DEFAULT_TIMEOUT
    except ValueError:
        return DEFAULT_TIMEOUT


_held = threading.local()


def locks():
    """The folders this thread holds, with how many times each was entered."""
    if getattr(_held, 'pid', None) != os.getpid():
        _held.pid, _held.locks = os.getpid(), {}
    return _held.locks


@contextlib.contextmanager
def folder_lock(folder, timeout=None):
    """Hold ``folder`` for the block; reentrant for the thread that holds it."""
    folder = os.path.realpath(folder)
    held = locks()
    if held.get(folder):
        held[folder] += 1
        try:
            yield
        finally:
            held[folder] -= 1
        return
    os.makedirs(folder, exist_ok=True)
    deadline = time.monotonic() + (timeout_seconds() if timeout is None else timeout)
    with open(os.path.join(folder, LOCK_FILE), 'a') as stream:
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise Busy(f'the lock of {folder} is held by another run') from None
                time.sleep(.01)
        held[folder] = 1
        try:
            yield
        finally:
            held.pop(folder, None)
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
