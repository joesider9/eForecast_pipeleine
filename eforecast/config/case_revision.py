# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Optional optimistic concurrency for web-editor writes, without case migration.

Case GET/PUT/PATCH and create/delete/share serialize per case: readers share the lock, a
writer excludes everything, and nobody waits past ``EFORECAST_CASE_LOCK_TIMEOUT`` -- the
server answers 503 rather than holding a worker thread for as long as the holder lives.
List summaries, copy-source reads and CLI/external file writers do not join this lock;
immutable run inputs and broader filesystem transactions remain separate work.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import time

DEFAULT_TIMEOUT = 5.0


class CaseBusy(RuntimeError):
    """Another request held the case longer than a save takes; the caller is told to retry."""


def revision(document):
    # The HTTP representation has string mapping keys, including legacy YAML keys.
    normalized = json.loads(json.dumps(document, default=str))
    payload = json.dumps(normalized, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def timeout_seconds():
    """Read at call time, so a deployment -- or a test -- can choose how long a request waits."""
    try:
        return max(0., float(os.environ.get('EFORECAST_CASE_LOCK_TIMEOUT', DEFAULT_TIMEOUT)))
    except ValueError:
        return DEFAULT_TIMEOUT


@contextmanager
def locked(workspace, name, shared=False):
    workspace._check_name(name)
    directory = workspace.root / '.case-locks'
    directory.mkdir(parents=True, exist_ok=True)
    # Names are unique across owners in a workspace, including simultaneous creates.
    key = hashlib.sha256(f'{workspace.root.resolve()}:{name}'.encode()).hexdigest()
    mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    deadline = time.monotonic() + timeout_seconds()
    with (directory / f'{key}.lock').open('a') as stream:
        while True:
            try:
                fcntl.flock(stream.fileno(), mode | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise CaseBusy(f'case {name!r} is busy: another request is writing it. Try again.') from None
                time.sleep(.01)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
