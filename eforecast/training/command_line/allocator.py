# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Compatibility locking and trial fences used by the Step 30/31 training loop.

The compatibility path holds a section across admission, spawning and claim publication. That
serializes cooperating schedulers, but does not commit a durable reservation before spawning.
The local launch_allocations integration commits through FileAllocator before spawning instead.
The local backend uses flock and fences.json; the legacy PostgreSQL backend provides the same
locking/fence functions. Neither compatibility interface provides resource leases or resize.

The functions below are the Step 30/31 compatibility interface. The complete allocation API is
``allocation_contract.Allocator``, implemented locally by ``allocator_file.FileAllocator``.
The legacy PostgreSQL/fence interface is not an implementation of that newer protocol yet.
"""
import contextlib
import fcntl
import json
import logging
import math
import os
import threading
import time
import uuid


log = logging.getLogger(__name__)

from eforecast.folder_lock import Busy as AllocatorBusy, DEFAULT_TIMEOUT, LOCK_FILE, timeout_seconds  # noqa: F401
from eforecast.folder_lock import folder_lock as file_critical_section, locks as _locks                # noqa: F401

FENCE_FILE = 'fences.json'
DEFAULT_DIRECTORY = '/tmp/eforecast-admission'


def directory():
    """Where this host's allocation ledger is kept (``EFORECAST_ADMISSION_DIR``)."""
    return os.environ.get('EFORECAST_ADMISSION_DIR') or DEFAULT_DIRECTORY


def backend():
    """``file`` (one host) or ``postgres`` (several). The contract is the same; the storage is not."""
    return os.environ.get('EFORECAST_ALLOCATOR_BACKEND') or ('postgres' if os.environ.get('EFORECAST_ALLOCATOR_DSN')
                                                             else 'file')


def holding():
    """Whether this thread owns the configured local critical section right now."""
    return bool(_locks().get(os.path.realpath(directory()), 0))


@contextlib.contextmanager
def critical_section(timeout=None):
    """Hold the host's allocation lock, or refuse. Re-entrant within its owning thread."""
    if backend() != 'file':
        from eforecast.training.command_line import allocator_postgres

        with allocator_postgres.critical_section(timeout=timeout):
            yield
        return
    with file_critical_section(directory(), timeout):
        yield


# --------------------------------------------------------------------------- fences
def _fence_path(folder=None):
    return os.path.join(folder or directory(), FENCE_FILE)


def read_fences(strict=False, folder=None):
    if backend() != 'file':
        from eforecast.training.command_line import allocator_postgres

        return allocator_postgres.read_fences()
    try:
        with open(_fence_path(folder), encoding='utf-8') as stream:
            fences = json.load(stream)
        if not isinstance(fences, dict):
            raise ValueError('legacy fences must be a mapping')
        return fences
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        if strict:
            raise
        return {}


def _write_fences(fences, folder=None):
    path = _fence_path(folder)
    temporary = f'{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(fences, stream, indent=1)
    os.replace(temporary, path)


def project_fence(task, trial, allocation, folder=None):
    """Compatibility view of a fence allocated by the authoritative local ledger."""
    if not _locks().get(os.path.realpath(folder or directory())):
        raise RuntimeError('a fence projection is written inside the host critical section')
    fences = read_fences(strict=True, folder=folder)
    fences[trial_key(task, trial)] = {'fence': allocation.token.fence,
                                    'run_id': allocation.request.run_id,
                                    'allocation': allocation.token.allocation_id,
                                    'task_id': allocation.request.task_id,
                                    'attempt': allocation.request.attempt_id, 'taken': time.time()}
    _write_fences(fences, folder=folder)


def trial_key(task, trial):
    return f'{task}/{trial}'


def current_fence(task, trial):
    """The number the worker that may write to this trial folder carries, or ``None``."""
    record = read_fences().get(trial_key(task, trial))
    return record.get('fence') if isinstance(record, dict) else None


def claim_fence(task, trial, run_id, attempt=0):
    """Take the next fence for a trial. Must be called inside the critical section."""
    if backend() != 'file':
        from eforecast.training.command_line import allocator_postgres

        return allocator_postgres.claim_fence(task, trial, run_id, attempt)
    if not holding():
        raise RuntimeError('a fence is taken inside the critical section, never beside it')
    fences = read_fences()
    key = trial_key(task, trial)
    record = fences.get(key) if isinstance(fences.get(key), dict) else {}
    fence = int(record.get('fence') or 0) + 1
    fences[key] = {'fence': fence, 'run_id': run_id, 'attempt': attempt,
                   'allocation': uuid.uuid4().hex, 'taken': time.time()}
    _write_fences(fences)
    return fence


def owner_of(task, trial):
    """``(fence, run_id, allocation)`` of whoever may write to this trial folder."""
    record = read_fences().get(trial_key(task, trial)) or {}
    return record.get('fence'), record.get('run_id'), record.get('allocation')


def accepts(task, trial, fence):
    """Whether a result carrying ``fence`` comes from the worker that is allowed to write here.

    A result with no fence predates this bookkeeping and is accepted, as every other reader in
    this series accepts evidence written before it existed; one with an older fence does not.
    """
    current = current_fence(task, trial)
    if fence is None or current is None:
        return True, 'unfenced'
    if int(fence) == int(current):
        return True, 'current'
    return False, f'fence {fence} is not the current {current}'
