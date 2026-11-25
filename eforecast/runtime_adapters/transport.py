# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Move numerical payloads across a runtime boundary without pandas or pickle.

A16-02.  The model images cannot share pipe_env's pandas: foundationforecast resolves at
pandas 2.3.3 and Lag-Llama at 2.1.4, while the host runs 3.0.3 -- and a pandas 3 pickle is
unreadable by pandas 2, which is the failure A13-01 closed by upgrading the reader.  Here
there is no reader to upgrade, because the whole point of the image is to hold an older
stack.  So nothing but `.npy` buffers and JSON crosses the boundary, which is the format
A16-01's contract already specifies, and timestamps travel as UTC nanosecond integers
rather than as a DatetimeIndex.

pandas inside an image is fine and the adapters use it; what must not cross is a pandas
*object*.  This module is imported inside the images, where only numpy and the stdlib are
guaranteed, so it imports neither the contract nor anything from eforecast.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

ITEMSIZE = {'<f4': 4, '<f8': 8, '<i8': 8, '|b1': 1}

ARRAYS = 'arrays'
REQUEST = 'request.json'
RESULT = 'result.json'
CAPABILITY = 'capability.json'
FAILURE = 'failure.json'


class TransportError(ValueError):
    """A payload did not match what its descriptor promised."""


def _digest(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def array_path(task_dir, artifact_id):
    # an artifact id names a file in this task's own directory; a separator or a traversal
    # would reach outside it, and the image mounts the task directory writable
    if not artifact_id or os.path.basename(artifact_id) != artifact_id or artifact_id.startswith('.'):
        raise TransportError(f'artifact_id: {artifact_id!r} does not name a file in the task directory')
    return os.path.join(task_dir, ARRAYS, artifact_id + '.npy')


def load(task_dir, descriptor, field='array'):
    """The array a descriptor names, checked against everything the descriptor promised."""
    path = array_path(task_dir, descriptor['artifact_id'])
    if not os.path.exists(path):
        raise TransportError(f'{field}: {descriptor["artifact_id"]}.npy is missing from the task directory')
    if _digest(path) != descriptor['sha256']:
        raise TransportError(f'{field}: {descriptor["artifact_id"]} does not match its recorded digest')
    # allow_pickle=False is load-bearing: an object-dtype .npy would otherwise run pickle,
    # which is both the compatibility problem this module exists to avoid and arbitrary code
    value = np.load(path, allow_pickle=False)
    if value.dtype.str != descriptor['dtype']:
        raise TransportError(f'{field}: dtype {value.dtype.str} where {descriptor["dtype"]} was declared')
    if list(value.shape) != list(descriptor['shape']):
        raise TransportError(f'{field}: shape {list(value.shape)} where {descriptor["shape"]} was declared')
    return value


def serialized(value):
    """The exact npy bytes an array would be written as.

    Digesting the payload before writing is what makes hash-based staging possible: the
    descriptor is known without touching the disk, so an artifact already present with that
    digest need not be written again. The cost is holding one array's bytes twice, which is
    bounded here by the batch.
    """
    import io
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def store(task_dir, name, value, *, dtype=None, reuse=None):
    """Write an array into the task directory and describe it the way the contract reads it.

    ``reuse`` is a predicate ``(name, descriptor) -> bool``. When it says the artifact is
    already there, complete and the one wanted, nothing is written and the descriptor is
    returned as it stands -- which is what "reuse complete pinned inputs" asks for.
    """
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    value = np.ascontiguousarray(value)
    if value.dtype.str not in ITEMSIZE:
        raise TransportError(f'{name}: {value.dtype.str} is not a transportable dtype')
    if value.dtype.kind == 'f' and not np.isfinite(value).all():
        count = int((~np.isfinite(value)).sum())
        raise TransportError(f'{name}: {count} non-finite value(s); a forecast crosses the boundary finite')
    directory = os.path.join(task_dir, ARRAYS)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name + '.npy')
    payload = serialized(value)
    descriptor = {'artifact_id': name, 'format': 'npy',
                  'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
                  'dtype': value.dtype.str, 'shape': [int(n) for n in value.shape],
                  'order': 'C'}
    if reuse is not None and reuse(name, descriptor):
        return descriptor               # already there, complete, and the one wanted
    # Atomic, for the same reason write_document is: an interrupted write -- a kill, a full
    # disk -- otherwise leaves a partial payload under the final name, and nothing
    # distinguishes it from a complete one by name. With reuse above, that would mean
    # reusing a half-written input. The visible name only ever appears via a rename.
    temporary = path + '.partial'
    try:
        with open(temporary, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass                        # the partial is unreachable either way: it is not `path`
        raise
    return descriptor


def read_document(task_dir, name):
    with open(os.path.join(task_dir, name), 'r', encoding='utf-8') as handle:
        return json.load(handle)


def write_document(task_dir, name, document):
    """Publish a document the host reads as the attempt's outcome, completely or not at all."""
    # the same lesson as A16-10's credential store: a truncated write must not be mistaken
    # for a finished one, so the visible name only ever appears via a rename
    path = os.path.join(task_dir, name)
    temporary = path + '.partial'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(document, handle, sort_keys=True, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path
