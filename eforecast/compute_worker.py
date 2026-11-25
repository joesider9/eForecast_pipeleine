# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Neutral, stdlib-only worker envelope: read the input, run the target, write the result.

Importing this module is safe in the lightweight online image. A supervisor starts ``main`` in
a process of its own; whatever the target loads lives and dies with that process tree.
Targets accept (payload, context) and return a JSON mapping; output stays in the worker's own
folder. Optional ``artifacts`` is a list of relative files under ``artifacts/``, which the
supervisor hashes once the tree has exited.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from eforecast import execution_policy
from eforecast.compute_context import active_worker as _active


def current_worker():
    return _active.get()


def atomic_bytes(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix='.worker-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def check_inputs(root, records):
    for name, expected in records.items():
        if not name or Path(name).name != name or name in {'.', '..'}:
            raise ValueError('compute input must have a simple file name')
        path = Path(root) / 'inputs' / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('compute input is missing or is a symlink')
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(block)
        if expected != {'bytes': path.stat().st_size, 'sha256': checksum.hexdigest()}:
            raise ValueError('pinned compute input changed')


@dataclass(frozen=True)
class WorkerContext:
    """What one start of a piece of work knows about itself.

    ``launch_id`` is unique to this start and names what it writes; ``threads`` is how many its
    libraries are told to use and ``device_uuid`` the one card it sees, if any. Nothing here is
    a reservation: the operating system gives the process what it gives any process.
    """
    run_id: str
    task_id: str
    attempt_id: str
    launch_id: str
    threads: int
    device_uuid: str | None
    result_directory: str

    def __post_init__(self):
        for name in ('run_id', 'task_id', 'attempt_id', 'launch_id', 'result_directory'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'invalid worker {name}')
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError('a worker needs at least one thread')


def atomic_result(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix='.worker-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(path):
    document = json.loads(Path(path).read_text())
    context = WorkerContext(**document['context'])
    if os.environ.get('EFORECAST_COMPUTE_ATTACHED') != context.launch_id:
        # Its supervisor is what stops it, with everything it started, when the stage that
        # asked for it goes away; a worker run by hand has nobody doing that.
        raise RuntimeError('a compute worker is started by its supervisor')
    check_inputs(context.result_directory, document.get('input_files', {}))
    policy = document.get('policy') or execution_policy.resolve(context.threads, context.device_uuid)
    execution_policy.validate(policy, context.threads, context.device_uuid)
    execution_policy.apply(policy)
    # Retained artifacts may come from an interpreter with a newer pandas. This installs
    # a fallback that only engages when an ordinary load would otherwise raise, and it
    # runs after the policy gate, so it cannot affect policy-before-import ordering.
    from eforecast.common_utils.artifact_compat import install as install_artifact_compatibility
    install_artifact_compatibility()
    # Imports (including optional online targets) happen only after the policy is installed.
    module, name = document['entrypoint'].split(':', 1)
    token = _active.set(context)
    from eforecast import model_library
    model_library.reset()
    try:
        execution_policy.configure_torch(policy)
        target = getattr(importlib.import_module(module), name)
        value = target(document['payload'], context)
        inputs = model_library.records()
    except model_library.ModelPrerequisite as error:
        atomic_result(Path(context.result_directory) / 'prerequisite.json',
                      {'version': 1, 'context': asdict(context), 'prerequisite': error.to_dict()})
        raise
    finally:
        _active.reset(token)
    if not isinstance(value, dict):
        raise TypeError('compute target must return a JSON mapping')
    if inputs:
        value = dict(value, model_library=inputs)
    atomic_result(Path(context.result_directory) / 'worker_result.json',
                  {'version': 1, 'context': asdict(context), 'value': value})


if __name__ == '__main__':
    main(sys.argv[1])
