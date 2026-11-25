# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Host adapter for RBF fitting, activation and transfer. No networks load here."""
from contextlib import nullcontext
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import joblib

from eforecast.common_utils import file_stamps, lineage
from eforecast.common_utils.fuzzy_bundle import GENERATIONS, POINTER
from eforecast.common_utils.stage_context import current_stage
from eforecast.datasets.files_manager import FilesManager
from eforecast.folder_lock import folder_lock
from eforecast.training.command_line import resource_reporting
from .compute_workers import ComputeWorkers, artifact_files, failure_reason
from .executors import atomic_json
from . import run_reporting

log = logging.getLogger(__name__)


def gpu_pool():
    """CUDA order/visibility is resolved by the driver; NVML supplies physical capacity."""
    try:
        import pynvml as nvml
    except ImportError:
        return None
    try:
        nvml.nvmlInit()
        keys, totals = {}, {}
        # CUDA caches visibility at cuInit. A controller may have enumerated a
        # different mask previously; resolve the current environment in a fresh
        # stdlib-only process, without initializing CUDA in this host.
        probe = '''import json, sys
from eforecast.training.command_line.gpu_identity import cuda_uuid
keys = []
for ordinal in range(int(sys.argv[1])):
    try:
        keys.append('GPU-' + cuda_uuid(ordinal))
    except (OSError, RuntimeError):
        break
print(json.dumps(keys))
'''
        visible = json.loads(subprocess.check_output([sys.executable, '-c', probe,
            str(nvml.nvmlDeviceGetCount())], text=True, timeout=10))
        for ordinal, key in enumerate(visible):
            try:
                handle = nvml.nvmlDeviceGetHandleByUUID(key.encode())
            except nvml.NVMLError:
                continue
            keys[ordinal] = key
            totals[ordinal] = nvml.nvmlDeviceGetMemoryInfo(handle).total / 2 ** 20
        return SimpleNamespace(gpus=list(keys), key=keys.__getitem__, total=totals, declared_total=False) if keys else None
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, nvml.NVMLError) as error:
        log.info('GPU inventory unavailable; an auto preference resolves to the CPU: %s', error)
        return None


def _plan(clusterer, operation):
    """``(threads, device, execution)`` for the clusterer of a case.

    The threads are what the case gives the clusterer (``clustering.n_jobs``, else ``n_cpus``,
    else one). The device is the visible card its preference names, the one with the most free
    memory for ``auto``; a preference no visible card answers is the CPU.
    """
    static = clusterer.static_data
    configured = static['clustering'].get('n_jobs')
    if configured is None:
        configured = static.get('n_cpus')
    if configured is None:
        configured = 1
    available = resource_reporting.cpu_slots()
    threads = max(1, min(int(configured) if int(configured) > 0 else available, available))
    choice = os.environ.get('EFORECAST_FUZZY_DEVICE', static['clustering']['params'].get('device', 'auto'))
    from .model_compute import device_for
    # Carrying the rules of a fitted clusterer over to another case is not network work.
    device = None if operation == 'transfer' else device_for(choice)
    if device is None and operation != 'transfer' and str(choice).lower() not in {'cpu', 'none', 'auto'}:
        log.warning('the clusterer asks for device %r and no visible GPU answers it; it runs on the CPU', choice)
    return threads, device, {'device': choice if device else 'cpu', 'threads': configured,
                             'n_jobs': configured, 'loader_workers': 0}


def promote(worker, handle, result, base):
    """Copy the worker's verified output into a generation of its own, then point at it."""
    _, spec = worker.executor._spec(handle)
    context = spec['compute']['context']
    if result.get('context') != context or result.get('state') != 'success':
        raise ValueError('fuzzy promotion has a foreign or unfinished result')
    source = Path(spec['compute']['context']['result_directory'])
    files = result.get('files') or {}
    if not files or artifact_files(source, list(files)) != files:
        raise ValueError('fuzzy result has no verified bundle payload')
    base = Path(base)
    destination = base / GENERATIONS / context['launch_id']
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name('.copy-' + uuid.uuid4().hex)
    temporary.mkdir()
    expected = {}
    try:
        for name, record in files.items():
            relative = Path(name).relative_to('artifacts/bundle')
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, path)
            if path.stat().st_size != record['bytes'] or lineage.digest_file(str(path)) != record['sha256']:
                raise ValueError('fuzzy copy differs from the accepted worker output')
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
            expected[relative.as_posix()] = record
        if destination.exists():
            if destination.is_symlink() or any(p.is_symlink() for p in destination.rglob('*')):
                raise ValueError('unsafe existing fuzzy generation')
            if any(not (destination / name).is_file() or (destination / name).stat().st_size != rec['bytes']
                   or lineage.digest_file(str(destination / name)) != rec['sha256'] for name, rec in expected.items()):
                raise ValueError('existing fuzzy generation differs; refusing to overwrite it')
        else:
            os.replace(temporary, destination)
        stamps = {name: file_stamps.stamp(destination / name, follow=False) for name in expected}
        record = {'version': 1, 'generation': context['launch_id'], 'files': expected,
                  'producer': {k: context[k] for k in ('run_id', 'task_id', 'attempt_id', 'launch_id')}}
        manifest = destination / 'bundle_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != record:
            raise ValueError('fuzzy generation manifest already has different output')
        atomic_json(manifest, record)
        stamps['bundle_manifest.json'] = file_stamps.stamp(manifest, follow=False)
        for directory in [*sorted((p for p in destination.rglob('*') if p.is_dir()), reverse=True),
                          destination, destination.parent]:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        run_reporting.check_cancelled()
        for name, before in stamps.items():
            # The files were written a moment ago: in a synced folder each is marked as uploaded
            # about now, which moves its change time and is not a change (``file_stamps``).
            if not file_stamps.same(before, file_stamps.stamp(destination / name, follow=False)):
                raise ValueError('fuzzy generation changed before pointer publication')
        atomic_json(base / POINTER, record)
        return record
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def run(clusterer, operation, *, cv_masks=None, with_predictions=False, paths=None, fuzzy_file=None, workers=None):
    stage = current_stage.get()
    canonical = str(stage.base) if stage is not None else str(Path(clusterer.static_data['path_model']).absolute())
    # Direct callers share the existing case writer boundary. Managed callers already own it.
    from .stage_artifacts import DIRECTORY
    lock = nullcontext() if stage is not None else folder_lock(Path(canonical) / DIRECTORY, timeout=2)
    with lock:
        return _run(clusterer, operation, canonical, cv_masks=cv_masks, with_predictions=with_predictions,
                    paths=paths, fuzzy_file=fuzzy_file, workers=workers)


def _run(clusterer, operation, canonical, *, cv_masks, with_predictions, paths, fuzzy_file, workers):
    workers = workers or ComputeWorkers()
    threads, device, execution = _plan(clusterer, operation)
    ident = uuid.uuid4().hex
    ctx = run_reporting.context()
    run_id = ctx['run_id'] if ctx else 'fuzzy:' + ident
    scope = hashlib.sha256(os.path.realpath(canonical).encode()).hexdigest()
    task = f'fuzzy:{scope}:' + ('model' if operation in {'fit', 'transfer'} else 'activate:' + ident)
    payload = {'operation': operation, 'train': clusterer.train, 'online': clusterer.online,
               'bundle': clusterer.bundle, 'canonical_model': canonical,
               'destination_base': str(Path(canonical) / 'cluster_organizer' / 'RBF'),
               'paths': [os.fspath(p) for p in paths] if paths is not None else None,
               'fuzzy_file': os.fspath(fuzzy_file) if fuzzy_file is not None else None}
    manifest = Path(clusterer.bundle) / 'bundle_manifest.json'
    if operation == 'activate' and Path(clusterer.bundle).parent.name == GENERATIONS:
        record = json.loads(manifest.read_text())
        files = record.get('files')
        required = {'fuzzy_model.pickle', *[f'RBF_fuzzy_net_{i}/{name}'
                    for i in range(len(clusterer.fuzzy_models or []))
                    for name in ('net_weights.pickle', 'net_weights.pt')]}
        if (record.get('version') != 1 or record.get('generation') != Path(clusterer.bundle).name
                or not isinstance(files, dict) or not required.issubset(files)):
            raise ValueError('published fuzzy generation has an incomplete manifest')
        payload['expected_bundle'] = files
    inputs = {'static.pickle': pickle.dumps(clusterer.static_data, protocol=5)}
    if cv_masks is not None:
        inputs['masks.pickle'] = pickle.dumps(cv_masks, protocol=5)
    request = workers.request(run_id, task, ident, threads, device=device)
    handle = workers.prepare(request, 'eforecast.clustering.fuzzy_worker:execute', payload,
                             execution=execution, input_files=inputs)
    try:
        result = workers.wait(handle, timeout=None)
        if result['state'] != 'success':
            raise resource_reporting.ResourceUnavailable(f'fuzzy {operation} {failure_reason(result)}')
        if operation in {'fit', 'transfer'}:
            promote(workers, handle, result, clusterer.path_fuzzy)
            value = None
        else:
            # The worker owns model/data computation; this reads the frames it left.
            predicted, activated = joblib.load(workers.root(handle) / 'artifacts' / 'activations.pickle')
            value = (predicted, activated) if with_predictions else activated
        workers.discard_result(handle)
        return value
    except BaseException:
        workers.cancel(handle, force=True)
        raise
