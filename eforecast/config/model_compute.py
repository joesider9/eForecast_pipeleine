# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Work on pretrained models -- foundation families, image encoders -- in a process of its own.

The model is loaded, used and dropped inside one worker, so that nothing of it stays resident
in the stage that asked. Nothing is reserved for the worker: it is given some threads and, when
the preference asks for one, the visible card with the most free memory at that moment.
"""
import hashlib
import logging
import os
from pathlib import Path
import pickle
import shutil
import time
import uuid

from eforecast.bounded_pools import capacity
from eforecast.compute_context import dataset_backend, install_resident_backend
from eforecast.training.command_line.resource_reporting import ResourceUnavailable
from .compute_workers import ComputeWorkers, failure_reason
from . import run_reporting

log = logging.getLogger(__name__)


def model_workers(requested='cpu'):
    """The launcher of this process: the one a caller injected, or a new one."""
    injected = dataset_backend.get()
    return injected if injected is not None else ComputeWorkers()


def amount(name, default):
    value = float(os.environ.get(name, 0))
    return value if value > 0 else default


def preview_estimate(document, tag, method=None):
    """Metadata-only model-residence allowance, without importing model libraries.

    Count independent backbones, including paired transformers and one vision
    model per lag. Generic allowances need calibration for custom architectures;
    a positive whole-preview override supports those without rewriting a case.
    """
    from .schema import NetworkExperimentConfig
    from .loader import _inline_experiment_layer_params
    from .graph_service import _resolve_params, data_shapes
    experiment = NetworkExperimentConfig.model_validate(document['experiments'][tag])
    layers = _inline_experiment_layer_params({tag: experiment}, resolve=True)[tag]
    family = _resolve_params(document, method, tag)
    shapes = data_shapes(document)['shapes']
    image_lags = [v.get('lags') for v in (document.get('data') or {}).get('variables', []) if v.get('type') == 'image']
    shapes.setdefault('images', [max([len(v) if isinstance(v, list) else v
                                     for v in image_lags if isinstance(v, (list, int)) and v] + [1])])
    backbones = []
    for branch, values in layers.items():
        for layer in values:
            params = {**family, **{name: value for name, value in layer['params'].items()
                                   if value not in (None, '')}}      # an empty one is not set, as in the builder
            kind, name = layer['type'], params.get('model_name')
            count, per_model = 1, 0
            if kind in {'transformer', 'bi_transformer'} and name in {'Chronos', 'Chronos2'}:
                count = 2 if kind == 'bi_transformer' else 1
                per_model = 4096
            elif kind in {'vit_net', 'timm_net', 'time_distr_vit_net'}:
                shape = shapes.get(branch.split('/')[-1])
                count = max(1, int(shape[0])) if kind == 'time_distr_vit_net' and isinstance(shape, list) else 1
                per_model = 4096
            elif kind == 'unet':
                name, per_model = params.get('encoder_name'), 2048
            if per_model:
                backbones.append({'branch': branch, 'layer': layer['name'], 'model': name,
                                  'count': count, 'ram_mb': count * per_model})
    ram = amount('EFORECAST_PREVIEW_RAM_MB', 2048 + sum(b['ram_mb'] for b in backbones))
    return {'ram_mb': ram, 'backbones': backbones, 'device': 'cpu', 'cpu_threads': 1}


_inventory = {'at': None, 'pool': None, 'mask': None}


def _cards():
    """The visible cards by UUID; the driver is asked again at most every thirty seconds.

    And at once when this process is shown other cards (``CUDA_VISIBLE_DEVICES`` changed): what
    was seen under another mask says nothing about this one.
    """
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    if _inventory['at'] is None or mask != _inventory['mask'] or time.monotonic() - _inventory['at'] > 30:
        from .fuzzy_compute import gpu_pool
        _inventory.update(at=time.monotonic(), pool=gpu_pool(), mask=mask)
    return _inventory['pool']


def _free_mb(key):
    try:
        import pynvml as nvml
        return nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByUUID(key.encode())).free / 2 ** 20
    except Exception:           # noqa: BLE001 - a card that cannot be asked is simply not preferred
        return 0.


def cards_for(requested, *, exclude=()):
    """The visible cards the preference ``requested`` names, the one with the most free memory first.

    ``cpu`` and ``None`` name none. ``auto`` names every visible card; ``cuda`` and ``cuda:N``
    name a card by its order, and a ``GPU-``/``MIG-`` UUID names it outright. Cards in
    ``exclude`` are passed over.
    """
    spelling = str(requested).lower() if requested is not None else 'cpu'
    if spelling in {'cpu', 'none'}:
        return []
    pool = _cards()
    keys = [pool.key(ordinal) for ordinal in pool.gpus] if pool else []
    if spelling != 'auto':
        if str(requested).startswith(('GPU-', 'MIG-')):
            keys = [key for key in keys if key == requested]
        else:
            try:
                ordinal = int(spelling.split(':')[-1]) if spelling != 'cuda' else 0
                keys = [pool.key(ordinal)] if pool and ordinal in pool.gpus else []
            except ValueError:
                keys = []
    return sorted((key for key in keys if key not in exclude), key=_free_mb, reverse=True)


def device_for(requested, *, exclude=()):
    """The card a worker is started on for the preference ``requested``; ``None`` is the CPU.

    The first of ``cards_for``: for ``auto``, the visible card with the most free memory right
    now. A preference that no visible card answers is the CPU.

    Nothing is taken from the card: the driver gives the worker memory as it asks for it, and
    a worker that asks for more than is there fails with the driver's own error.
    """
    keys = cards_for(requested, exclude=exclude)
    return keys[0] if keys else None


def foundation_cards(static):
    """``{family: [card, ...]}`` for the foundation families a case has switched on that ask for a card.

    What the predict stage is offered: its worker runs on the CPUs and predicts the trials of
    these families in readers of their own, one a card (``ensemble.predict_trials``). A family
    that says ``cpu``, or names a card that is not there, is not in it and is predicted on the
    CPUs like any other trial.
    """
    from .capabilities import FOUNDATION_METHODS, foundation_device
    enabled = {method for group in ('global_methods', 'methods_cpu')
               for method, on in (static.get(group) or {}).items() if on}
    offered = {}
    for method in FOUNDATION_METHODS:
        if method in enabled:
            cards = cards_for(foundation_device(static, method))
            if cards:
                offered[method] = cards
    return offered


def image_plan(static, *, serial=False):
    """``(threads, device, payload)`` for the image encoder of a case."""
    variables = [v for v in static.get('variables', []) if v.get('type') == 'image']
    batch = max(1, min(32, int(amount('EFORECAST_IMAGE_BATCH_SIZE', 32))))
    requested = os.environ.get('EFORECAST_IMAGE_DEVICE', 'auto')
    device = device_for(requested)
    wanted = int(static.get('n_jobs') or 1)
    threads = 1 if serial else max(1, min(capacity(), wanted if wanted > 0 else capacity()))
    return threads, device, {'batch_size': batch, 'resident_encoders': 1, 'requested_device': requested,
                             'variables': len(variables)}


def image_stats(creator, *, workers=None):
    """The encoder phase of the image dataset, in a worker of its own."""
    workers = workers or model_workers()
    threads, device, info = image_plan(creator.static_data, serial=not creator.parallel)
    inputs = {'image.pickle': pickle.dumps({'static': creator.static_data, 'dates': creator.dates,
        'transformer': creator.transformer, 'online': creator.is_online, 'parallel': creator.parallel}, protocol=5)}
    ident = uuid.uuid4().hex
    task = 'image-stats:' + hashlib.sha256(str(Path(creator.static_data['path_model']).absolute()).encode()).hexdigest()
    request = workers.request('image-stats:' + ident, task, ident, threads if device is None else 1, device=device)
    handle = workers.prepare(request, 'eforecast.datasets.image_data.encoder_worker:execute', info,
        input_files=inputs, execution={'device': info['requested_device'] if device else 'cpu',
                                       'threads': threads if device is None else 1})
    try:
        result = workers.wait(handle, timeout=None)
        if result['state'] != 'success':
            raise ResourceUnavailable(f'image encoder {failure_reason(result)}')
        import joblib
        values = joblib.load(workers.root(handle) / 'artifacts/image-stats.pickle')
        creator.transformer.__dict__.update(values['transformer'])
        workers.discard_result(handle)
        return values['stats']
    except BaseException:
        workers.cancel(handle, force=True)
        raise


def foundation(model, operation, *, masks=None, dates=None, workers=None):
    params = dict(model.params)
    requested = params.get('device') or 'auto'
    workers = workers or model_workers()
    requested_batch = int(params.get('batch_size') or 256)
    # Chronos cross-learning cohorts must not silently change with a memory cap.
    batch = max(1, requested_batch) if params.get('cross_learning') else max(1, min(256, requested_batch))
    params['batch_size'] = batch
    threads = max(1, min(capacity(), int(model.n_jobs or 1)))
    device = device_for(requested)
    if device is None and str(requested).lower() not in {'cpu', 'none'}:
        log.warning('foundation %s: no visible GPU answers the preference %r; it runs on the CPU',
                    model.method, requested)
    payload = {'operation': operation, 'requested_batch_size': requested_batch, 'batch_size': batch}
    inputs = {'model.pickle': pickle.dumps({'static': model.static_data, 'params': params,
        'masks': masks, 'dates': dates, 'online': model.online, 'train': model.train}, protocol=5)}
    ident = uuid.uuid4().hex
    canonical = Path(model.path_weights).absolute()
    if canonical.name.startswith('.attempt-cpu-'):
        canonical = canonical.parent
    task = 'foundation:' + hashlib.sha256(str(canonical).encode()).hexdigest() + ':' + operation
    ctx = run_reporting.context()
    request = workers.request(ctx['run_id'] if ctx else 'foundation:' + ident, task, ident, threads, device=device)
    handle = workers.prepare(request, 'eforecast.shallow_models.foundation_worker:execute', payload,
        input_files=inputs, execution={'device': requested if device else 'cpu', 'threads': threads})
    try:
        result = workers.wait(handle, timeout=None)
        if result['state'] != 'success':
            raise ResourceUnavailable(f'foundation {model.method} {failure_reason(result)}')
        root = workers.root(handle)
        import joblib
        if operation == 'fit':
            source = root / 'artifacts/net_weights.pickle'
            state = joblib.load(source)
            target = Path(model.path_weights) / 'net_weights.pickle'
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name('.foundation-' + ident)
            try:
                shutil.copyfile(source, temporary)
                from eforecast.common_utils.lineage import digest_file
                if digest_file(str(temporary)) != result['files']['artifacts/net_weights.pickle']['sha256']:
                    raise ValueError('foundation metadata changed during handoff')
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            model.__dict__.update(state)
            value = result['value']['score']
        else:
            forecasts = joblib.load(root / 'artifacts/predictions.pickle')
            model.quantile_predictions = forecasts['quantiles']
            value = forecasts['median']
        workers.discard_result(handle)
        return value
    except BaseException:
        workers.cancel(handle, force=True)
        raise


# Registration is intentionally host-side. The neutral inference modules never
# import this module or reach training packages in the CPU image.
install_resident_backend('foundation', foundation)
install_resident_backend('images', image_stats)
