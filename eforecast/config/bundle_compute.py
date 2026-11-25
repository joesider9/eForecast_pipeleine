# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Bundle qualification in a worker of its own; the UI/approval process never loads estimators."""
import json
import os
from pathlib import Path
import uuid

from . import model_compute


def placement():
    """``(workers, card or None, device preference the worker is told)`` for bundle work."""
    requested = os.environ.get('EFORECAST_BUNDLE_DEVICE', 'cpu')
    workers = model_compute.model_workers(requested)
    card = model_compute.device_for(requested)
    if requested not in ('cpu', 'none', 'auto') and card is None:
        raise ValueError('requested bundle qualification GPU is unavailable; no CPU fallback')
    return workers, card, requested if card else 'cpu'


def validate(root, *, images=None, fixture=None, _action="validate"):
    root = Path(root).resolve()
    workers, card, effective = placement()
    threads = min(2, model_compute.capacity())
    ident = uuid.uuid4().hex
    request = workers.request('bundle:' + ident, 'bundle-qualification', ident, threads, device=card)
    handle = workers.prepare(request, 'eforecast.model_bundle_validate:run',
        {'bundle': str(root), 'images': images, 'fixture': fixture, 'action': _action},
        execution={'device': effective, 'threads': threads})
    try:
        result = workers.wait(handle, timeout=float(os.environ.get('EFORECAST_BUNDLE_TIMEOUT', '600')))
        require_success(workers, handle, result)
        return json.loads((workers.root(handle) / 'artifacts/qualification.json').read_text())
    finally:
        workers.discard_result(handle)


def capture(static, **options):
    import pickle
    from . import graph_publication
    base = Path(static['path_model']).resolve()
    workers, card, effective = placement()
    ident = uuid.uuid4().hex
    threads = min(2, model_compute.capacity())
    with graph_publication.ownership({base / '.stage-artifacts'}):
        request = workers.request('reference:' + ident, 'bundle-reference', ident, threads, device=card)
        handle = workers.prepare(request, 'eforecast.bundle_reference:run', options,
            input_files={'static.pickle': pickle.dumps(static, protocol=5)},
            execution={'device': effective, 'threads': threads})
        try:
            result = workers.wait(handle, timeout=float(os.environ.get('EFORECAST_BUNDLE_TIMEOUT', '600')))
            require_success(workers, handle, result)
            return json.loads((workers.root(handle) / 'artifacts/reference.json').read_text())
        finally:
            workers.discard_result(handle)


def require_success(workers, handle, result):
    if result.get('state') == 'success':
        return
    from eforecast.model_bundle import BundleError
    log = Path(workers.executor._spec(handle)[1]['log'])
    detail = log.read_text(errors='replace')[-1800:] if log.is_file() else str(result.get('state'))
    raise BundleError('bundle worker failed before qualification: ' + detail)
