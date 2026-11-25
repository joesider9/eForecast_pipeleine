# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Fuzzy operation targets. Imported only after common-worker attachment and policy."""
import json
import os
from copy import deepcopy
from pathlib import Path
import shutil

import joblib
import torch

from eforecast.compute_worker import atomic_result, current_worker
from eforecast.common_utils import lineage
from eforecast.clustering.tf_rbf_clusterer import TfRBFClusterer


def _rebase(value, mappings):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, str):
        for old, new in mappings:
            if value == old or value.startswith(old + '/'):
                return new + value[len(old):]
        return value
    if isinstance(value, dict):
        return {_rebase(k, mappings): _rebase(v, mappings) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebase(v, mappings) for v in value]
    if isinstance(value, tuple):
        return tuple(_rebase(v, mappings) for v in value)
    return value


def _input_view(source, destination):
    """Isolate runtime metadata/caches; large dataset inputs remain read links."""
    destination.mkdir(parents=True)
    mutable = {'transformers.pickle', 'partition_report.json', 'scalers.pickle', 'imputers.pickle',
               'scalers_provenance.json', 'scalers_fingerprint.json'}
    for item in Path(source).iterdir():
        if item.name == '.stage-artifacts':
            continue
        target = destination / item.name
        if item.name == 'DATA' and item.is_dir():
            target.mkdir()
            for data in item.iterdir():
                if data.name == 'day_light_hours.csv':
                    shutil.copy2(data, target / data.name)
                else:
                    (target / data.name).symlink_to(data.resolve(), target_is_directory=data.is_dir())
        elif item.name in mutable and item.is_file():
            shutil.copy2(item, target)
        elif not item.name.endswith('.lock'):
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())


def _metadata(file):
    if file.name != 'net_weights.pickle':
        return joblib.load(file)
    from eforecast.deep_models.pytorch_2x.network import DeepNetwork, load_tolerant
    try:
        state = joblib.load(file)
    except AttributeError:
        # Same legacy reading as DeepNetwork.load: older CLI saves included
        # __main__ runtime objects. They are not model inputs or portable state.
        state = load_tolerant(file)
    if not isinstance(state, dict):
        raise ValueError('fuzzy network metadata must be a mapping')
    return {key: value for key, value in state.items() if key not in DeepNetwork.UNSAVED}


def execute(payload, context):
    if current_worker() != context:
        raise RuntimeError('a fuzzy operation runs in its own worker process')
    root = Path(context.result_directory)
    original = joblib.load(root / 'inputs' / 'static.pickle')
    model_view = root / 'work' / 'model'
    _input_view(original['path_model'], model_view)
    static = _rebase(deepcopy(original), [(original['path_model'], str(model_view))])
    static['n_cpus'], static['n_jobs'] = context.threads, 1
    static['clustering']['n_jobs'] = 1
    operation = payload['operation']
    before = lineage.input_observations(static, static['path_model'])
    outputs = root / 'artifacts'
    outputs.mkdir()
    local = TfRBFClusterer(static, train=payload['train'], online=payload['online'], refit=operation == 'fit')
    if operation == 'fit':
        bundle = outputs / 'bundle'
        bundle.mkdir()
        local._bundle_override = str(bundle)
        local._fit(joblib.load(root / 'inputs' / 'masks.pickle'))
    elif operation == 'transfer':
        bundle = outputs / 'bundle'
        bundle.mkdir()
        inputs = [Path(payload['fuzzy_file'])]
        sources = [Path(source).resolve(strict=True) for source in payload['paths']]
        for source in sources:
            inputs.extend(p for p in source.iterdir() if p.is_file())
        signatures = {str(p): lineage.digest_file(str(p)) for p in inputs}
        shutil.copyfile(payload['fuzzy_file'], bundle / 'fuzzy_model.pickle')
        for index, source in enumerate(sources):
            target = bundle / f'RBF_fuzzy_net_{index}'
            target.mkdir()
            for file in source.iterdir():
                if file.is_file():
                    shutil.copyfile(file, target / file.name)
        if signatures != {str(p): lineage.digest_file(str(p)) for p in inputs}:
            raise ValueError('fuzzy transfer source changed during copying')
        local._bundle_override = str(bundle)
        local.load()
        if not local.is_trained or len(local.fuzzy_models or []) != len(payload['paths']):
            raise ValueError('transfer source does not contain a complete trained fuzzy bundle')
        for index in range(len(local.fuzzy_models)):
            for name in ('net_weights.pickle', 'net_weights.pt'):
                if not (bundle / f'RBF_fuzzy_net_{index}' / name).is_file():
                    raise ValueError('transfer source is missing fuzzy weights')
    elif operation == 'activate':
        local._bundle_override = payload['bundle']
        local.load()
        local.train, local.online = payload['train'], payload['online']
        # Pin legacy sources as well as immutable generations across model loading.
        source = Path(payload['bundle'])
        names = ['fuzzy_model.pickle', *[f'RBF_fuzzy_net_{i}/{name}' for i in range(len(local.fuzzy_models or []))
                                      for name in ('net_weights.pickle', 'net_weights.pt')]]
        def observed():
            return {name: lineage.digest_file(str(source / name)) for name in names}
        for name, record in payload.get('expected_bundle', {}).items():
            file = source / name
            if (Path(name).is_absolute() or '..' in Path(name).parts or file.is_symlink()
                    or file.stat().st_size != record['bytes'] or lineage.digest_file(str(file)) != record['sha256']):
                raise ValueError('fuzzy source differs from its published payload')
        model_before = observed()
        predictions, activations = local._compute_activations(with_predictions=True)
        if model_before != observed():
            raise ValueError('fuzzy model changed during activation; result was not published')
        joblib.dump((predictions, activations), outputs / 'activations.pickle')
        bundle = None
    else:
        raise ValueError('unknown fuzzy operation')
    if lineage.inputs_changed(before, lineage.input_observations(static, static['path_model'])):
        raise ValueError('fuzzy inputs changed during compute; result was not published')
    if bundle is not None:
        # Persisted paths must survive the removal of the worker's folder and the stage's own
        # canonicalization. The producer's input view remains read-only to this worker.
        from eforecast.common_utils.fuzzy_bundle import GENERATIONS
        destination = str(Path(payload['destination_base']) / GENERATIONS / context.launch_id)
        mappings = [(str(bundle), destination),
                    (static['path_model'], payload['canonical_model']),
                    (original['path_model'], payload['canonical_model'])]
        if operation == 'transfer':
            mappings = [(str(source), str(Path(destination) / f'RBF_fuzzy_net_{i}'))
                        for i, source in enumerate(sources)] + [
                        (str(Path(payload['fuzzy_file']).parent), destination)] + mappings
        for file in bundle.rglob('*'):
            if file.is_symlink():
                raise ValueError('fuzzy bundle cannot contain symlinks')
            if file.name in {'fuzzy_model.pickle', 'net_weights.pickle', 'parameters.pickle'}:
                joblib.dump(_rebase(_metadata(file), mappings), file)
            elif file.suffix == '.json':
                atomic_result(file, _rebase(json.loads(file.read_text()), mappings))
    files = [p.relative_to(root).as_posix() for p in sorted(outputs.rglob('*')) if p.is_file()]
    return {'operation': operation, 'artifacts': files,
            'rules': local.rule_names, 'fuzzy_networks': len(local.fuzzy_models or []),
            'worker': {'pid': os.getpid(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                       'torch_threads': torch.get_num_threads(), 'cuda_initialized': torch.cuda.is_initialized(),
                       'cuda_uuid': str(torch.cuda.get_device_properties(0).uuid) if torch.cuda.is_initialized() else None}}
