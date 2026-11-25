# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Inference target for the neutral worker envelope; no training imports."""
import hashlib
import json
from pathlib import Path
import resource

import joblib

from eforecast.compute_worker import current_worker
from eforecast.compute_context import inference_sites, inference_input_records
from eforecast.common_utils import file_stamps
from eforecast.inference_inputs import observations, configuration_state, changed, moved
from eforecast.inference_view import View


def discover(statics, configs, bundles):
    """Legacy GRAPH markers deserialize only after attachment, never in a host/API."""
    from eforecast.init.initialize import initializer
    from eforecast.common_utils.graph_bundle import bundle_root
    from eforecast.common_utils import artifact
    result = dict(statics)
    for original in list(statics.values()):
        if original.get('_graph_serving'):
            continue  # the host already expanded the immutable neighbour configurations
        for trial in Path(original['path_model']).glob('global/GRAPH/test_*'):
            verdict, reason = artifact.inspect(trial)
            if verdict == 'rejected':
                continue
            marker = joblib.load(trial / 'net_weights.pickle')
            folder = bundle_root(marker['site_graph'])
            bundles.add(str(folder))
            meta = json.loads((folder / 'meta.json').read_text())
            for name, config in meta.get('configs', {}).items():
                if name not in result:
                    before = configuration_state(config)
                    result[name] = initializer(config)
                    if changed(before, configuration_state(config)):
                        raise ValueError('referenced graph configuration changed during initialization')
                    configs.update(before)
    return result


def load_predictions(static, *, selected=False):
    from eforecast.combine_predictions.ensemble import trial_specs
    out = {}
    for spec in trial_specs(static, selected_only=selected):
        file = Path(spec['path_weights']) / 'predictions_eval.pickle'
        if file.is_file():
            out[spec['path_weights']] = joblib.load(file)
    return out


def combination(operation, static, arguments):
    from eforecast.combine_predictions.ensemble import Ensemble, TrialPredictor, predict_trials, select_trials, predict_online
    if operation == 'trial':
        predictor = TrialPredictor(static, online=arguments.get('online', False))
        spec = arguments['spec']
        prediction = predictor.predict(spec, refit=arguments.get('refit', False))
        return {'prediction': prediction, 'quantiles': predictor.quantiles_of(spec)}
    if operation == 'quantiles':
        return TrialPredictor(static, online=arguments.get('online', False)).quantiles_of(arguments['spec'])
    if operation in {'predict', 'trials'}:
        return predict_trials(static, refit=arguments.get('refit', False), selected_only=arguments.get('selected_only', False),
                              cards=arguments.get('cards'))
    if operation in {'select', 'calibrate', 'evaluate'}:
        predictions = arguments.get('predictions', arguments.get('trial_predictions'))
        if predictions is None:
            predictions = load_predictions(static)
        if operation == 'select':
            return select_trials(static, predictions=predictions, refit=arguments.get('refit', False))
        result = arguments.get('result')
        if result is None:
            result = joblib.load(Path(static['path_model']) / 'combined_eval.pickle')
        return getattr(Ensemble(static), operation)(result, trial_predictions=predictions)
    if operation in {'combine', 'ensemble'}:
        result = Ensemble(static, online=arguments.get('online', False)).run(
            fit=arguments.get('fit', False), refit_predictions=arguments.get('refit_predictions', False))
        if operation == 'combine':
            joblib.dump(result, Path(static['path_model']) / 'combined_eval.pickle')
        return result
    if operation == 'online':
        return predict_online(static, path=arguments.get('path'), intervals=arguments.get('intervals'))
    raise ValueError(f'unknown inference operation {operation}')


def compute(values, context, *, validate=None):
    from eforecast.graph_serving import locations
    with locations(values['statics'][values['main']]):
        return _compute(values, context, validate=validate)


def _compute(values, context, *, validate=None):
    import pandas as pd
    from eforecast.common_utils import lineage
    from eforecast.bounded_pools import cancel_check
    from eforecast.shallow_models.foundation_model import clear_pipelines
    root = Path(context.result_directory)
    operation = values['operation']
    online = operation in {'forecast', 'forecast_file', 'service', 'online'} or values['arguments'].get('online', False)
    statics = values['statics']
    supplied = values.get('configs', {})
    configs = dict(supplied) if isinstance(supplied, dict) else {p: values['observations'][p] for p in supplied}
    bundles = set(values.get('bundles', []))
    problem = moved('inference inputs changed before worker loading', values['observations'],
                    observations(statics, configs, bundles))
    if problem:
        raise ValueError(problem)
    if online:
        statics = discover(statics, configs, bundles)
    before = observations(statics, configs, bundles)
    if any(not file_stamps.same(stamp, before.get(path)) for path, stamp in configs.items()):
        raise ValueError('inference configuration changed after initialization')
    dates = values['arguments'].get('dates')
    views = {}
    for name, static in statics.items():
        view = View(static, root / 'views' / hashlib.sha256(name.encode()).hexdigest(), online=online)
        if online:
            view.external('path_group_nwp')
            if static.get('use_image'):
                # Pixel caches and downloads are writable. Keep copies private;
                # source window filtering happens before copying below.
                view.pixels(dates)
                view.satellite(dates)
        views[name] = view
    private = {name: view.static for name, view in views.items()}
    main = views[values['main']]
    arguments = main.rebase(values['arguments'])
    token = inference_sites.set(private)
    input_records = []
    input_token = inference_input_records.set(input_records)
    # The supervisor handles force cancellation. These callbacks additionally
    # make bounded Python loops cooperative without importing host reporting.
    def check():
        if (root / 'cancel.flag').exists():
            raise InterruptedError('inference cancelled')
    cancel = cancel_check.set(check)
    output = root / 'artifacts'
    output.mkdir(exist_ok=True)
    try:
        check()
        if operation in {'forecast', 'forecast_file'}:
            from eforecast_online.forecast import forecast
            result = forecast(arguments['dates'], None, path=arguments.get('path'),
                intervals=arguments.get('intervals'), static_data=main.static)
            if operation == 'forecast_file':
                frame, _ = result
                file = output / ('forecast.csv' if arguments.get('file') else 'forecast.txt')
                if arguments.get('file'):
                    frame.to_csv(file)
                else:
                    file.write_text(frame.round(3).to_string() + '\n')
                result = {'rows': len(frame)}
        elif operation == 'service':
            from eforecast_online.service import run
            arguments = {k: v for k, v in arguments.items() if k not in {'out_dir', 'dates'}}
            result = run(None, static_data=main.static, out_dir=str(output / 'service'), **arguments)
            if result['state'] not in {'prepared', 'rejected'}:
                # A rejected forecast is a failed job: it produced no publishable number.
                # Carry the publication's own reason so the host does not have to guess.
                raise ValueError(result.get('error') or
                                 '; '.join(result.get('publication', {}).get('detail') or ()) or
                                 'operational inference failed')
            canonical_folder = Path(values['output_folder'])
            result['files'] = [str(canonical_folder / Path(p).name) for p in result['files']]
            result['compute'] = {'launch_id': context.launch_id, 'device_uuid': context.device_uuid,
                                 'cpu_threads': context.threads}
            (output / 'service/status.json').write_text(json.dumps(result, indent=1))
        else:
            from eforecast.combine_predictions import ensemble
            ensemble.said.clear()
            result = combination(operation, main.static, arguments)
        check()
        problem = moved('inference inputs changed during execution', before, observations(statics, configs, bundles))
        if problem:
            raise ValueError(problem)
        validated = validate(main, result) if validate else {}
        # Canonical identities and paths are restored before the host sees any
        # payload; immutable model manifests are never relabelled as new trials.
        result = main.rebase(result, canonical=True)
        main.canonicalize()
        validated = {name: checksum for name, checksum in validated.items()
                     if lineage.digest_file(str(main.model / Path(name).relative_to(main.base))) == checksum}
        model = output / 'model'
        changed, deleted = [], []
        if not online:
            for file in main.files(main.model):
                relative = file.relative_to(main.model)
                original = main.base / relative
                checksum = lineage.digest_file(str(file))
                if original.is_file() and lineage.digest_file(str(original)) == checksum:
                    continue
                target = model / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copyfile(file, target)
                changed.append(relative.as_posix())
            deleted = [p.as_posix() for p in main.writes if (main.base / p).is_file()
                       and not (main.model / p).exists()]
        joblib.dump(result, output / 'result.pickle')
        dependencies = {name: lineage.dependencies(sd, selectors=False) for name, sd in statics.items()}
        evidence = {'statics': statics, 'configs': configs, 'bundles': sorted(bundles),
                    'observations': before, 'dependencies': dependencies, 'validated': validated}
        evidence['online_inputs'] = input_records
        query = values['arguments']
        evidence['request'] = {'operation': operation, 'path': list(query.get('path') or []),
            'intervals': list(query.get('intervals') or []),
            'dates': [str(d) for d in pd.DatetimeIndex(query['dates'])] if query.get('dates') is not None else [],
            **{k: query[k] for k in ('at', 'days', 'back', 'skip_nwp', 'skip_sat') if k in query}}
        joblib.dump(evidence, output / 'evidence.pickle')
        import torch
        peak_gpu = torch.cuda.max_memory_allocated() / 2 ** 20 if context.device_uuid else 0.
        from eforecast.combine_predictions import ensemble
        return {'operation': operation, 'changed': changed, 'deleted': deleted, 'said': list(ensemble.said),
                'artifacts': [p.relative_to(root).as_posix() for p in View.files(output)],
                'device_uuid': context.device_uuid, 'cpu_threads': context.threads,
                'peak_gpu_allocated_mb': peak_gpu,
                'peak_host_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    finally:
        clear_pipelines()
        cancel_check.reset(cancel)
        inference_sites.reset(token)
        inference_input_records.reset(input_token)


def execute(payload, context):
    if current_worker() != context:
        raise RuntimeError('inference requires confirmed attachment')
    values = joblib.load(Path(context.result_directory) / 'inputs/inference.pickle')
    return compute(values, context)
