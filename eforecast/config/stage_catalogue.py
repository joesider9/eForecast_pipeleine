# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Read-only producer/consumer plans for the existing single-entry-point workflows.

This catalogue describes execution and external prerequisites. File presence is explicitly
legacy evidence, not A09-07's verified stage completion. It never fits, deletes or rewrites.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os

from .pipeline_stages import ALL_DATASET_STAGES, DATASET_STAGE_GROUPS, COMBINE_STAGES, MODEL_STAGES
from .capabilities import NWP_EXTRACTORS, CPU_METHODS, FOUNDATION_METHODS, NO_SEQUENCES, has_sequences
from .service import ServiceError

VERSION = 1
POLICIES = {
    'partition': 'one global whole-day partition; shared cluster samples retain its assignment',
    'fractions': 'configured train/validation/test fractions and cluster split policy are unchanged',
    'scalers': 'one globally fitted scaler family; never fit one family per cluster',
    'selection': 'internal test-based model selection and the separate later selection period are unchanged',
    'readiness': 'receipt content and inputs are verified at execution; absent receipts retain explicit legacy provenance',
}


class BlockedStage(ServiceError):
    pass


def ordered(nodes):
    """Stable topological order; reject unknown references and cycles before effects."""
    by_key = {node['key']: node for node in nodes}
    if len(by_key) != len(nodes):
        raise ValueError('duplicate stage key')
    pending, result = list(by_key), []
    for node in nodes:
        if any(dep not in by_key for dep in node['dependencies']):
            raise ValueError(f"unknown dependency of {node['key']}")
    while pending:
        ready = [key for key in pending if all(dep in result for dep in by_key[key]['dependencies'])]
        if not ready:
            raise ValueError('cyclic stage dependencies')
        result.append(ready[0])
        pending.remove(ready[0])
    return result


def _file(paths):
    candidates = [Path(path) for path in paths if path]
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size:
                return {'status': 'legacy', 'path': str(path), 'reason': 'payload exists; stage completion unverified'}
        except OSError as error:
            return {'status': 'unknown', 'path': str(path), 'reason': f'cannot inspect prerequisite: {error}'}
    return {'status': 'missing', 'paths': [str(path) for path in candidates], 'reason': 'required payload is missing'}


def _directory(paths):
    for value in paths:
        path = Path(value)
        try:
            if path.is_dir() and next(path.iterdir(), None) is not None:
                return {'status': 'legacy', 'path': str(path),
                        'reason': 'source directory contains data; date coverage and payload integrity are unverified'}
        except OSError as error:
            return {'status': 'unknown', 'reason': f'cannot inspect image source: {error}'}
    return {'status': 'missing', 'paths': paths, 'reason': 'satellite sources or processed image cache are missing'}


def _models(static, method=None, global_only=False):
    from eforecast.common_utils.completion import is_finished
    model = Path(static['path_model'])
    scopes = [model / 'global']
    if not global_only and static.get('is_Fuzzy'):
        clusters = model / 'clusters.pickle'
        if clusters.is_file():
            import joblib
            try:
                mapping = joblib.load(clusters)
                scopes.extend(Path(path) for path in mapping.values())
            except Exception as error:
                return {'status': 'unknown', 'reason': f'cannot inspect cluster model scopes: {error}'}
    reasons, usable_methods, first = [], set(), None
    for scope in scopes:
        sections = ('global_methods',) if scope == model / 'global' else ('cluster_methods', 'methods_cpu')
        enabled = ({method} if method else {name for section in sections
                   for name, on in (static.get(section) or {}).items() if on})
        for name in sorted(enabled):
            from eforecast.common_utils.trial_layout import method_directory
            directory = method_directory(static, scope, name)
            for trial in sorted(directory.glob('test_*')):
                if not trial.is_dir():
                    continue
                if not (trial / 'net_weights.pickle').is_file():
                    reasons.append(f'{trial}: missing net_weights.pickle')
                    continue
                usable, reason = is_finished(str(trial), name)
                if usable:
                    usable_methods.add(name)
                    if first is None:
                        first = {'status': 'legacy' if 'legacy' in reason else 'trial_completion',
                                 'path': str(trial), 'reason': reason}
                    break
                reasons.append(f'{trial}: {reason}')
    if first is not None:
        return {**first, 'methods': sorted(usable_methods)}
    return {'status': 'missing', 'reason': 'no admissible completed model' +
            (': ' + '; '.join(reasons[:3]) if reasons else '')}


def _other_configuration(static):
    """The families whose studies hold trials of another configuration, or None.

    A model is one configuration's. A family whose parameters or search space changed after it
    was trained is not part of that model until it is trained again, and predicting, selecting
    and combining with what it left would publish a mixture of two configurations under the
    name of one. So the combination step waits for the training step, as it waits for a
    dataset that was built from other variables.
    """
    try:
        from eforecast.training.command_line import revisions
        return revisions.retrained(revisions.plan(static, action='continue')) or None
    except Exception:                                   # noqa: BLE001 -- a plan never fails on a study
        return None


def other_configuration_reason(families):
    from eforecast.training.command_line import revisions
    names = ', '.join(families)
    return (f'the trained models of {names} are of another configuration of {"it" if len(families) == 1 else "them"} '
            f'-- {revisions.describe_retrained(families)} -- and are not part of this model; train '
            f'{"it" if len(families) == 1 else "them"} again, or put the parameters back')


def _choices(value):
    if isinstance(value, dict):
        return _choices(value.get('choices', value.get('value')))
    if isinstance(value, (list, tuple, set)):
        return [name for item in value for name in _choices(item)]
    return [value] if isinstance(value, str) else []


def _inputs(static, methods=None, *, cpu=False, pixels=False):
    experiments = static.get('experiments') or {}
    if methods and all(method == 'GRAPH' for method in methods):
        return ['target']  # GRAPH's own exporter supplies cached predictions, not neural inputs.
    if cpu:
        tags = set()
        if any(method in FOUNDATION_METHODS for method in methods or []):
            tags.add('lstm')
        if any(method not in FOUNDATION_METHODS for method in methods or []):
            tags.add('row_all')
    else:
        chosen = set()
        if methods is not None:
            for method in methods:
                chosen.update(_choices(static.get(method, {}).get('experiment_tag')))
        definitions = (experiments[name] for name in chosen if name in experiments) if chosen else experiments.values()
        tags = {tag for exp in definitions for tag in exp if tag not in {'output', 'hidden_layer'}}
    products = ['target']
    if not tags or any('row' in tag for tag in tags):
        products.append('row')
    if any('lstm' in tag for tag in tags):
        products.append('lstm')
    if any('nwp' in tag for tag in tags):
        products.append('nwp')
    if any('img' in tag or 'image' in tag or 'stats' in tag for tag in tags):
        products.append('images')
    if pixels and any(('img' in tag or 'image' in tag) and 'stats' not in tag for tag in tags):
        products.append('image_pixels')
    if 'row_all' in tags:
        if static.get('NWP') and 'nwp' not in products:
            products.append('nwp')
        if static.get('use_image') and 'images' not in products:
            products.append('images')
    return products


def _configured(value):
    if isinstance(value, dict) and 'choices' in value:
        return any(_configured(item) for item in value['choices'])
    # A search over choices reaches the plan as a set. ``{None}`` is "choices: [none]" -- what
    # the editor writes when a family follows a case that selects nothing -- and it read as
    # configured, so training went on requiring a selector the case had just stopped building.
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_configured(item) for item in value)
    if isinstance(value, dict) and value.get('tag') == 'fixed':
        return _configured(value.get('value'))
    return value is not None and value is not False


def _configuration_reference(value):
    # Transfer dictionaries may legitimately contain dates/sets from a stored case.
    # Reuse A07's typed encoding rather than rejecting them when the DAG is pinned.
    from .run_inputs import encode
    return encode(value)


def _methods(static, options):
    if options.get('cpu'):
        from eforecast.common_utils.cpu_scope import methods
        return methods(static, options.get('cpu_scope'), options.get('cpu_method'))
    cluster = [options['cluster_method']] if options.get('cluster_method') else [
        name for name, on in (static.get('cluster_methods') or {}).items() if on]
    glob = [options['global_method']] if options.get('global_method') else [
        name for name, on in (static.get('global_methods') or {}).items() if on]
    if not options.get('global_method'):
        glob += [name for name in cluster if static.get(name, {}).get('transfer_learning_from_global') and name not in glob]
    return [m for m in glob if m not in CPU_METHODS], cluster


def resolve(static_data, script, options, *, inspect=True, selected_keys=None, completion_version='current'):
    """Resolve all contracts and selected operations; never add unrequested dataset stages.

    Training branch contracts describe A08's existing study/transfer dispatch. A09-06 does
    not replace that dispatcher or invent separate outcomes for its concurrent study groups.
    """
    if completion_version == 'current':
        # Only an owned managed launch can select the pre-receipt compatibility contract.
        from .run_reporting import context
        completion_version = (None if context() and os.environ.get('EFORECAST_STAGE_COMPLETION_VERSION') == '0' else 1)
    if script not in {'datasets', 'training', 'combine'}:
        raise ValueError(f'unknown pipeline script {script!r}')
    static = deepcopy(static_data)
    if not static.get('path_model'):
        from eforecast.init.initialize import create_paths
        create_paths(static, dry_run=True)
    model, data = Path(static['path_model']), Path(static['path_data'])
    options = deepcopy(options)
    nwp, images = bool(static.get('NWP')), bool(static.get('use_image'))
    feature = bool([value for value in static.get('feature_selection_methods', []) if value is not None])
    requested = options.get('stage') or (['datasets'] if script == 'datasets' else ['all'])
    stages = set()
    if script == 'datasets':
        for name in requested:
            stages.update(DATASET_STAGE_GROUPS.get(name, (name,)))
        if stages - set(ALL_DATASET_STAGES):
            raise ValueError('unknown dataset stage')
    elif script == 'combine':
        stages = set(COMBINE_STAGES if 'all' in requested else requested)
        if stages - set(COMBINE_STAGES):
            raise ValueError('unknown combination stage')
    selected = set(stages)
    implicit = set()
    if script == 'training':
        selected = {'training'}
    if script == 'combine' and stages:
        # These were internal calls already. Do not silently add fitting or selection.
        applicable = stages - ({'calibrate'} if not static.get('interval_coverage', [.8, .9]) else set())
        if applicable:
            implicit.add('predict')
        if applicable & {'calibrate', 'evaluate'}:
            implicit.add('combine')
        implicit -= selected
        selected |= implicit
    if selected_keys is not None:
        selected = set(selected_keys)
    legacy_partition = (static.get('is_Fuzzy') and static.get('split_policy') == 'legacy_resplit' and
                        (script in {'training', 'combine'} or bool(set(MODEL_STAGES) & selected)))
    if legacy_partition and completion_version == 1:
        # This supported policy intentionally replaces an already fitted partition. Preserve
        # reproduction through explicit unverified observations; never silently migrate it or
        # assert the frozen producer/input contract it cannot satisfy.
        completion_version = None

    products = {
        'source': {'producer': None, 'paths': [static.get('filename')]},
        'extracted_nwp': {'producer': 'extract', 'paths': []},
        'nwp': {'producer': 'nwp', 'paths': [str(data / 'dataset_nwps.pickle')]},
        'images': {'producer': 'images', 'paths': [str(data / 'dataset_images.pickle')]},
        'image_source': {'producer': None, 'paths': [str(value) for value in
            (static.get('sat_folder'), Path(static['path_image']) / 'processed' if static.get('path_image') else None) if value]},
        'target': {'producer': 'target', 'paths': [str(data / 'dataset_target_data.csv')]},
        'row': {'producer': 'row', 'paths': [str(data / 'dataset_row_data.pickle'), str(data / 'dataset_row_data.csv')]},
        'lstm': {'producer': 'lstm', 'paths': [str(data / 'dataset_lstm_data.pickle')]},
        'partition': {'producer': 'split', 'paths': [str(data / 'cv_mask.pickle')]},
        'scalers': {'producer': 'preprocess', 'paths': [str(model / 'scalers.pickle')]},
        'global_masks': {'producer': 'clustering', 'paths': [str(model / 'global/cv_mask.pickle')]},
        'memberships': {'producer': 'clustering', 'paths': [str(model / 'clusters.pickle')]},
        'features': {'producer': 'features', 'paths': [str(model / 'feature_selectors.pickle')]},
        'models': {'producer': 'training', 'paths': []},
        'predictions': {'producer': 'predict', 'paths': []},
        'selection': {'producer': 'select', 'paths': []},
        'combined': {'producer': 'combine', 'paths': []},
        'calibration': {'producer': 'calibrate', 'paths': [str(model / 'Results/calibration.json')]},
        'evaluation': {'producer': 'evaluate', 'paths': [str(model / 'Results/evaluation.csv')]},
    }
    if completion_version == 1:
        products['image_pixels'] = {'producer': 'images', 'paths': []}
    if completion_version == 1:
        products['prepared_trials'] = {'producer': 'training', 'paths': []}
    pixels = completion_version == 1
    inventory = _models(static) if script == 'combine' and inspect else None
    glob, clusters = _methods(static, options)
    if script == 'training':
        active_methods = glob + (clusters if static.get('is_Fuzzy') else [])
        inputs = _inputs(static, active_methods, cpu=bool(options.get('cpu')), pixels=pixels)
    elif script == 'combine' and inventory and inventory.get('methods'):
        trained = inventory['methods']
        deep = [method for method in trained if method not in CPU_METHODS]
        shallow = [method for method in trained if method in CPU_METHODS]
        inputs = list(dict.fromkeys((_inputs(static, deep, pixels=pixels) if deep else []) +
                                   (_inputs(static, shallow, cpu=True) if shallow else [])))
    else:
        inputs = _inputs(static, pixels=pixels)
    lstm_inputs = ['source'] + (['nwp'] if any(var.get('type') == 'nwp' for var in static.get('variables', [])) else [])
    extra_csv = []
    for variable in static.get('variables', []):
        source = variable.get('source')
        if isinstance(source, str) and source.endswith('.csv'):
            product = 'csv:' + source
            products[product] = {'producer': None, 'paths': [source]}
            if product not in extra_csv:
                extra_csv.append(product)
    selector_required = any(_configured(static.get(name, {}).get('feature_selection_method'))
                            for name in glob + (clusters if static.get('is_Fuzzy') else []))
    # Which family asks for which selection: the refusal below names them when none is computed.
    asking = {name: sorted(set(_choices(static.get(name, {}).get('feature_selection_method'))))
              for name in glob + (clusters if static.get('is_Fuzzy') else [])}
    asking = {name: values for name, values in asking.items() if values}
    if options.get('cpu'):
        methods = static.get('feature_selection_methods', [])
        selector_required = (bool([item for item in methods if item is not None])
            if any(method not in FOUNDATION_METHODS for method in glob + clusters) else
            bool(methods and methods[0] is not None))
    preprocessing_inputs = ['partition', 'target', 'row']
    sequences = has_sequences(static)
    for name, enabled in (('nwp', nwp), ('images', images), ('lstm', sequences)):
        if enabled and (name in selected or inspect and _file(products[name]['paths'])['status'] == 'legacy'):
            preprocessing_inputs.append(name)
    fuzzy_inputs = _inputs({**static, 'experiments': {'clustering': {'row_all': []}}})
    specs = [
        ('extract', 'group', ['source'], ['extracted_nwp'], nwp, 'no NWP provider configured'),
        ('nwp', 'global', ['source', 'extracted_nwp'], ['nwp'], nwp, 'no NWP provider configured'),
        ('images', 'global', ['source', 'image_source'], ['images'] +
            (['image_pixels'] if pixels and 'image_pixels' in _inputs(static, pixels=True) else []), images, 'images are disabled'),
        ('target', 'global', ['source'], ['target'], True, ''),
        ('row', 'global', ['source', *extra_csv], ['row'], True, ''),
        ('lstm', 'global', lstm_inputs + extra_csv, ['lstm'], sequences, NO_SEQUENCES),
        ('split', 'global', ['target', 'row'], ['partition'], True, ''),
        ('preprocess', 'global', preprocessing_inputs, ['scalers'], True, ''),
        ('clustering', 'clusters' if static.get('is_Fuzzy') else 'global',
            list(dict.fromkeys(['partition', 'scalers', *fuzzy_inputs])) if static.get('is_Fuzzy') else ['partition'],
            ['global_masks'] + (['memberships'] if static.get('is_Fuzzy') else []), True, ''),
        # Selectors are fitted for every input the case's experiments name, used or not. A
        # sequence branch in a definition nobody runs must not ask a long_term case for a
        # dataset it cannot have; a method that does run one is refused at training, by name.
        ('features', 'global+clusters' if static.get('is_Fuzzy') else 'global',
            list(dict.fromkeys([name for name in _inputs(static) if name != 'lstm' or sequences]
                               + ['partition', 'scalers'] + (['memberships'] if static.get('is_Fuzzy') else []))),
            ['features'], feature, 'no feature-selection method configured'),
        ('training', ('global+clusters' if glob and clusters and static.get('is_Fuzzy') else 'global' if glob else 'clusters') if options.get('cpu') else 'global+clusters' if static.get('is_Fuzzy') else 'global',
            list(dict.fromkeys(inputs + ['partition', 'scalers'] + (['global_masks'] if glob else []) +
                (['memberships'] if static.get('is_Fuzzy') and clusters else []) +
                (['features'] if selector_required else []))),
            ['prepared_trials'] if options.get('prepare_only') and completion_version == 1 else ['models'],
            bool(glob or clusters if static.get('is_Fuzzy') else glob), 'no enabled training method for the requested scope'),
        ('predict', 'global+clusters' if static.get('is_Fuzzy') else 'global',
            list(dict.fromkeys(inputs + ['scalers', 'models'] + (['memberships'] if static.get('is_Fuzzy') else []))),
            ['predictions'], True, ''),
        ('select', 'global+clusters', ['target', 'predictions'], ['selection'], True, ''),
        ('combine', 'models', ['target', 'predictions'] + (['selection'] if 'select' in selected else []), ['combined'], True, ''),
        ('calibrate', 'models', ['target', 'combined', 'predictions'], ['calibration'],
            bool(static.get('interval_coverage', [.8, .9])), 'prediction intervals are disabled'),
        ('evaluate', 'models', ['target', 'combined', 'predictions'], ['evaluation'], True, ''),
    ]
    nodes = []
    for key, scope, requires, produces, enabled, skip in specs:
        execution_inputs = list(requires)
        reuse = None
        if inspect and not options.get('recreate') and key in {'nwp', 'images', 'target', 'row', 'lstm', 'split'}:
            existing = _file(products[produces[0]]['paths'])
            if existing['status'] == 'legacy':
                execution_inputs = ['source']
                reuse = existing
        nodes.append({'key': key, 'stage': key, 'scope': scope, 'selected': key in selected,
            'implicit': key in implicit, 'enabled': enabled, 'skip_reason': None if enabled else skip,
            'inputs': requires, 'outputs': produces,
            'execution_inputs': execution_inputs, 'legacy_reuse_candidate': reuse,
            'dependencies': list(dict.fromkeys(products[name]['producer'] for name in requires
                if products[name]['producer'] is not None)),
            'options': deepcopy(options) if key in selected else {},
            'mode': ('apply' if key == 'combine' and key not in stages else
                     'prepare' if key == 'training' and options.get('prepare_only') else 'run')})
    order = ordered(nodes)
    by_key = {node['key']: node for node in nodes}
    needed = {name for node in nodes if node['selected'] and node['enabled'] for name in node['execution_inputs']}
    if script == 'datasets' and any(node['selected'] and node['enabled'] for node in nodes):
        needed.add('source')  # DatasetCreator reads this even for a subset using cached outputs.
    for name in needed:
        product = products[name]
        producer = by_key.get(product['producer'])
        if producer and producer['selected'] and producer['enabled']:
            product['evidence'] = {'status': 'planned', 'reason': f"produced by selected stage {producer['key']}"}
        elif not inspect:
            product['evidence'] = {'status': 'unknown', 'reason': 'prerequisite inspection was not requested'}
        elif name == 'models':
            product['evidence'] = inventory or _models(static)
            other = _other_configuration(static) if script == 'combine' else None
            if other and product['evidence']['status'] not in {'missing', 'unknown'}:
                product['evidence'] = {'status': 'missing', 'families': other, 'reason': other_configuration_reason(other)}
        elif name == 'image_source':
            product['evidence'] = _directory(product['paths'])
        elif name == 'image_pixels':
            from eforecast.common_utils.image_assets import paths as image_paths
            try:
                assets = image_paths(static)
                product['evidence'] = (_file([str(path)]) for path in assets)
                product['evidence'] = next((value for value in product['evidence'] if value['status'] != 'legacy'),
                    {'status': 'legacy', 'reason': 'processed image pixels exist; producer provenance is unverified'}) if assets else {
                        'status': 'missing', 'reason': 'no referenced image pixels'}
            except (OSError, ValueError, KeyError, TypeError):
                product['evidence'] = {'status': 'missing', 'reason': 'processed image pixel references are unavailable'}
        elif name == 'features':
            from .stage_artifacts import _scopes
            scopes = [scope for scope in _scopes(static) if scope != model / 'global' or any((static.get('global_methods') or {}).values())]
            views = [_file([str(scope / 'feature_selectors.pickle')]) for scope in scopes]
            product['evidence'] = (next((view for view in views if view['status'] != 'legacy'), None) or
                {'status': 'legacy', 'reason': 'scope selectors exist; historical producer provenance is unverified'}) if views else _file(product['paths'])
        elif name == 'extracted_nwp':
            providers = static.get('NWP') or []
            found = []
            for provider in providers:
                found.extend(sorted(Path(static['path_group_nwp']).glob(provider['model'] + '_*.pickle'))[:1])
            product['evidence'] = (_file([str(path) for path in found]) if len(found) == len(providers) else
                {'status': 'missing', 'reason': 'no extracted daily NWP payload for every provider; select extract or supply pre-extracted data'})
        elif name in {'predictions', 'combined', 'selection'}:
            product['evidence'] = {'status': 'unknown', 'reason': 'derived in the selected command'}
        else:
            product['evidence'] = _file(product['paths'])
    # A modern rejection never falls back to legacy file presence. Preview checks metadata;
    # the worker checks content before consuming it. No case/history is backfilled here.
    unverified = []
    if inspect and completion_version == 1:
        from .stage_artifacts import inspect_stage
        for name in needed:
            producer = products[name]['producer']
            evidence = products[name]['evidence']
            if producer and name != 'models' and evidence['status'] not in {'planned', 'missing', 'unknown'}:
                readiness = inspect_stage(static, producer, verify=False, product=name, check_sources=False)
                if readiness['status'] == 'rejected':
                    products[name]['evidence'] = {**readiness, 'status': 'missing'}
                elif readiness.get('id'):
                    products[name]['evidence'] = {**evidence, 'readiness': readiness}
                if readiness['status'] == 'recorded' and readiness.get('metadata') == 'changed':
                    # The cheap check cannot answer this one: the payload's metadata moved since
                    # its receipt, so only hashing its content can say whether it is still the
                    # artifact that was published. That is not proof of corruption and must not
                    # block -- but nothing downstream may read this plan as "ready" either.
                    products[name]['evidence'] = {**products[name]['evidence'], 'verification': 'required'}
                    unverified.append({'product': name, 'producer': producer,
                                       'reason': f'{name} needs full content verification: {readiness["reason"]}'})
    blocked = []
    for key in order:
        node = by_key[key]
        node['blocked'] = []
        if not node['selected'] or not node['enabled']:
            node['status'] = 'not_selected' if node['enabled'] else 'skipped'
            continue
        for name in node['execution_inputs']:
            evidence = products[name]['evidence']
            producer = by_key.get(products[name]['producer'])
            if evidence['status'] in {'missing', 'unknown'}:
                # A stage this case cannot run is not a remedy: say why the product cannot exist.
                remedy = ('' if not producer else f"; produce with stage {producer['key']}" if producer['enabled']
                          else f"; stage {producer['key']} does not apply to this case ({producer['skip_reason']})")
                if evidence.get('families'):
                    remedy = ''         # the reason says what to do: which family to train again
                if name == 'features' and producer and not producer['enabled'] and asking and not options.get('cpu'):
                    remedy = ('; ' + ', '.join(f"{method} asks for {' / '.join(values)}" for method, values in asking.items())
                              + ', and the case computes no selector (methods.feature_selection_methods names none). '
                                'Name the method there and run the features stage, or set feature_selection_method '
                                'to none for ' + ', '.join(asking))
                node['blocked'].append({'node': key, 'product': name, 'producer': products[name]['producer'],
                    'reason': f"{key} requires {name}: {evidence['reason']}" + remedy, 'evidence': evidence})
            elif producer and producer['selected'] and producer['blocked']:
                node['blocked'].append({'node': key, 'product': name, 'producer': producer['key'],
                    'reason': f"{key} waits for blocked stage {producer['key']}"})
        node['status'] = 'blocked' if node['blocked'] else 'planned'
        blocked.extend(node['blocked'])
    if script == 'datasets' and 'source' in needed and products['source']['evidence']['status'] in {'missing', 'unknown'}:
        first = next(node for node in nodes if node['selected'] and node['enabled'])
        if not any(item['product'] == 'source' for item in blocked):
            reason = {'node': first['key'], 'product': 'source', 'producer': None,
                'reason': 'dataset entry point requires its main CSV even when selected outputs are cached'}
            first['blocked'].append(reason)
            first['status'] = 'blocked'
            blocked.append(reason)
    if by_key['extract']['selected'] and nwp:
        unsupported = [provider['model'] for provider in static['NWP'] if provider['model'] not in NWP_EXTRACTORS]
        if unsupported:
            reason = {'node': 'extract', 'product': 'extracted_nwp', 'producer': None,
                'reason': f'no extractor for {unsupported}; select nwp for pre-extracted providers'}
            by_key['extract']['blocked'].append(reason)
            by_key['extract']['status'] = 'blocked'
            blocked.append(reason)
    if script == 'training' and options.get('cpu') and not static.get('is_Fuzzy') and (options.get('cpu_scope') or static.get('cpu_scope', 'clusters')) == 'clusters':
        reason = {'node': 'training', 'product': 'memberships', 'producer': 'clustering',
                  'reason': 'CPU training requires a fuzzy case and cluster memberships'}
        by_key['training']['blocked'].append(reason)
        by_key['training']['status'] = 'blocked'
        blocked.append(reason)
    if inspect and completion_version == 1:
        from .stage_artifacts import inspect_stage
        cached = {'extract', 'nwp', 'images', 'target', 'row', 'lstm', 'split', 'clustering', 'features'}
        for node in nodes:
            if not node['selected'] or not node['enabled'] or node['key'] not in cached:
                continue
            rebuild = (options.get('recreate_extraction') if node['key'] == 'extract' else
                       options.get('recreate') or options.get('refit'))
            if rebuild:
                continue
            evidence = inspect_stage(static, node['key'], verify=False)
            if evidence['status'] == 'rejected':
                reason = {'node': node['key'], 'product': node['outputs'][0], 'producer': node['key'],
                    'reason': f'{node["key"]} cached completion is invalid: {evidence["reason"]}; select an explicit rebuild',
                    # Nothing external is missing: this stage can produce the artifact again, and
                    # an accepted resume plan that says so is the explicit rebuild it asks for.
                    'rebuildable': True, 'evidence': evidence}
                node['blocked'].append(reason)
                node['status'] = 'blocked'
                blocked.append(reason)
    branches = training_branches(static, options, pixels=pixels)
    by_key['combine']['optional_inputs'] = {
        'selected_trials': 'use existing selection when present; otherwise use all admissible trials',
        'fitted_combiners': 'apply existing model when present; preserve mean fallback when absent or incompatible',
    }
    if script == 'training' and static.get('transfer_learning'):
        by_key['training']['transfer_base'] = {
            'configuration': _configuration_reference(static.get('transfer_learning_from', {})),
            'status': 'pinned' if static.get('static_data_base') else 'legacy',
            'reason': 'source configuration is materialized by A07; source models retain the completion reader contract',
        }
    if script == 'training':
        for branch in branches:
            external = branch.get('external_global_source')
            if external:
                evidence = (_models(static, method=external['method'], global_only=True) if inspect else
                            {'status': 'unknown', 'reason': 'prerequisite inspection was not requested'})
                branch['external_global_evidence'] = evidence
                if evidence['status'] in {'unknown', 'missing'}:
                    branch['disposition'] = 'unknown' if evidence['status'] == 'unknown' else 'nothing_to_do'
                    branch['reason'] = ('source is unobservable; the transfer reader must inspect it' if evidence['status'] == 'unknown'
                        else 'no admissible external global source; preserve FixedUnit empty-transfer behavior')
    policies = deepcopy(POLICIES)
    if completion_version is None:
        policies['readiness'] = 'existing payloads are legacy prerequisites; verified stage publication is A09-07'
    plan = {'version': VERSION, 'script': script, 'options': options, 'nodes': nodes, 'products': products,
        'order': order, 'execution_order': [key for key in order if by_key[key]['selected']],
        'blocked': blocked, 'unverified': unverified, 'dry_run': bool(options.get('dry_run')), 'policies': policies,
        'training_branches': branches, 'path_model': str(model)}
    if completion_version == 1:
        plan['completion_version'] = 1
    if legacy_partition:
        plan['compatibility'] = {'status': 'legacy', 'reason': 'legacy_resplit replaces the canonical partition after fitting; '
            'operation observations remain unverified. The stored policy is preserved; verified frozen publication '
            'requires a separately accepted policy/generation change.'}
    plan['identity'] = plan_identity(plan)
    return plan


def plan_identity(plan):
    contract = {k: v for k, v in plan.items()
                if k not in {'identity', 'compatibility', 'unverified', 'resume', 'resume_plan'}}
    contract['blocked'] = []
    contract['nodes'] = [{k: v for k, v in node.items() if k not in
                         {'status', 'blocked', 'execution_inputs', 'legacy_reuse_candidate'}} for node in plan['nodes']]
    contract['products'] = {name: {k: v for k, v in product.items() if k != 'evidence'}
                            for name, product in plan['products'].items()}
    contract['training_branches'] = [{k: v for k, v in branch.items() if k not in
                                    {'external_global_evidence', 'disposition', 'reason'}}
                                    for branch in plan['training_branches']]
    return hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest()


def training_branches(static, options, *, pixels=False):
    """Explicit method/scope dependencies; A08 retains dispatch and actual study outcomes."""
    glob, cluster = _methods(static, options)
    nodes = []
    for method in glob:
        nodes.append({'key': f'global:{method}', 'stage': 'train_global', 'scope': 'global',
            'method': method, 'dependencies': [], 'inputs': _inputs(static, [method], cpu=bool(options.get('cpu')), pixels=pixels) + ['partition', 'scalers', 'global_masks'],
            'backend': 'cpu' if options.get('cpu') else 'gpu', 'options': deepcopy(options)})
    if static.get('is_Fuzzy'):
        for method in cluster:
            key = f'clusters:{method}'
            nodes.append({'key': key, 'stage': 'train_clusters', 'scope': 'cluster-set', 'method': method,
                'dependencies': [], 'inputs': _inputs(static, [method], cpu=bool(options.get('cpu')), pixels=pixels) + ['partition', 'scalers', 'memberships'],
                'backend': 'cpu' if options.get('cpu') else 'gpu', 'options': deepcopy(options)})
            if not options.get('cpu') and static.get(method, {}).get('transfer_learning_from_global'):
                parent = f'global:{method}'
                nodes.append({'key': f'transfer:{method}', 'stage': 'transfer', 'scope': 'cluster-set',
                    'method': method, 'dependencies': [parent] if method in glob else [],
                    'dependency_policy': 'global study settled; inspect admitted sources, otherwise nothing to do',
                    'external_global_source': None if method in glob else {'method': method, 'scope': 'global'},
                    'inputs': ['memberships', 'scalers', 'global_weights'], 'options': deepcopy(options)})
    ordered(nodes)
    if static.get('transfer_learning'):
        for node in nodes:
            node['transfer_base'] = _configuration_reference(static.get('transfer_learning_from', {}))
    return nodes


def require_ready(plan):
    approved = {key for key, decision in (plan.get('resume') or {}).items() if decision == 'rebuild'}
    remaining = [item for item in plan['blocked']
                 if not (item.get('rebuildable') and item.get('node') in approved)]
    if remaining and not plan.get('dry_run'):
        raise BlockedStage('; '.join(item['reason'] for item in remaining[:8]))


def require_inputs(static_data, plan, key, completed):
    """Recheck external payloads immediately before a selected operation can have effects.

    The recheck is fresh, the authorization is not: a stage this run was accepted to rebuild is
    still accepted when its own cached output is the thing that is invalid. A missing external
    prerequisite remains a refusal.
    """
    current = resolve(static_data, plan['script'], plan['options'], selected_keys={key},
                      completion_version=plan.get('completion_version'))
    current['resume'] = dict(plan.get('resume') or {})
    current['blocked'] = [item for item in current['blocked'] if not
        (item.get('product') in {'predictions', 'combined', 'selection'} and item.get('producer') in completed)]
    require_ready(current)


def describe(plan):
    lines = [f"stage plan v{plan['version']} {plan['identity'][:12]} ({plan['script']})"]
    if plan.get('compatibility'):
        lines.append('  legacy completion: ' + plan['compatibility']['reason'])
    for key in plan['execution_order']:
        node = next(node for node in plan['nodes'] if node['key'] == key)
        lines.append(f"  {key} [{node['scope']}] {node['status']}" +
                     (f": {node['skip_reason']}" if node['skip_reason'] else ''))
        if node['enabled']:
            lines.append('    requires: ' + ', '.join(node['inputs']))
        lines.extend('    ' + item['reason'] for item in node['blocked'])
    if plan['script'] == 'training':
        for branch in plan['training_branches']:
            lines.append(f"    {branch['key']} [{branch['scope']}] after " +
                         (', '.join(branch['dependencies']) or 'external input prerequisites'))
    return '\n'.join(lines)
