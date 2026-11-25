# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Semantic checks that do not import models, datasets or hardware libraries."""
import math

from eforecast.config.capabilities import CLUSTERING_METHODS, CPU_METHODS


def parameter_values(value):
    """Values/endpoints to validate in fixed, legacy or tagged family parameters."""
    if isinstance(value, dict):
        if set(value) in ({'choices'}, {'range'}, {'values'}):
            value = next(iter(value.values()))
        elif set(value) == {'tag', 'value'}:
            value = value['value']
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _numeric_parameter(value, path, minimum=None, maximum=None, integer=False, exclusive_min=False,
                       exclusive_max=False):
    values = parameter_values(value)
    if not values:
        raise ValueError(f'{path} needs at least one value')
    if isinstance(value, dict) and ('range' in value or value.get('tag') == 'range'):
        if len(values) != 2:
            raise ValueError(f'{path}: a range needs exactly [low, high]')
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values) and values[0] > values[1]:
            raise ValueError(f'{path}: the low range bound must not exceed the high bound')
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f'{path} must contain finite numbers')
        if integer and int(item) != item:
            raise ValueError(f'{path} must contain integers')
        if minimum is not None and (item < minimum or (exclusive_min and item == minimum)):
            raise ValueError(f'{path} must be {">" if exclusive_min else ">="} {minimum}')
        if maximum is not None and (item > maximum or (exclusive_max and item == maximum)):
            raise ValueError(f'{path} must be {"<" if exclusive_max else "<="} {maximum}')


def validate_family_parameters(families):
    for family, params in families.items():
        if family == 'clustering' and params.get('params') is not None and not isinstance(params['params'], dict):
            raise ValueError('clustering.params must be a mapping of fixed network parameters')
        # Old clustering documents keep network parameters nested; that block wins at runtime.
        effective = {**params, **(params.get('params') or {})} if family == 'clustering' else params
        for key in ('n_trials', 'n_jobs', 'batch_size', 'max_iterations', 'n_rules', 'min_samples'):
            if key in effective:
                _numeric_parameter(effective[key], f'{family}.{key}', minimum=1, integer=True)
        if 'warming_iterations' in effective:
            _numeric_parameter(effective['warming_iterations'], f'{family}.warming_iterations', minimum=0, integer=True)
        if 'thres_act' in effective:
            _numeric_parameter(effective['thres_act'], f'{family}.thres_act', minimum=0, maximum=1,
                               exclusive_min=True, exclusive_max=True)
        if 'max_samples_ratio' in effective:
            _numeric_parameter(effective['max_samples_ratio'], f'{family}.max_samples_ratio',
                               minimum=0, maximum=1, exclusive_min=True)
        if family != 'clustering':
            continue
        for key in ('n_rules', 'n_jobs', 'batch_size', 'max_iterations', 'min_samples', 'thres_act', 'max_samples_ratio'):
            if key in effective and not isinstance(effective[key], (int, float)):
                raise ValueError(f'clustering.{key} must be fixed: the clusterer does not run a parameter search')
        methods = params.get('methods', ['RBF'])
        if not isinstance(methods, list) or not methods or any(m not in CLUSTERING_METHODS for m in methods):
            raise ValueError('clustering.methods must be a non-empty list of implemented methods: RBF')
        selected = params.get('clusters_for_method', 'RBF')
        if selected != 'both' and selected not in methods:
            raise ValueError('clustering.clusters_for_method must name a selected clustering method or both')
        if 'rbf_var_imp' in params and (not isinstance(params['rbf_var_imp'], list) or not params['rbf_var_imp']
                                       or any(not isinstance(rule, list) or not rule for rule in params['rbf_var_imp'])):
            raise ValueError('clustering.rbf_var_imp needs at least one non-empty rule set')


def validate_experiment_references(cfg, experiments):
    """Check enabled network families after module and inline templates have been merged."""
    families = {name for flags in (cfg.methods.global_methods, cfg.methods.cluster_methods)
                for name, enabled in flags.items() if enabled and name != 'GRAPH' and name not in CPU_METHODS}
    if cfg.project.is_fuzzy:
        families.add('clustering')
    for family in sorted(families):
        params = cfg.methods.method_params.get(family, {})
        if family == 'clustering':
            params = {**params, **(params.get('params') or {})}
        tags = parameter_values(params.get('experiment_tag'))
        if not tags or any(not isinstance(tag, str) or not tag for tag in tags):
            raise ValueError(f'{family}.experiment_tag must name at least one experiment template')
        missing = [tag for tag in tags if tag not in experiments]
        if missing:
            raise ValueError(f'{family}.experiment_tag refers to missing templates: {", ".join(missing)}; '
                             'load the templates in Experiments or correct the method family')
