# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Load typed YAML configuration and expose the legacy static-data shape."""

import copy
import importlib
import importlib.util
from pathlib import Path
from typing import Any

import yaml

from .schema import lag_step, ExperimentConfig, HorizonType, NetworkExperimentConfig, VariableConfig
from .validation import validate_experiment_references


SET_VALUED_KEYS: dict[str, set[str]] = {
    "CNN": {"experiment_tag", "attention", "merge", "scale_row_method", "scale_nwp_method"},
    "TRANSFORMER": {"experiment_tag", "merge", "scale_row_method", "scale_nwp_method", "scheduler"},
    "TIMM": {"experiment_tag", "pretrained", "merge", "scale_row_method", "scale_nwp_method"},
    "LSTM": {
        "experiment_tag", "attention", "merge", "scale_row_method", "scale_nwp_method",
        "feature_selection_method", "scheduler",
    },
    "MLP": {
        "experiment_tag", "merge", "compress", "scale_row_method", "scale_nwp_method",
        "feature_selection_method",
    },
    "CrossViVit": {
        "experiment_tag", "sat_image_type", "pretrained", "final_image_size",
        "scheduler", "scale_img_method", "image_data_type", "learning_rate",
    },
    "Distributed": {
        "experiment_tag", "batch_size", "merge", "compress", "scale_row_method",
        "scale_nwp_method", "feature_selection_method", "act_func", "optimizer",
        "scheduler", "learning_rate", "n_rules", "is_fuzzy", "clustering_method",
    },
    "RF": {"max_depth", "max_features"},
    "CatBoost": {"learning_rate", "objective", "bootstrap_type"},
    "lightgbm": {"learning_rate", "max_depth"},
    "lasso": {"eps", "fit_intercept", "selection"},
}


def load_config(yaml_path: str | Path) -> ExperimentConfig:
    with Path(yaml_path).open(encoding="utf-8") as stream:
        config = ExperimentConfig.model_validate(yaml.safe_load(stream))
    object.__setattr__(config, "_yaml_path", str(yaml_path))
    return config


def _load_experiments(spec: str | None, yaml_path: str | Path) -> Any:
    if not spec:
        return None
    path = Path(spec)
    is_file_spec = path.suffix == ".py" or path.exists() or "/" in spec
    if is_file_spec and not path.is_absolute():
        candidates = [Path.cwd() / path]
        if yaml_path:
            # the YAML may sit in cases/<case>.yaml or in cases/<case>/<part>.yaml,
            # so look up to four folders above it (project root in both layouts)
            yaml_parent = Path(yaml_path).resolve().parent
            candidates.extend(parent / path for parent in [yaml_parent, *yaml_parent.parents[:3]])
        path = next((candidate for candidate in candidates if candidate.exists()), path)
    if is_file_spec:
        module_name = f"_eforecast_experiments_{abs(hash(str(path)))}"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Cannot load experiments module from {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(spec)
    return getattr(module, "experiments")


def _inline_experiments(experiments: dict[str, NetworkExperimentConfig] | None) -> dict[str, Any] | None:
    if experiments is None:
        return None
    return {
        experiment_name: {
            branch_name: [
                (layer.type, layer.legacy_size())
                for layer in layers
                if layer.enabled
            ]
            for branch_name, layers in experiment.root.items()
        }
        for experiment_name, experiment in experiments.items()
    }


def _legacy_param_value(value: Any) -> Any:
    """``{choices: [...]}`` -> set, ``{range: [...]}`` -> list, anything else unchanged."""
    if isinstance(value, dict) and len(value) == 1:
        if "choices" in value and isinstance(value["choices"], list):
            try:
                return set(value["choices"])
            except TypeError:
                return list(value["choices"])
        if "range" in value and isinstance(value["range"], list):
            return list(value["range"])
    return value


def _inline_experiment_layer_params(
    experiments: dict[str, NetworkExperimentConfig] | None,
    resolve: bool = False,
) -> dict[str, Any]:
    """Per-layer parameters of every experiment: ``{tag: {branch: [{type, name, params}, ...]}}``.

    ``size`` is left out: it travels positionally with the ``(type, size)`` tuples of
    :func:`_inline_experiments`, and the builder passes it as its own argument.

    ``resolve=False`` keeps each parameter tagged (``{"tag": ..., "value": ...}``) so the search
    space can turn ``choice`` and ``range`` into Optuna dimensions.  ``resolve=True`` collapses
    every parameter to a single value -- the first choice, the low bound of a range -- which is
    what a probe build needs when no trial has chosen anything.
    """
    if experiments is None:
        return {}

    def params_of(layer: Any) -> dict[str, Any]:
        return {key: (param.resolved() if resolve else {"tag": param.tag, "value": param.value})
                for key, param in layer.params.items() if key != "size"}

    return {
        experiment_name: {
            branch_name: [
                {"type": layer.type, "name": layer.name, "params": params_of(layer)}
                for layer in layers
                if layer.enabled
            ]
            for branch_name, layers in experiment.root.items()
        }
        for experiment_name, experiment in experiments.items()
    }


def _experiments_static_data(cfg: ExperimentConfig) -> Any:
    external = _load_experiments(cfg.experiments_module, getattr(cfg, "_yaml_path", ""))
    inline = _inline_experiments(cfg.experiments)
    if external is None:
        return inline
    if inline is None:
        return external
    result = dict(external)
    result.update(inline)
    return result


def _variable_lags(
    name: str,
    input_type: str,
    lags: None | int | list[int | float],
    horizon: int,
    horizon_type: HorizonType,
    ts_resolution: str,
    use_data_before_and_after_target: bool,
) -> list[int | float]:
    resolution = lag_step(ts_resolution)
    if lags is None or lags == 0:
        result: list[int | float] = (
            [0] if horizon_type != HorizonType.SHORT_TERM
            else [int(i) for i in range(int(horizon / resolution))]
        )
    elif isinstance(lags, int):
        result = [-int(i) for i in range(int(lags / resolution))]
    elif isinstance(lags, list):
        result = list(lags)
    else:
        raise ValueError("lags should be None or int or list")
    if name in {"Flux", "wind"} and use_data_before_and_after_target and horizon == 0:
        result = [min(result) - 1] + result + [max(result) + 1]
    return result


def _variable_dict(
    variable: VariableConfig,
    cfg: ExperimentConfig,
    nwp: list[dict[str, Any]],
    horizon: int,
) -> dict[str, Any]:
    providers = None
    if variable.nwp_provider is not None:
        if variable.nwp_provider == "ALL":
            providers = [item["model"] for item in nwp]
        else:
            providers = [item["model"] for item in nwp if item["model"] == variable.nwp_provider]
    return {
        "name": variable.name,
        "type": variable.type,
        "source": variable.source,
        "lags": _variable_lags(
            variable.name,
            variable.type,
            variable.lags,
            horizon,
            cfg.project.horizon_type,
            cfg.project.ts_resolution,
            cfg.data.use_data_before_and_after_target,
        ),
        "timezone": variable.timezone,
        "transformer": variable.transformer,
        "transformer_params": variable.transformer_params,
        "bands": variable.bands,
        "nwp_provider": providers,
        "use_diff_between_lags": variable.use_diff_between_lags,
    }


def _resolved_variables(variable: VariableConfig, cfg: ExperimentConfig, nwp: list[dict[str, Any]],
                        horizon: int) -> list[dict[str, Any]]:
    """What a configured variable is to a run: itself, or its sine and its cosine.

    A variable tagged ``cyclical`` becomes two inputs, ``<name>_sin`` and ``<name>_cos``, each
    with the lags, the source and every other setting of the variable. They are ordinary
    variables from here on, so the row data, the sequences, the scalers and the names a
    clustering or a selection may use (``DWIN_sin``) need no rule of their own. ``cyclical``
    keeps the name the variable was configured under, for a reader that looks a column up by it.
    """
    from .schema import is_cyclical
    resolved = _variable_dict(variable, cfg, nwp, horizon)
    if variable.type != 'timeseries' or not is_cyclical(variable.transformer):
        return [resolved]
    period = (variable.transformer_params or {}).get('cyclical')
    period = 360.0 if period is None else float(period)
    out = []
    for component in ('sin', 'cos'):
        item = copy.deepcopy(resolved)
        item['name'] = f'{variable.name}_{component}'
        item['transformer_params'] = {**(variable.transformer_params or {}),
                                      'cyclical': {'period': period, 'component': component}}
        item['cyclical'] = {'of': variable.name, 'component': component, 'period': period}
        out.append(item)
    return out


def _as_legacy_method_value(method: str, key: str, value: Any) -> Any:
    if key in SET_VALUED_KEYS.get(method, set()) and isinstance(value, list):
        return set(value)
    return value


def _legacy_method_params(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Legacy form of a method family: explicit ``{choices: [...]}`` -> set and
    ``{range: [lo, hi]}`` -> list (also inside nested dicts such as
    ``clustering.params``); plain lists follow ``SET_VALUED_KEYS`` as before."""
    result: dict[str, Any] = {}
    for key, value in params.items():
        converted = _legacy_param_value(value)
        if converted is value and isinstance(value, dict):
            converted = {sub_key: _legacy_param_value(sub_value)
                         for sub_key, sub_value in value.items()}
        elif converted is value:
            converted = _as_legacy_method_value(method, key, value)
        result[key] = converted
    return result


# ``methods.method_params.clustering`` is stored flat so the editor's parameter table can reach every
# key, but the clusterer reads two nested blocks (``clustering_manager``, ``tf_rbf_clusterer``):
#   data_type  the six data options of the row_all dataset it clusters on
#   params     the network that fits the fuzzy rules
# Both are derived here.  These keys belong to the clusterer itself and never to the network.
CLUSTERER_KEYS = frozenset({
    "methods", "clusters_for_method", "prediction_for_method", "thres_act", "rbf_var_imp",
    "explode_clusters", "n_var_lin", "min_samples", "max_samples_ratio",
})
CLUSTERING_DATA_TYPE_KEYS = ("scale_row_method", "scale_nwp_method", "scale_img_method",
                             "merge", "compress", "image_data_type")


def _clustering_runtime(flat: dict[str, Any], base_data_types: dict[str, Any] | None) -> dict[str, Any]:
    """``clustering`` with the ``data_type`` and ``params`` blocks the clusterer expects.

    A value explicitly nested in the stored case wins, so a case written before the families were
    flattened is read exactly as before.  Otherwise ``data_type`` takes each option from the flat
    family, falling back to ``methods.base_data_types``, and ``params`` takes every flat key that is
    not the clusterer's own -- which is the network, plus the data options the network also reads.
    The flat keys are left in place as well: nothing reads them there, and leaving them keeps a
    hand-written case that does working.
    """
    result = dict(flat)
    data_type = dict(flat.get("data_type") or {})
    for key in CLUSTERING_DATA_TYPE_KEYS:
        if key in data_type:
            continue
        if key in flat:
            data_type[key] = flat[key]
        elif base_data_types and key in base_data_types:
            data_type[key] = base_data_types[key]
    params = dict(flat.get("params") or {})
    for key, value in flat.items():
        if key not in CLUSTERER_KEYS and key not in {"data_type", "params"} and key not in params:
            params[key] = value
    result["data_type"] = data_type
    result["params"] = params
    return result


def _methods_static_data(
    cfg: ExperimentConfig,
    experiments: Any,
    experiment_layer_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    methods = cfg.methods
    result: dict[str, Any] = {
        "env_name": methods.env_name,
        "path_env": methods.path_env,
        "backend": methods.backend,
        "cpu_scope": methods.cpu_scope,
        "methods_cpu": methods.methods_cpu,
        "cluster_methods": methods.cluster_methods,
        "global_methods": methods.global_methods,
        "transfer_learning": methods.transfer_learning,
        "transfer_learning_from": methods.transfer_learning_from,
        "transfer_learning_merge_data": methods.transfer_learning_merge_data,
        "base_data_types": methods.base_data_types,
        "feature_selection_methods": methods.feature_selection_methods,
        "experiments": experiments,
        "max_performance": methods.max_performance,
        "val_test_ratio": methods.val_test_ratio,
        "keep_best_models": methods.keep_best_models,
        "selection_fraction": methods.selection_fraction,
        "calibration_fraction": methods.calibration_fraction,
        "cluster_coverage_policy": methods.cluster_coverage_policy,
        "interval_coverage": list(methods.interval_coverage),
        "CVs": methods.CVs,
        "split_seed": methods.split_seed,
        "split_policy": methods.split_policy,
        "kernels": methods.kernels,
    }
    if experiment_layer_params:
        result["experiment_layer_params"] = experiment_layer_params
    # An absent new control must not change retained artifact identities.
    if methods.runtime_ids:
        result['runtime_ids'] = dict(methods.runtime_ids)
    for method, params in methods.method_params.items():
        result[method] = _legacy_method_params(method, params)
    if isinstance(result.get("clustering"), dict):
        result["clustering"] = _clustering_runtime(result["clustering"], methods.base_data_types)
    result["combining"] = {
        "methods": methods.combine_methods,
        "data_type": methods.base_data_types,
    }
    return result


def to_static_data(cfg: ExperimentConfig) -> dict[str, Any]:
    nwp = [
        {
            "model": item.model,
            "resolution": item.resolution,
            "path_nwp_source": item.path_nwp_source,
            "levels": item.levels,
            # Only for a case that names one: an absent new control must not change retained identities.
            **({"path_nwp_gdrive": item.path_nwp_gdrive} if item.path_nwp_gdrive else {}),
        }
        for item in cfg.project.nwp_models
    ]
    nwp_value = nwp or None
    if nwp_value is not None:
        for coordinate in _flatten_coordinates(cfg.project.area_group):
            for model in nwp_value:
                resolution = model["resolution"]
                if resolution is not None and (round(coordinate / resolution) - coordinate / resolution) > 1e-6:
                    raise ValueError(
                        "Latitude Longitude in area group should be multiple of NWP resolution "
                        f"{resolution}, but it is {_flatten_coordinates(cfg.project.area_group)}"
                    )

    resolution = lag_step(cfg.project.ts_resolution)
    horizon = (
        0 if cfg.project.horizon_type in {HorizonType.DAY_AHEAD, HorizonType.INTRADAY}
        else cfg.project.horizon
    )
    variables = [resolved for variable in cfg.data.variables
                 for resolved in _resolved_variables(variable, cfg, nwp_value or [], horizon)]
    names = [variable['name'] for variable in variables]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"two inputs are named {', '.join(repeated)}: a cyclical variable X becomes X_sin and "
                         f"X_cos, so no other variable can have those names")
    target = cfg.data.target_variable
    target_lags = (
        target.lags if target.lags is not None else (
            [int(i) for i in range(int(horizon / resolution))]
            if cfg.project.horizon_type == HorizonType.SHORT_TERM else [0]
        )
    )
    target_dict = {
        "name": target.name,
        "source": target.source,
        "lags": target_lags,
        "columns": target.columns,
        "transformer": target.transformer,
        "transformer_params": target.transformer_params,
    }
    global_past = cfg.data.global_past_lags
    if global_past is None:
        global_past = [-int(i) for i in range(1, int(3 / resolution))]
    global_future = cfg.data.global_future_lags
    if global_future is None:
        global_future = [int(i) for i in range(int(horizon / resolution))]

    project = cfg.project
    folders = cfg.folders
    static_data: dict[str, Any] = {
        "project_name": project.name,
        "project_owner": project.owner,
        "projects_group": project.group,
        "version_model": project.version_model,
        "version_group": project.version_group,
        "horizon_type": project.horizon_type.value,
        "horizon": horizon,
        "rated": project.rated_power,
        "coord": project.coordinates,
        "area_group": project.area_group,
        "use_image": project.use_images,
        "image_coord": project.image_coord,
        "image_size": project.image_size,
        "site_indices": project.site_indices,
        "type": project.type.value,
        "NWP": nwp_value,
        "filename": project.filename,
        "Evaluation_start": project.evaluation_start,
        "local_timezone": project.local_timezone,
        "site_timezone": project.site_timezone,
        "country": project.country,
        "ts_resolution": project.ts_resolution,
        "is_Global": project.is_global,
        "is_Fuzzy": project.is_fuzzy,
        "is_probabilistic": project.is_probabilistic,
        "Docker": project.docker,
        "regional": (
            isinstance(project.coordinates, dict)
            or (isinstance(project.coordinates, list) and len(project.coordinates) == 4)
        ),
        # The runtime namespace has an explicit contract. Retain the known legacy
        # keys for compatibility without exporting every future JobsConfig field.
        "n_cpus": cfg.jobs.n_cpus,
        **({'model_cache_mode': cfg.jobs.model_cache_mode} if cfg.jobs.model_cache_mode is not None else {}),
        "cpu_workers": cfg.jobs.cpu_workers,
        "cpu_study_threads": cfg.jobs.cpu_study_threads,
        "cpu_study_device": cfg.jobs.cpu_study_device,

        "n_jobs": cfg.jobs.n_jobs,
        "n_gpus": cfg.jobs.n_gpus,
        "intra_op": cfg.jobs.intra_op,
        "n_jobs_rbfnn": cfg.jobs.n_jobs_rbfnn,
        "n_jobs_lstm": cfg.jobs.n_jobs_lstm,
        "n_jobs_cnn_3d": cfg.jobs.n_jobs_cnn_3d,
        "n_jobs_cnn": cfg.jobs.n_jobs_cnn,
        "n_jobs_mlp": cfg.jobs.n_jobs_mlp,
        "pycharm_path": folders.pycharm_path,
        "sys_folder": folders.sys_folder,
        "nwp_folder": folders.nwp_folder,
    }
    if folders.sat_folder is not None:
        static_data["sat_folder"] = folders.sat_folder
    if folders.path_image is not None:
        static_data["path_image"] = folders.path_image
    static_data.update({
        "nwp_data_merge": cfg.data.nwp_data_merge,
        "compress_data": cfg.data.compress_data,
        "img_data": cfg.data.img_data,
        "use_data_before_and_after_target": cfg.data.use_data_before_and_after_target,
        "nwp_selection_policy": cfg.data.nwp_selection_policy,
        "nwp_availability_hours": cfg.data.nwp_availability_hours,
        "remove_night_hours": cfg.data.remove_night_hours,
        "variables": variables,
        "target_variable": target_dict,
        "time_merge_variables": cfg.data.time_merge_variables,
        "global_past_lags": global_past,
        "global_future_lags": global_future,
        "scale_row_method": cfg.data.scale_row_method,
        "scale_img_method": cfg.data.scale_img_method,
        "scale_nwp_method": cfg.data.scale_nwp_method,
        "scale_target_method": cfg.data.scale_target_method,
        "use_polars": cfg.data.use_polars,
    })
    static_data.update(_methods_static_data(
        cfg,
        _experiments_static_data(cfg),
        _inline_experiment_layer_params(cfg.experiments),
    ))
    validate_experiment_references(cfg, static_data['experiments'] or {})
    return static_data


def _flatten_coordinates(area_group: list[list[float]]) -> list[float]:
    return [coordinate for corner in area_group for coordinate in corner]


def config_from_yaml(yaml_path: str | Path) -> dict[str, Any]:
    cfg = load_config(yaml_path)
    return to_static_data(cfg)
