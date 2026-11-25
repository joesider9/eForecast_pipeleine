# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Choice lists of the configuration web UI (served by ``/api/options``).

One place for the vocabularies that are not in the pydantic schema: variable types and sources,
transformers, NWP providers, the legacy option lists of the data section, the method selections
and their families, and which parameters of a family are categorical (``SET_VALUED_KEYS`` of the
loader decides how a plain list is read).
"""
from __future__ import annotations

from typing import Any

from eforecast.config.loader import SET_VALUED_KEYS
from eforecast.config.family_presets import preset_keys
from eforecast.config.sync_options import DATA_TYPE_KEYS
from eforecast.config.capabilities import (CPU_METHODS, CLUSTER_METHODS, GLOBAL_METHODS, COMBINE_METHODS,
                                            CLUSTERING_METHODS, BACKENDS)
from eforecast.nwp_extraction.gdrive import FOLDERS as NWP_GDRIVE_FOLDERS

NONE_LABEL = 'None'

VARIABLE_TYPES: list[str] = ['nwp', 'timeseries', 'calendar', 'image']
VARIABLE_SOURCES: list[str] = ['grib', 'database', 'target', 'nwp_dataset', 'index', 'astral', 'eumetview', 'eumetdac']
TRANSFORMER_CHOICES: list[str] = ['mean', 'clear_sky', 'pv_clear_sky', 'img_clear_sky', 'area_adjust', 'resize', 'fillnan', 'alignment', 'inverse', 'crop', 'normalize', 'sum', 'cyclical']
NWP_MODEL_CHOICES: list[str] = ["ecmwf", "skiron", "gfs"]
NWP_RESOLUTIONS: list[float] = [0.25, 0.1, 0.05]
NWP_LEVELS: list[int] = [2, 4, 8]
TIMEZONES: list[str] = ["UTC", "CET", "Etc/GMT-1", "Europe/Athens"]
TS_RESOLUTIONS: list[str] = ["1min", "5min", "15min", "H", "D"]
# name suggestions per variable type (free entry stays possible) and the usual source of each type
VARIABLE_NAME_PRESETS: dict[str, list[str]] = {
    "timeseries": ["Obs"],
    "nwp": ["Uwind", "Vwind", "WS", "WD", "Uwind10", "Vwind10", "WS10", "WD10", "Temperature", "Cloud", "High_Cloud", "Low_Cloud",
            "Medium_Cloud", "Flux", "Pressure", "Humidity", "Clear_Sky_Flux", "WindGust"],
    "calendar": ["month", "dayweek", "dayofyear", "hour", "minute", "sp_index", "horizon"],
    "image": ["ir039_ir108_vis006", "rgb_snow", "rgb_fog", "RBG"],
}
VARIABLE_SOURCE_BY_TYPE: dict[str, str] = {"timeseries": "target", "nwp": "grib", "calendar": "index", "image": "eumetview"}
KERNEL_PRESETS: dict[str, list[list[int]]] = {
    "nwp": [[2, 2, 1], [2, 1, 2], [2, 2, 2], [1, 1, 2], [1, 2, 1], [1, 2, 2]],
    "images": [[2, 5, 5], [2, 10, 10], [2, 40, 40], [4, 30, 40], [4, 30, 30], [2, 30, 30]],
}
SCALE_CHOICES: list[str] = ['None', 'minmax', 'standard', 'maxabs']
DATA_LIST_FIELDS: dict[str, list[str]] = {
    "nwp_data_merge": [
        "None",
        "all",
        "by_area",
        "by_area_variable",
        "by_variable",
        "by_nwp_provider"
    ],
    "compress_data": [
        "None",
        "dense",
        "semi_full",
        "full",
        "load"
    ],
    "img_data": [
        "None"
    ],
    "scale_row_method": [
        "None",
        "minmax",
        "standard",
        "maxabs"
    ],
    "scale_nwp_method": [
        "None",
        "minmax",
        "standard",
        "maxabs"
    ],
    "scale_img_method": [
        "None",
        "minmax",
        "standard",
        "maxabs"
    ]
}

METHODS_CPU_CHOICES: list[str] = list(CPU_METHODS)
CLUSTER_METHOD_CHOICES: list[str] = list(CLUSTER_METHODS)
GLOBAL_METHOD_CHOICES: list[str] = list(GLOBAL_METHODS)
COMBINE_CHOICES: list[str] = list(COMBINE_METHODS)
CLUSTERING_CHOICES: list[str] = list(CLUSTERING_METHODS)
FEATURE_SELECTION_CHOICES: list[str] = ['None', 'ShapValues', 'Leshy_native', 'Leshy_pimp', 'Leshy_shap', 'BoostAGroota_native', 'BoostAGroota_pimp', 'BoostAGroota_shap', 'lasso', 'boruta']
BACKEND_CHOICES: list[str] = list(BACKENDS)
METHOD_FAMILIES: list[str] = ['CNN', 'TRANSFORMER', 'TIMM', 'LSTM', 'MLP', 'CrossViVit', 'Distributed', 'RF', 'CatBoost', 'lightgbm', 'lasso', 'RBFols', 'GA_RBFols', 'Chronos2', 'TimesFM', 'TiRex']
FLAG_FIELDS: dict[str, list[str]] = {
    "methods_cpu": METHODS_CPU_CHOICES,
    "cluster_methods": CLUSTER_METHOD_CHOICES,
    "global_methods": GLOBAL_METHOD_CHOICES,
}
METHODS_LIST_FIELDS: dict[str, list[str]] = {
    "feature_selection_methods": FEATURE_SELECTION_CHOICES,
    "combine_methods": COMBINE_CHOICES,
}


# Parameters of a method family whose value is a structure rather than a scalar.  The parameter
# table of the editor renders a scalar, a set of choices or a range; anything else used to fall
# through to a one-line YAML field, and a two-number list was even mistaken for a searched range
# (``quantile_levels: [0.1, 0.9]``).  Declaring the kind here gives each one a real editor and
# stops the guessing.
#
#   number_list  an ordered list of numbers           quantile_levels
#   option_list  a list drawn from a fixed vocabulary clustering.methods
#   number_map   a mapping of name -> number          TIMM.area_adjust
#   rbf_var_imp  the fuzzy rule sets                  clustering / Distributed
#
# Only keys that really appear inside ``methods.method_params`` belong here: a list field of the
# schema itself (``methods.interval_coverage``) already gets an array editor from the form.
STRUCTURED_PARAMS: dict[str, dict[str, Any]] = {
    "quantile_levels": {
        "kind": "number_list", "minimum": 0.0, "maximum": 1.0, "sorted": True, "step": 0.05,
        "label": "quantile levels",
        "help": "probabilities the model predicts, between 0 and 1 and in order; 0.5 is the median",
        "presets": [[0.1, 0.5, 0.9], [0.05, 0.25, 0.5, 0.75, 0.95], [0.5]],
    },
    "methods": {
        "kind": "option_list", "options": CLUSTERING_CHOICES, "label": "clustering methods",
        "help": "how the clusters are built; only RBF is implemented",
    },
    "prediction_for_method": {
        "kind": "option_list", "options": CLUSTERING_CHOICES, "label": "predict for",
        "help": "the clustering methods whose clusters get their own models",
    },
    "clusters_for_method": {
        # ClusterOrganizer.makes_clusters: a method builds its cluster folders when it is this
        # one, or when this is 'both', which means every method in the list above.
        "kind": "option_list", "single": True, "options": [*CLUSTERING_CHOICES, "both"],
        "label": "build clusters for",
        "help": "the clustering method whose clusters get folders and masks; 'both' means every method",
    },
    "area_adjust": {
        "kind": "number_map", "keys": ["eumetdac", "eumetview"], "label": "area adjust",
        "help": "pixels cropped around the site, per image provider",
    },
    "rbf_var_imp": {
        "kind": "rbf_var_imp", "label": "fuzzy variables",
        "help": "one rule set per fuzzy network; each entry names row_all variables",
    },
}


def structured_param(key: str) -> dict[str, Any] | None:
    """The declaration of a structured parameter, or ``None`` for an ordinary one."""
    return STRUCTURED_PARAMS.get(key)


def ui_options() -> dict[str, Any]:
    """Everything the web UI needs to build its pickers, JSON-ready."""
    return {
        "none_label": NONE_LABEL,
        "variable_types": VARIABLE_TYPES,
        "variable_sources": VARIABLE_SOURCES,
        "transformers": TRANSFORMER_CHOICES,
        "nwp_models": NWP_MODEL_CHOICES,
        # the shared Google Drive folder each provider's archive is in: the providers that offer
        # that source in the Project tab, and what choosing it fills in
        "nwp_gdrive": dict(NWP_GDRIVE_FOLDERS),
        "data_lists": DATA_LIST_FIELDS,
        "methods_lists": METHODS_LIST_FIELDS,
        "flags": FLAG_FIELDS,
        "families": METHOD_FAMILIES,
        "clustering_methods": CLUSTERING_CHOICES,
        "backends": BACKEND_CHOICES,
        "set_valued_keys": {method: sorted(keys) for method, keys in SET_VALUED_KEYS.items()},
        "timezones": TIMEZONES,
        "ts_resolutions": TS_RESOLUTIONS,
        "nwp_resolutions": NWP_RESOLUTIONS,
        "nwp_levels": NWP_LEVELS,
        "variable_names": VARIABLE_NAME_PRESETS,
        "variable_source_by_type": VARIABLE_SOURCE_BY_TYPE,
        "kernel_presets": KERNEL_PRESETS,
        "data_type_keys": DATA_TYPE_KEYS,
        "family_preset_keys": preset_keys(),
        "structured_params": STRUCTURED_PARAMS,
    }
