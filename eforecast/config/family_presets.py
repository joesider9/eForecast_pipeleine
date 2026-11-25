# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Starting parameters of every method family, from the legacy ``config_methods.py``.

Values are the stored YAML forms: a scalar is fixed, ``{"choices": [...]}`` is searched among the
values, ``{"range": [lo, hi]}`` is sampled between them.  A few keys depend on the case and are
markers resolved by :func:`family_preset` against the document: ``ALL`` (every option of the
matching data list, as choices), ``FIRST`` (its first option), ``DATA_TYPE`` (the first option of
every data list), ``RBF_VAR_IMP`` (the RBF variables of the problem type) and the CatBoost
objective / metric that follow the horizon type.
"""
from __future__ import annotations

import copy
from typing import Any

from eforecast.config.presets import rbf_variables
from eforecast.config.sync_options import current_options, data_type_from_options

ALL, FIRST, DATA_TYPE, RBF_VAR_IMP, OBJECTIVE, EVAL_METRIC = "<ALL>", "<FIRST>", "<DATA_TYPE>", "<RBF_VAR_IMP>", "<OBJECTIVE>", "<EVAL_METRIC>"

_DATA_TYPE_COMMON = {"merge": ALL, "compress": None, "scale_row_method": ALL, "scale_nwp_method": ALL, "scale_img_method": None,
                "image_data_type": None, "feature_selection_method": None}

PRESETS: dict[str, dict[str, Any]] = {
    "CNN": {"experiment_tag": {"choices": ["cnn1"]}, "use_nwp": False, "n_trials": 3, "filters": 16, "stride": 1, "conv_dim": 2,
            "batch_norm": False, "use_norm": True, "attention": {"choices": ["cbam", "att", "geodatt"]}, "area_adjust": None,
            "final_image_size": None, "act_func": "gelu", "sat_image_type": "rgb_snow", "create_image_dataset_real_time": True,
            "merge": ALL, "compress": ALL, "scale_row_method": None, "scale_nwp_method": ALL,
            "feature_selection_method": None, "optimizer": "prodigy", "scheduler": "ReduceLROnPlateau", "transfer_learning_from_global": True,
            "learning_rate": 1e-4, "batch_size": 128, "max_iterations": 100, "warming_iterations": 0, "n_jobs": 3,
            "mixed_precision": False},

    "TRANSFORMER": {"experiment_tag": {"choices": ["transformer1", "transformer2", "transformer3"]}, "n_trials": 3, "model_name": {"choices": []},
                    "transfer_learning_from_global": True, "use_norm": True, "use_nwp": True, "filters": 16, "stride": 1, "conv_dim": 2,
                    "attention": "cbam", "batch_norm": False, **_DATA_TYPE_COMMON, "act_func": "gelu", "optimizer": "prodigy",
                    "scheduler": {"choices": ["ReduceLROnPlateau"]}, "batch_size": 128, "max_iterations": 96, "warming_iterations": 0,
                    "learning_rate": 1e-4, "n_jobs": 2, "mixed_precision": False},

    "TIMM": {"experiment_tag": {"choices": ["timm_net"]}, "n_trials": 1, "model_name": "visformer_small", "pretrained": {"choices": ["tune"]},
             "sat_image_type": None, "clip_value": False, "final_image_size": 160, "transfer_learning_from_global": True,
             "optimizer": "prodigy", "scheduler": "ReduceLROnPlateau", "fix_grid_search": False, **_DATA_TYPE_COMMON,
             "create_image_dataset_real_time": False, "spatal_coord": True, "area_adjust": {"eumetdac": 50, "eumetview": 80},
             "use_classes": False, "filters": 12, "conv_dim": 2, "batch_size": 128, "max_iterations": 72, "warming_iterations": 0,
             "learning_rate": {"range": [0.9e-4, 1.1e-4]}, "use_data": None, "act_func": "gelu", "n_jobs": 1,
             "mixed_precision": False},

    "LSTM": {"units": 24, "n_trials": 3, "experiment_tag": {"choices": ["lstm1"]}, "attention": {"choices": ["cbam", "att", "geodatt"]},
             "filters": 16, "stride": 1, "conv_dim": 2, "use_nwp": True, "batch_norm": False, "transfer_learning_from_global": True,
             "use_embedding": False, "merge": ALL, "compress": None, "scale_row_method": ALL, "scale_nwp_method": ALL,
             "feature_selection_method": ALL, "act_func": "gelu", "batch_size": 128, "max_iterations": 96, "warming_iterations": 5,
             "optimizer": "prodigy", "scheduler": {"choices": ["ReduceLROnPlateau"]}, "learning_rate": 1e-4, "n_jobs": 2},

    "MLP": {"experiment_tag": {"choices": ["mlp1", "mlp2", "mlp3", "mlp4"]}, "n_trials": 4, "hold_prob": 1, "fix_grid_search": False,
            "use_nwp": False, "batch_norm": False, "transfer_learning_from_global": True, "use_embedding": False, "filters": 16,
            "stride": 1, "conv_dim": 2, "merge": ALL, "compress": ALL, "scale_row_method": ALL, "scale_nwp_method": ALL,
            "scale_img_method": None, "image_data_type": None, "feature_selection_method": ALL, "act_func": "gelu", "batch_size": 512,
            "max_iterations": 100, "warming_iterations": 5, "optimizer": "prodigy", "scheduler": "ReduceLROnPlateau",
            "learning_rate": 1e-4, "n_jobs": 3},

    "CrossViVit": {"experiment_tag": {"choices": ["CrossViVit_net"]}, "n_trials": 2, "model_name": "visformer_small", "transformer_name": None,
                   "sat_image_type": {"choices": ["RBG:Infrared1:RBG_opt"]}, "pretrained": {"choices": ["tune"]},
                   "transfer_learning_from_global": False, "area_adjust": 24, "final_image_size": {"choices": [128, 224]},
                   "create_image_dataset_real_time": True, "stride": 1, "filters": 12, "conv_dim": 3, "optimizer": "adamw",
                   "scheduler": {"choices": ["ReduceLROnPlateau"]}, "merge": FIRST, "compress": FIRST, "scale_row_method": FIRST,
                   "scale_nwp_method": FIRST, "scale_img_method": ALL, "image_data_type": ALL, "feature_selection_method": None,
                   "use_data": 3, "act_func": "gelu", "batch_size": 128, "max_iterations": 100, "warming_iterations": 0,
                   "learning_rate": {"choices": [1e-3, 1e-5]}, "n_jobs": 1},

    "Distributed": {"experiment_tag": {"choices": ["distributed_cnn1", "distributed_mlp2"]}, "keep_n_models": 2, "n_trials": 10, "hold_prob": 1,
                    "batch_size": {"choices": [32, 64, 128]}, "filters": 6, "conv_dim": 2, "transfer_learning_from_global": True,
                     **_DATA_TYPE_COMMON, "merge": ALL, "compress": ALL, "scale_row_method": ALL, "scale_nwp_method": ALL,
                    "feature_selection_method": ALL, "act_func": {"choices": ["sigmoid"]}, "max_iterations": 800,
                    "optimizer": {"choices": ["prodigy", "adamw"]}, "scheduler": {"choices": ["CosineAnnealing"]}, "warming_iterations": 4,
                    "learning_rate": {"choices": [1e-3]}, "thres_act": 0.001, "min_samples": 200, "max_samples_ratio": 0.8,
                    "train_schedule": "simple", "n_rules": {"choices": [9]}, "is_fuzzy": {"choices": [False]},
                    "clustering_method": {"choices": ["RBF", None]}, "rbf_var_imp": RBF_VAR_IMP, "n_jobs": 2},

    "clustering": { "methods": ["RBF"], "clusters_for_method": "RBF", "prediction_for_method": ["RBF"],
                   "thres_act": 0.001, "rbf_var_imp": RBF_VAR_IMP, "explode_clusters": True, "n_var_lin": 24, "min_samples": 8000,
                   "max_samples_ratio": 0.45, "n_rules": 7,
                   "experiment_tag": "exp_fuzzy1", "n_trials": 1, "hold_prob": 1, "batch_size": 2048, "filters": 12, "conv_dim": 2,
                  "merge": FIRST, "compress": FIRST, "scale_row_method": FIRST, "scale_nwp_method": FIRST,
                  "feature_selection_method": None, "train_schedule": "simple", "optimizer": "adamw",
                  "scheduler": "CosineAnnealing", "act_func": None, "max_iterations": 1000, "warming_iterations": 2,
                  # the CPUs of the clusterer's worker: the six candidate widths of its rule start are
                  # fitted side by side with them (the same result as one after another)
                  "learning_rate": 0.25e-3, "n_jobs": 6},

    "RF": {"n_trials": 40, "max_depth": {"choices": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 42, 48]},
           "max_features": {"choices": ["sqrt", "log2", 0.8, 0.7, 0.5, 0.4, 0.3, 0.2]}, "min_samples_leaf": {"range": [2, 200]},
           "min_samples_split": {"range": [2, 100]}, "max_samples": {"range": [0.3, 1]}, "oob_score": True},

    "CatBoost": {"n_trials": 15, "iterations": 1000, "learning_rate": {"choices": [0.01, 0.05, 0.1]}, "l2_leaf_reg": {"range": [5, 6]},
                 "objective": OBJECTIVE, "min_data_in_leaf": {"range": [50, 100]}, "colsample_bylevel": {"range": [0.8, 1]},
                 "depth": {"range": [5, 9]}, "boosting_type": "Ordered", "bootstrap_type": {"choices": ["Bayesian", "Bernoulli", "MVS"]},
                 "eval_metric": EVAL_METRIC},

    "lightgbm": {"n_trials": 20, "num_iterations": 1000, "boosting_type": "gbdt", "learning_rate": {"choices": [0.01, 0.05, 0.1]},
                 "reg_lambda": 0, "num_leaves": {"range": [10, 100]}, "min_child_samples": {"range": [10, 100]},
                 "colsample_bytree": {"range": [0.8, 1]}, "max_depth": {"choices": [-1, 5, 9, 15, 21]}},

    "lasso": {"n_trials": 100, "max_iter": 150000, "eps": {"choices": [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]},
              "fit_intercept": {"choices": [True, False]}, "selection": {"choices": ["cyclic", "random"]}},

    "RBFols": {"n_trials": 10, "warming_iterations": 4, "width": {"range": [0.01, 15]}, "keep": {"range": [2, 6]}},

    "GA_RBFols": {"n_trials": 1000, "warming_iterations": 4, "width": {"range": [0.01, 15]}, "keep": {"range": [2, 6]}},
    # zero-shot foundation model on the lstm data (past observations + future covariates); nothing is trained

    "Chronos2": {"n_trials": 1, "model_name": "amazon/chronos-2", "covariates": "all", "context": None,
                 "quantile_levels": [0.1, 0.5, 0.9], "batch_size": 256, "device": "auto", "cross_learning": False, "n_jobs": 1},

    "TimesFM": {"n_trials": 1, "model_name": "google/timesfm-2.5-200m-pytorch", "covariates": "calendar", "context": None,
                "quantile_levels": [0.1, 0.5, 0.9], "batch_size": 256, "device": "auto", "xreg_mode": "xreg + timesfm", "n_jobs": 1},

    "TiRex": {"n_trials": 1, "model_name": "NX-AI/TiRex", "covariates": "none", "context": None,
              "quantile_levels": [0.1, 0.5, 0.9], "batch_size": 256, "device": None, "n_jobs": 1},
}


def _resolve(value: Any, options: dict[str, list[Any]], key: str, document: dict[str, Any]) -> Any:
    project = document.get("project") or {}
    multi = project.get("horizon_type") == "short_term"
    if value == ALL:
        return {"choices": list(options.get(key, [None]))}
    if value == FIRST:
        return options.get(key, [None])[0]
    if value == DATA_TYPE:
        return data_type_from_options(options)
    if value == RBF_VAR_IMP:
        rules = rbf_variables(project.get("type") or "")
        return rules if key == "rbf_var_imp" and "clustering" in document.get("_family", "") else rules[0]
    if value == OBJECTIVE:
        return {"choices": ["MultiRMSE"]} if multi else {"choices": ["RMSE", "MAE"]}
    if value == EVAL_METRIC:
        return "MultiRMSE" if multi else "MAE"
    if isinstance(value, dict) and not (set(value) <= {"choices", "range"}):
        return {k: _resolve(v, options, k, document) for k, v in value.items()}
    return copy.deepcopy(value)


def family_preset(name: str, document: dict[str, Any]) -> dict[str, Any]:
    """The preset of one family with the data-dependent keys resolved against ``document``; {} when unknown."""
    preset = PRESETS.get(name)
    if preset is None:
        return {}
    options = current_options(document)
    doc = dict(document, _family=name)
    resolved = {key: _resolve(value, options, key, doc) for key, value in preset.items()}
    if name == "clustering":
        resolved["rbf_var_imp"] = rbf_variables((document.get("project") or {}).get("type") or "")
    return resolved


def preset_keys() -> dict[str, list[str]]:
    return {name: list(preset) for name, preset in PRESETS.items()}
