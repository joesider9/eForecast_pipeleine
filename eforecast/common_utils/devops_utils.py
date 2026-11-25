# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Helpers of the training step that survived the move to Optuna and the orchestrator:
the CUDA OOM detector, the per-(cluster, method) results table and a free-memory probe.
"""
import os

import joblib
import numpy as np
import pandas as pd
import torch

from eforecast.common_utils.train_utils import distance


def is_cuda_oom(exception: str) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, XPU out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "out of memory.",  # CUDA OOM
        "CUDA out of memory.",  # CUDA OOM
        "XPU out of memory.",  # XPU OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    return any(err in exception for err in _statements)


def get_results(clusters, method, refit, static_data=None):
    from . import trial_layout
    best_models = dict()
    for cluster_name, cluster_dir in clusters.items():
        if not os.path.exists(str(trial_layout.results_file(static_data or {}, cluster_dir, cluster_name, method))) or refit:
            path_trials = str(trial_layout.method_directory(static_data or {}, cluster_dir, method) / 'trials')
            if not os.path.exists(path_trials):
                continue
            shared_trials = []
            for trial in sorted(os.listdir(path_trials)):
                shared_trials.append(joblib.load(os.path.join(path_trials, trial)))
            trials = []
            for trial in shared_trials:
                param_dict = dict()
                for key in trial.keys():
                    param_dict[key] = trial[key]
                trials.append(param_dict)

            trials = pd.DataFrame(trials)
            results = trials.sort_values(by='value', ascending=False)
            cols = ['mae_test', 'mae_val',
                    'sse_test', 'sse_val']
            res = results[cols]
            res = res.clip(1e-6, 1e6)
            diff_mae = pd.DataFrame(np.abs(res['mae_test'].values - res['mae_val'].values),
                                    index=res.index, columns=['diff_mae'])
            res = pd.concat([res, 100 * diff_mae], axis=1)
            diff_sse = pd.DataFrame(np.abs(res['sse_test'].values - res['sse_val'].values), index=res.index,
                                    columns=['diff_sse'])
            res = pd.concat([res, diff_sse], axis=1)
            res_old, res_max, res_min = np.inf * np.ones(6), np.inf * np.ones(6), np.inf * np.ones(6)
            i = 0
            best_trials = []
            weights = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            while res.shape[0] > 0:
                flag_res, res_old, res_max, res_min = distance(res.iloc[i].values, res_old, res_max, res_min,
                                                               weights=weights)
                if flag_res:
                    best = i
                i += 1
                if i == res.shape[0]:
                    best_trials.append(res.index[best])
                    i = 0
                    res_old, res_max, res_min = np.inf * np.ones(6), np.inf * np.ones(6), np.inf * np.ones(6)
                    res = res.drop(index=res.index[best])
            results = results.loc[best_trials]
            results.to_csv(str(trial_layout.results_file(static_data or {}, cluster_dir, cluster_name, method)))

            best_models[cluster_name] = dict()
            best_models[cluster_name]['best'] = results.trial_number.values[0]
            best_models[cluster_name]['path'] = cluster_dir
    return best_models


def get_cuda_available_memory():
    # Always allow for 1 GB overhead.
    if torch.cuda.is_available():
        return get_cuda_assumed_available_memory()
    return 0
