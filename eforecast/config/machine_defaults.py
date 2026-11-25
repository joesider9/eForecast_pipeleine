# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Machine-detected defaults mirroring ``configuration/config_utils.py``.

These are used by the Gradio editor to prefill the System tab (folders, jobs,
runtime environment) when creating a new configuration. Unlike the legacy
module, nothing raises if a folder does not exist; validation happens through
``FoldersConfig`` when ``check_exists`` is enabled.
"""

import os
import sys
from typing import Any


def find_pycharm_path() -> str:
    if not os.path.exists("/media/sider/data"):
        if not ("smartrue" in os.getcwd() or "joesider" in os.getcwd()):
            return "/client"
        return "/home/smartrue/Dropbox/current_codes/PycharmProjects"
    return "/media/sider/data/Dropbox/current_codes/PycharmProjects"


def default_folders(docker: bool = False) -> dict[str, str]:
    folders: dict[str, str] = {"pycharm_path": "/client" if docker else find_pycharm_path()}
    if docker:
        folders.update(
            sys_folder="/models/",
            nwp_folder="/nwp/",
            sat_folder="/nwp/EUMETSAT/Bitakos",
            path_image="/nwp/SAT_DATA",
        )
    elif not os.path.exists("/media/sider/data"):
        folders.update(
            sys_folder="/media/smartrue/HHD1/George/models/",
            nwp_folder="/media/smartrue/HHD2/",
            sat_folder="/media/smartrue/HHD2/Satellites/EUMETSAT/Greece",
            path_image="/media/smartrue/FastAssistant/SAT_DATA",
        )
    else:
        folders.update(
            sys_folder="/media/sider/data/Dropbox/data_transfer/models",
            nwp_folder="/media/sider/data/",
        )
    return folders


def default_jobs() -> dict[str, int]:
    if sys.platform != "linux":
        if os.path.exists("D:/"):
            return {"n_cpus": 18, "n_jobs": 18, "n_jobs_rbfnn": 4, "n_jobs_lstm": 3,
                    "n_jobs_cnn_3d": 1, "n_jobs_cnn": 1, "n_jobs_mlp": 2,
                    "intra_op": 2, "n_gpus": 1}
        return {"n_cpus": 18, "n_jobs": 18, "n_jobs_rbfnn": 3, "n_jobs_lstm": 3,
                "n_jobs_cnn_3d": 1, "n_jobs_cnn": 1, "n_jobs_mlp": 4,
                "intra_op": 2, "n_gpus": 1}
    if not os.path.exists("/media/sider/data"):
        return {"n_cpus": 16, "n_jobs": 16, "n_jobs_rbfnn": 3, "n_jobs_lstm": 2,
                "n_jobs_cnn_3d": 2, "n_jobs_cnn": 2, "n_jobs_mlp": 3,
                "intra_op": 2, "n_gpus": 2}
    return {"n_cpus": 20, "n_jobs": 20, "n_jobs_rbfnn": 3, "n_jobs_lstm": 2,
            "n_jobs_cnn_3d": 2, "n_jobs_cnn": 2, "n_jobs_mlp": 3,
            "intra_op": 2, "n_gpus": 1}


def default_environment(backend: str = "TORCH") -> tuple[str, str]:
    if backend not in ("TF_1", "TF_2", "TORCH"):
        raise ValueError(f"Wrong backend name {backend}. It should be one of TF_1, TF_2, TORCH")
    # These legacy labels never chose an interpreter. New cases should not advertise
    # a machine-specific path as a setting. Stored values are still round-tripped.
    return '', ''


def machine_defaults(docker: bool = False, backend: str = "TORCH") -> dict[str, dict[str, Any]]:
    """Return per-section defaults keyed like the YAML/raw configuration."""
    env_name, path_env = default_environment(backend)
    return {
        "folders": default_folders(docker),
        "jobs": default_jobs(),
        "methods": {"env_name": env_name, "path_env": path_env, "backend": backend},
    }
