# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""What a case has produced so far: dataset files, clusters, the Optuna studies of every cluster and
method, the selected trials and the evaluation tables.  Read-only; nothing is created.

The paths come from the case itself (``preview_static_data`` builds the same ``static_data`` the
pipeline uses without creating folders), so the page works before anything ran.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from eforecast.config.service import ServiceError, preview_static_data

STATES = ("COMPLETE", "RUNNING", "PRUNED", "FAIL", "WAITING")


def _static(document: dict[str, Any], path: str | None) -> dict[str, Any]:
    try:
        static_data = preview_static_data(document, path)
    except Exception as exc:  # noqa: BLE001 - pydantic or loader errors
        raise ServiceError(f"the case does not validate: {str(exc).splitlines()[0]}") from exc
    try:
        from eforecast.init.initialize import initializer
    except ImportError as exc:   # the configuration package alone is installed (no joblib, OpenCV, ...)
        raise ServiceError(f"the studies page needs the pipeline packages of requirements.txt ({exc.name} is missing)") from exc
    return initializer(static_data, dry_run=True)


def _size(path: str) -> float | None:
    return round(os.path.getsize(path) / 1e6, 1) if os.path.exists(path) else None


def _read_csv(path: Path, limit: int = 400) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return {"columns": [], "rows": [], "path": str(path)}
    return {"columns": rows[0], "rows": rows[1:limit + 1], "path": str(path), "total": len(rows) - 1}


def links() -> dict[str, str]:
    """Where a maintainer looks next. One definition, used by the studies page and A14-03's
    support report -- two copies of a URL is one copy that goes stale."""
    return {"optuna_dashboard": os.environ.get("EFORECAST_OPTUNA_DASHBOARD_URL", "http://localhost:8080"),
            "grafana": os.environ.get("EFORECAST_GRAFANA_URL", "http://localhost:3000"),
            "metrics": os.environ.get("EFORECAST_METRICS_URL", "http://localhost:9108/metrics")}


def clusters_of(static_data: dict[str, Any]) -> dict[str, str]:
    """{cluster name: folder}; ``global`` first, the clusters from ``clusters.pickle`` when it exists."""
    path_model = static_data["path_model"]
    out = {"global": os.path.join(path_model, "global")}
    pickle_path = os.path.join(path_model, "clusters.pickle")
    if os.path.exists(pickle_path):
        import joblib
        try:
            out.update({str(k): str(v) for k, v in joblib.load(pickle_path).items()})
        except Exception:  # noqa: BLE001
            pass
    return out


def study_summary(study) -> dict[str, Any]:
    trials = study.get_trials(deepcopy=False)
    counts = {state: 0 for state in STATES}
    for t in trials:
        counts[t.state.name] = counts.get(t.state.name, 0) + 1
    best = None
    complete = [t for t in trials if t.state.name == "COMPLETE" and t.value is not None]
    if complete:
        b = min(complete, key=lambda t: t.value)
        best = {"number": b.number, "value": b.value, "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in b.params.items()}}
    last = max((t.datetime_complete or t.datetime_start for t in trials if (t.datetime_complete or t.datetime_start)), default=None)
    return {"n_trials": len(trials), "counts": counts, "best": best, "last_activity": last.isoformat() if last else None,
            "user_attrs": {k: v for k, v in study.user_attrs.items() if isinstance(v, (int, float, str, bool))}}


def studies(document: dict[str, Any], path: str | None = None, load: bool = True) -> dict[str, Any]:
    """Datasets, clusters, studies per (cluster, method) with trial counts and best value, selections and evaluation."""
    static_data = _static(document, path)
    try:
        from eforecast.datasets.files_manager import FilesManager
    except ImportError as exc:
        raise ServiceError(f"the studies page needs the pipeline packages of requirements.txt ({exc.name} is missing)") from exc
    datasets = {name: {"path": file, "size_mb": None if size is None else round(size, 1)}
                for name, (file, size) in FilesManager(static_data).dataset_status().items()}
    clusters = clusters_of(static_data)
    storage_url = os.environ.get("EFORECAST_OPTUNA_STORAGE") or None
    rows: list[dict[str, Any]] = []
    for cluster_name, cluster_dir in clusters.items():
        if not os.path.isdir(cluster_dir):
            continue
        for method in sorted(os.listdir(cluster_dir)):
            from eforecast.common_utils.trial_layout import method_directory
            method_dir = str(method_directory(static_data, cluster_dir, method))
            journal = os.path.join(method_dir, "study.journal")
            if not os.path.isdir(method_dir) or not (os.path.exists(journal) or storage_url):
                continue
            n_folders = len([d for d in os.listdir(method_dir) if d.startswith("trial") or d.startswith("test_")])
            row: dict[str, Any] = {"cluster": cluster_name, "method": method, "folder": method_dir, "journal": journal if os.path.exists(journal) else None,
                                   "trial_folders": n_folders, "selected": None, "summary": None}
            selected = Path(method_dir) / "selected_trials.json"
            if selected.is_file():
                try:
                    row["selected"] = json.loads(selected.read_text())
                except ValueError:
                    row["selected"] = None
            if load:
                try:
                    import optuna
                    from eforecast.training.command_line.optuna_search import resolve_study_name, storage_for
                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                    storage = storage_for(cluster_dir, method, storage_url, static_data)
                    # a study created before the identity was in the name keeps that name
                    study = optuna.load_study(
                        study_name=resolve_study_name(static_data, cluster_name, method, storage), storage=storage)
                    row["summary"] = study_summary(study)
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            rows.append(row)
    results_dir = Path(static_data["path_model"]) / "Results"
    return {
        "project": static_data["_id"], "path_model": static_data["path_model"], "path_data": static_data["path_data"],
        "path_logs": static_data.get("path_logs"), "model_exists": os.path.isdir(static_data["path_model"]),
        "datasets": datasets,
        "clusters": {name: {"folder": folder, "exists": os.path.isdir(folder)} for name, folder in clusters.items()},
        "studies": rows,
        "evaluation": _read_csv(results_dir / "evaluation.csv"),
        "summary": _read_csv(results_dir / "summary.csv"),
        "intervals": _read_csv(results_dir / "intervals_summary.csv"),
        "links": links(),
    }


def trials(document: dict[str, Any], cluster: str, method: str, path: str | None = None, limit: int = 500) -> dict[str, Any]:
    """Every trial of one study: number, state, value, duration, parameters, user attributes."""
    static_data = _static(document, path)
    clusters = clusters_of(static_data)
    if cluster not in clusters:
        raise ServiceError(f"unknown cluster {cluster!r}")
    import optuna
    from eforecast.training.command_line.optuna_search import resolve_study_name, storage_for
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cluster_dir = clusters[cluster]
    storage_url = os.environ.get("EFORECAST_OPTUNA_STORAGE") or None
    from eforecast.training.command_line.optuna_search import journal_path
    if not storage_url and not os.path.exists(journal_path(cluster_dir, method, static_data)):
        raise ServiceError(f"no study for {cluster}/{method}")
    storage = storage_for(cluster_dir, method, storage_url, static_data)
    study = optuna.load_study(study_name=resolve_study_name(static_data, cluster, method, storage), storage=storage)
    out = []
    for t in study.get_trials(deepcopy=False)[-limit:]:
        duration = (t.datetime_complete - t.datetime_start).total_seconds() if t.datetime_complete and t.datetime_start else None
        out.append({"number": t.number, "state": t.state.name, "value": t.value, "duration_s": duration,
                    "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in t.params.items()},
                    "user_attrs": {k: v for k, v in t.user_attrs.items() if isinstance(v, (int, float, str, bool))},
                    "started": t.datetime_start.isoformat() if t.datetime_start else None})
    params = sorted({k for t in out for k in t["params"]})
    return {"study": study.study_name, "trials": out, "param_names": params, "summary": study_summary(study)}
