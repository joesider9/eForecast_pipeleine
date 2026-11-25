# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The case document as operations: validate it, preview it, read and write it.

One half of what ``service`` used to be. A *document* is the merged YAML as a dict; a *case path*
is a single ``<name>.yaml``, a parts directory or its ``main.yaml`` (see ``config_files``). These
are plain functions with no UI and no workspace: which file a case lives in, and who may read it,
is ``workspace``'s question, not this module's.

The editors and the web API (``eforecast.config.api``) call these; ``service`` re-exports them.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from eforecast.config import presets as presets_module
from eforecast.config.config_files import (load_document, missing_sections, save_document, validation_note)
from eforecast.config.loader import to_static_data
from eforecast.config.machine_defaults import machine_defaults
from eforecast.config.schema import (lag_step, DataConfig, ExperimentConfig, FoldersConfig, JobsConfig, MethodsConfig,
                                     ProjectConfig)
from eforecast.config.sync_options import (apply_base_data_types, describe_sync, ensure_clustering_family,
                                           sync_families_with_data)

SECTION_MODELS = {"project": ProjectConfig, "folders": FoldersConfig, "jobs": JobsConfig, "data": DataConfig,
                  "methods": MethodsConfig}


# ---------------------------------------------------------------------------- validation and preview

def validation_errors(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Pydantic errors of the whole document as ``{'path': 'data.variables.3.lags', 'message': ...}``; [] when valid."""
    try:
        ExperimentConfig.model_validate(document)
    except ValidationError as exc:
        # a validator on the whole model has no field location; name the document instead of
        # rendering an empty path next to the message
        return [{"path": ".".join(str(item) for item in error["loc"]) or "(whole case)",
                 "message": error["msg"], "type": error["type"]} for error in exc.errors()]
    return []


def validate_section(section: str, raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Errors of one section on its own (the editors validate a part before merging it)."""
    model = SECTION_MODELS[section]
    try:
        model.model_validate(raw)
    except ValidationError as exc:
        return [{"path": ".".join([section] + [str(item) for item in error["loc"]]), "message": error["msg"],
                 "type": error["type"]} for error in exc.errors()]
    return []


def validate(document: dict[str, Any]) -> dict[str, Any]:
    """``{'valid', 'complete', 'missing_sections', 'errors'}`` of a document."""
    missing = missing_sections(document)
    errors = [] if missing else validation_errors(document)
    return {"valid": not missing and not errors, "complete": not missing, "missing_sections": missing, "errors": errors}


def preview_static_data(document: dict[str, Any], path: str | None = None) -> dict[str, Any]:
    """The ``static_data`` the pipeline would build from the document (raises ValidationError when invalid)."""
    cfg = ExperimentConfig.model_validate(document)
    if path:
        object.__setattr__(cfg, "_yaml_path", str(path))
    return to_static_data(cfg)


def derived_facts(document: dict[str, Any]) -> dict[str, Any]:
    """What the pipeline derives from the project section: target lags and columns, the data folder."""
    project = document.get("project") or {}
    horizon_type = project.get("horizon_type")
    horizon = int(project.get("horizon") or 0)
    ts_resolution = project.get("ts_resolution") or "H"
    per_hour = round(1 / lag_step(ts_resolution))
    # the loader's rule: a short_term horizon of H hours is H / step target lags (16 for 4 h at 15 min)
    steps = int(horizon / lag_step(ts_resolution)) if horizon_type == "short_term" else (1 if horizon_type else 0)
    folders = document.get("folders") or {}
    path_data = None
    if folders.get("sys_folder") and project.get("owner") and project.get("group"):
        path_data = str(Path(folders["sys_folder"]) / project["owner"]
                        / f"{project['group']}_ver{project.get('version_group', 0)}")
    return {"steps": steps, "target_lags": list(range(steps)), "target_columns": [f"Step_{i}" for i in range(steps)],
            "samples_per_hour": per_hour, "group_folder": path_data}


def schema() -> dict[str, Any]:
    return ExperimentConfig.model_json_schema()


def defaults(docker: bool = False, backend: str = "TORCH") -> dict[str, Any]:
    return machine_defaults(docker=docker, backend=backend)


# ---------------------------------------------------------------------------- documents on disk

def load_case(path: str | Path) -> dict[str, Any]:
    """The whole document behind a case path (``{}`` for a new case)."""
    return load_document(path)


def save_case(document: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Write the single file and the part files; the document must be complete and valid."""
    ExperimentConfig.model_validate(document)
    main, parts = save_document(document, path)
    return {"main": str(main), "parts": [str(part) for part in parts], "note": validation_note(document)}


def write_document(document: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Write the single file and the parts without requiring completeness (the editors save step by step;
    a complete document is validated by the caller through ``merge_sections``)."""
    main, parts = save_document(document, path)
    return {"main": str(main), "parts": [str(part) for part in parts], "note": validation_note(document)}


def merge_sections(path: str | Path, raws: dict[str, dict[str, Any]], replace: tuple[str, ...] = ()) -> dict[str, Any]:
    """The existing document with ``raws`` merged in: sections in ``replace`` are replaced whole, the others
    updated key by key.  Each section is validated on its own; the whole document when it is complete."""
    for section, raw in raws.items():
        if section in SECTION_MODELS:
            SECTION_MODELS[section].model_validate(raw)
    merged = copy.deepcopy(load_document(path))
    for section, raw in raws.items():
        if section in replace or section not in merged:
            merged[section] = raw
        elif isinstance(merged.get(section), dict):
            merged[section] = {**merged[section], **raw}
        else:
            merged[section] = raw
    if not missing_sections(merged):
        ExperimentConfig.model_validate(merged)
    return merged


def save_sections(path: str | Path, raws: dict[str, dict[str, Any]], replace: tuple[str, ...] = ()) -> dict[str, Any]:
    """Merge and write; the editors' "save this part" operation.  Returns paths, the merged document and the note."""
    merged = merge_sections(path, raws, replace)
    main, parts = save_document(merged, path)
    return {"main": str(main), "parts": [str(part) for part in parts], "document": merged,
            "note": validation_note(merged)}


def apply_data_types(document: dict[str, Any]) -> dict[str, Any]:
    """The base data types pushed to every family, the clusterer and Distributed; the combiners use them as they are."""
    document = copy.deepcopy(document)
    document, changes = apply_base_data_types(document)
    where = sorted({f"{c['where']}.{c['key']}" for c in changes})
    note = (f"Base data types applied to {len(changes)} value(s): {', '.join(where)}." if changes
            else "Every family, the clusterer and Distributed already hold the base data types.")
    return {"document": document, "changes": changes, "note": note}


def sync_document(document: dict[str, Any]) -> dict[str, Any]:
    """Method families follow the data options, and a fuzzy project gets its clusterer.

    Returns the document and a description of the changes.
    """
    document = copy.deepcopy(document)
    document, added = ensure_clustering_family(document)
    document, changes, skipped = sync_families_with_data(document)
    note = describe_sync(changes, skipped)
    if added:
        note = ("Added the `clustering` family from the presets: the project has `is_fuzzy` on, so it "
                "is trained per cluster and the clusterer needs it. " + note)
    return {"document": document, "changes": added + changes, "skipped": skipped, "note": note}


def variable_presets(problem_type: str, horizon: int, horizon_type: str, ts_resolution: str,
                     timezone: str = "UTC") -> list[dict[str, Any]]:
    return [variable.model_dump(exclude_none=True) for variable in
            presets_module.variables(problem_type, int(horizon), horizon_type, ts_resolution, timezone)]
