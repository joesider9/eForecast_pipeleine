# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Keep an experiment configuration both as one YAML file and as several parts.

A case study can be stored in two equivalent shapes:

* **one file** (``configuration/cases/ptolemaida_pv.yaml`` or ``<dir>/main.yaml``)
  holding the whole document, and
* **a directory of parts**, one file per editor tab, merged back in this order::

      system.yaml           folders, jobs, methods.{env_name, path_env, backend}
      project.yaml          project
      input.yaml            data
      methods.yaml          methods (selection, combining, transfer learning, evaluation)
      experiments.yaml      experiments_module, experiments
      clustering.yaml       methods.base_data_types, methods.method_params.clustering
      method_families.yaml  methods.kernels, methods.method_params.<family>

``save_document`` always writes both shapes, ``load_document`` reads whichever
exists (the parts win when the directory holds any of them).  The four Gradio
web UI service goes through these two functions only.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .schema import ExperimentConfig, MethodsConfig

MAIN_FILE = "main.yaml"
PART_NAMES: tuple[str, ...] = ("system", "project", "input", "methods", "experiments",
                               "clustering", "method_families")
YAML_SUFFIXES = (".yaml", ".yml")

ENV_FIELDS: tuple[str, ...] = ("env_name", "path_env", "backend")
CLUSTERING_KEY = "clustering"
TOP_ORDER: tuple[str, ...] = ("project", "folders", "jobs", "data", "methods",
                              "experiments_module", "experiments")
METHODS_ORDER: tuple[str, ...] = tuple(MethodsConfig.model_fields)

# What every part holds, for messages and docs (the split itself is in ``split_config``).
PART_CONTENTS: dict[str, str] = {
    "system": "folders, jobs, methods.env_name / path_env / backend",
    "project": "project",
    "input": "data",
    "methods": "methods (method selection, combining, transfer learning, evaluation)",
    "experiments": "experiments_module, experiments",
    "clustering": "methods.base_data_types, methods.method_params.clustering",
    "method_families": "methods.kernels, methods.method_params.<family>",
}


# ---- paths -------------------------------------------------------------------
def is_part_path(path: str | Path) -> bool:
    """True for ``<dir>/main.yaml`` and for ``<dir>/<part>.yaml``."""
    path = Path(path)
    return path.suffix in YAML_SUFFIXES and (path.name == MAIN_FILE or path.stem in PART_NAMES)


def config_paths(path: str | Path) -> tuple[Path, Path]:
    """``(single file, parts directory)`` for a user supplied path.

    * a directory (or a path without suffix) -> ``<dir>/main.yaml`` and ``<dir>``
    * ``<dir>/main.yaml`` or ``<dir>/<part>.yaml`` -> ``<dir>/main.yaml`` and ``<dir>``
    * any other ``<name>.yaml`` -> that file and the sibling directory ``<name>/``
    """
    path = Path(path)
    if path.is_dir() or path.suffix == "":
        return path / MAIN_FILE, path
    if is_part_path(path):
        return path.parent / MAIN_FILE, path.parent
    return path, path.with_suffix("")


def part_path(split_dir: str | Path, name: str) -> Path:
    return Path(split_dir) / f"{name}.yaml"


def existing_parts(split_dir: str | Path) -> list[Path]:
    return [part_path(split_dir, name) for name in PART_NAMES
            if part_path(split_dir, name).is_file()]


# ---- split / merge ------------------------------------------------------------
def _compact(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


def split_config(raw: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Partition a whole document into the seven parts (every key goes to one part)."""
    raw = copy.deepcopy(raw or {})
    methods = dict(raw.get("methods") or {})
    method_params = dict(methods.pop("method_params", None) or {})
    clustering = method_params.pop(CLUSTERING_KEY, None)
    env = {key: methods.pop(key) for key in ENV_FIELDS if key in methods}
    base_data_types = methods.pop("base_data_types", None)
    kernels = methods.pop("kernels", None)

    parts: dict[str, dict[str, Any]] = {
        "system": _compact({"folders": raw.get("folders"), "jobs": raw.get("jobs"),
                            "methods": env or None}),
        "project": _compact({"project": raw.get("project")}),
        "input": _compact({"data": raw.get("data")}),
        "methods": _compact({"methods": methods or None}),
        "experiments": _compact({"experiments_module": raw.get("experiments_module"),
                                 "experiments": raw.get("experiments")}),
        "clustering": _compact({"methods": _compact({
            "base_data_types": base_data_types,
            "method_params": ({CLUSTERING_KEY: clustering}
                              if clustering is not None else None)}) or None}),
        "method_families": _compact({"methods": _compact({
            "kernels": kernels, "method_params": method_params or None}) or None}),
    }
    # keys the schema does not know yet stay with the system part, so nothing is lost
    for key, value in raw.items():
        if key not in TOP_ORDER:
            parts["system"][key] = value
    return parts


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _ordered(mapping: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    known = {key: mapping[key] for key in order if key in mapping}
    known.update({key: value for key, value in mapping.items() if key not in known})
    return known


def merge_configs(*parts: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge the parts in order (later parts win) and restore the canonical key order."""
    merged: dict[str, Any] = {}
    for part in parts:
        if part:
            _deep_update(merged, part)
    merged = _ordered(merged, TOP_ORDER)
    if isinstance(merged.get("methods"), dict):
        merged["methods"] = _ordered(merged["methods"], METHODS_ORDER)
    return merged


# ---- read / write -------------------------------------------------------------
def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


class _Dumper(yaml.SafeDumper):
    """``safe_dump`` with one exception: a layer parameter is written on a single line.

    ``{tag: fixed, value: 64}`` beats six lines of block mapping when a branch has a dozen of them.
    """


def _represent_mapping(dumper: yaml.SafeDumper, data: dict[str, Any]):
    inline = set(data) == {"tag", "value"} and not isinstance(data.get("value"), dict)
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=inline)


_Dumper.add_representer(dict, _represent_mapping)


def write_yaml(path: str | Path, document: dict[str, Any], comment: str | None = None,
               part: bool | None = None) -> Path:
    """Write a document; the first line names the JSON Schema for YAML editors (``part`` = a part file;
    None = decided from the file name), then the optional comment."""
    from .schema_export import modeline
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if part is None:
        part = path.name != MAIN_FILE and path.stem in PART_NAMES
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"# {modeline(part)}\n")
        if comment:
            stream.write(f"# {comment}\n")
        yaml.dump(document, stream, Dumper=_Dumper, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return path


def load_parts(split_dir: str | Path) -> dict[str, Any]:
    """Merged document of the part files found in ``split_dir`` (``{}`` when there are none)."""
    return merge_configs(*[read_yaml(path) for path in existing_parts(split_dir)])


def load_document(path: str | Path) -> dict[str, Any]:
    """Whole document behind ``path``.

    A plain single file is read as is.  For a directory, ``main.yaml`` or a part
    file, the parts are merged; ``main.yaml`` is used only when no part exists.
    Returns ``{}`` when nothing exists yet.
    """
    path = Path(path)
    if path.is_file() and not is_part_path(path):
        return read_yaml(path)
    main, split_dir = config_paths(path)
    if existing_parts(split_dir):
        return load_parts(split_dir)
    if main.is_file():
        return read_yaml(main)
    return {}


def save_parts(raw: dict[str, Any], split_dir: str | Path) -> list[Path]:
    """Write the seven part files of ``raw`` into ``split_dir``."""
    return [write_yaml(part_path(split_dir, name), part,
                       comment=f"eforecast configuration part: {PART_CONTENTS[name]}")
            for name, part in split_config(merge_configs(raw)).items()]


def save_document(raw: dict[str, Any], path: str | Path) -> tuple[Path, list[Path]]:
    """Write ``raw`` as one file and as the seven parts; returns ``(single file, part files)``."""
    main, split_dir = config_paths(path)
    document = merge_configs(raw)
    write_yaml(main, document,
               comment=f"whole eforecast configuration; the parts live in {split_dir.name}/")
    return main, save_parts(document, split_dir)


def save_merged(split_dir: str | Path, out: str | Path | None = None) -> Path:
    """Merge the parts of ``split_dir`` into ``main.yaml`` (or ``out``) after full validation."""
    document = load_parts(split_dir)
    ExperimentConfig.model_validate(document)
    target = Path(out) if out else Path(split_dir) / MAIN_FILE
    return write_yaml(target, document,
                      comment="merged from " + ", ".join(f"{name}.yaml" for name in PART_NAMES))


# ---- validation of partial documents ------------------------------------------
REQUIRED_SECTIONS: tuple[str, ...] = ("project", "folders", "data", "methods")


def missing_sections(document: dict[str, Any]) -> list[str]:
    return [section for section in REQUIRED_SECTIONS if not document.get(section)]


def validation_note(document: dict[str, Any]) -> str:
    """Empty when the whole document validates, otherwise a note for the status line."""
    from pydantic import ValidationError  # local import keeps the module light

    missing = missing_sections(document)
    if missing:
        return (" Only this part was validated; sections still missing: "
                + ", ".join(missing) + ".")
    try:
        ExperimentConfig.model_validate(document)
    except ValidationError as exc:
        return f" Full validation failed (other sections): {exc}"
    return ""
