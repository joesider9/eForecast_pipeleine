# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The JSON Schema of the configuration, generated from the pydantic models.

    python -m eforecast.config.schema_export            # writes schema/*.schema.json
    python -m eforecast.config.schema_export --check    # exit 1 when the files are stale

Two documents: ``experiment_config.schema.json`` for a whole case (``main.yaml`` or a single
``<case>.yaml``) and ``experiment_config_part.schema.json`` for the part files, where every
section is optional.  YAML editors pick them up through the ``yaml-language-server`` modeline that
``config_files`` writes at the top of every file it saves, or through ``.vscode/settings.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from eforecast.config.schema import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schema"
FULL_NAME = "experiment_config.schema.json"
PART_NAME = "experiment_config_part.schema.json"
SCHEMA_ID = "https://github.com/joesider9/ProjectPipelineTest/schema/"


def full_schema() -> dict[str, Any]:
    schema = ExperimentConfig.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = SCHEMA_ID + FULL_NAME
    schema["title"] = "eforecast case"
    schema["description"] = "A complete eforecast case (main.yaml or <case>.yaml)."
    return schema


def part_schema() -> dict[str, Any]:
    """The same properties, nothing required: a part file holds some sections of a case."""
    schema = copy.deepcopy(full_schema())
    schema["$id"] = SCHEMA_ID + PART_NAME
    schema["title"] = "eforecast case part"
    schema["description"] = "One part file of an eforecast case (system, project, input, methods, experiments, clustering, method_families)."
    _drop_required(schema)      # a part file may hold any subset of a section (env keys live in system.yaml, ...)
    return schema


def _drop_required(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("required", None)
        for value in node.values():
            _drop_required(value)
    elif isinstance(node, list):
        for item in node:
            _drop_required(item)


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def schema_path(part: bool = False) -> Path:
    return SCHEMA_DIR / (PART_NAME if part else FULL_NAME)


def modeline(part: bool = False) -> str:
    """The comment YAML language servers read to find the schema of a file."""
    return f"yaml-language-server: $schema={schema_path(part).as_uri()}"


def export(directory: str | Path = SCHEMA_DIR) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, schema in ((FULL_NAME, full_schema()), (PART_NAME, part_schema())):
        path = directory / name
        path.write_text(render(schema), encoding="utf-8")
        written.append(path)
    return written


def stale(directory: str | Path = SCHEMA_DIR) -> list[str]:
    """Names of the schema files that differ from the models (or are missing)."""
    directory = Path(directory)
    out = []
    for name, schema in ((FULL_NAME, full_schema()), (PART_NAME, part_schema())):
        path = directory / name
        if not path.exists() or path.read_text(encoding="utf-8") != render(schema):
            out.append(name)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only report whether the committed files are current")
    parser.add_argument("--dir", default=str(SCHEMA_DIR))
    args = parser.parse_args(argv)
    if args.check:
        names = stale(args.dir)
        if names:
            print(f"stale schema files: {', '.join(names)}; run python -m eforecast.config.schema_export")
            return 1
        print("schema files are current")
        return 0
    for path in export(args.dir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
