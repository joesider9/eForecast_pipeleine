# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Rewrite the ``experiments`` of a case in the current layer form.

Every layer parameter -- the size included -- lives under ``params`` and carries a tag::

    - type: conv                       - type: conv
      size: 3                            params:
      choices: null            ->          size: {tag: fixed, value: 3}
      range: null                          filters: {tag: choice, value: [12, 24]}
      params: {}                           pool: {tag: range, value: [1, 3]}
      name: null                         name: null
      enabled: true                      enabled: true

The loader reads both shapes, so this is a tidy-up rather than a requirement; running it makes the
file say what the editor will write the next time it saves.

    python -m eforecast.config.migrate_layer_params <case path> [...]   # rewrites in place
    python -m eforecast.config.migrate_layer_params --check <case path> # says what would change
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from eforecast.config.config_files import read_yaml, write_yaml
from eforecast.config.schema import NetworkExperimentConfig


def migrated(experiments: dict[str, Any]) -> dict[str, Any]:
    """``experiments`` with every layer in the canonical form."""
    out: dict[str, Any] = {}
    for tag, branches in experiments.items():
        model = NetworkExperimentConfig.model_validate(branches)
        out[tag] = {branch: [layer.model_dump() for layer in layers] for branch, layers in model.root.items()}
    return out


def migrate_file(path: Path, *, check: bool = False, backup: bool = True) -> bool:
    """Rewrite one YAML holding an ``experiments`` block; ``True`` when it changed."""
    document = read_yaml(path)
    experiments = document.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        return False
    updated = migrated(experiments)
    if updated == experiments:
        return False
    if check:
        return True
    if backup:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    document["experiments"] = updated
    comment = None
    for line in path.read_text(encoding="utf-8").splitlines()[:3]:
        if line.startswith("# eforecast configuration part:") or line.startswith("# whole eforecast"):
            comment = line[2:]
    write_yaml(path, document, comment=comment)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="case YAMLs, part files or directories to search for experiments.yaml")
    parser.add_argument("--check", action="store_true", help="report what would change and write nothing")
    parser.add_argument("--no-backup", action="store_true", help="do not keep a .bak copy")
    args = parser.parse_args(argv)

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("experiments.yaml")))
        else:
            files.append(path)

    changed = 0
    for path in files:
        if not path.is_file():
            print(f"skipped {path}: not a file", file=sys.stderr)
            continue
        try:
            if migrate_file(path, check=args.check, backup=not args.no_backup):
                changed += 1
                print(f"{'would rewrite' if args.check else 'rewrote'} {path}")
            else:
                print(f"unchanged {path}")
        except (yaml.YAMLError, ValueError) as error:
            print(f"failed {path}: {error}", file=sys.stderr)
            return 2
    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
