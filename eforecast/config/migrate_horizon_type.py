# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Rename the horizon-type folders of a model store, and the value inside case YAMLs.

The horizon type is a folder in the model store
(``<sys_folder>/<owner>/<group>/<type>/<project>/<horizon type>/model_ver<n>``), so the rename

    day-ahead -> day_ahead     intra-ahead  -> intraday
    multi-output -> short_term  multi-ahead -> long_term

moves every trained model.  Nothing breaks without this: ``create_paths`` keeps using a folder with
the former name when the new one does not exist.  Run this to finish the job, so the folders say
what the configuration says.

    python -m eforecast.config.migrate_horizon_type --check <path>...   # list what would move
    python -m eforecast.config.migrate_horizon_type <path>...           # rename

A path may be a model store root (searched for horizon-type folders at any depth), a case YAML or a
case directory (the ``horizon_type`` value is rewritten).  Folders are renamed, never copied, so a
store on a full disk migrates fine; a rename is refused when the destination already exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eforecast.config.config_files import read_yaml, write_yaml
from eforecast.config.schema import LEGACY_HORIZON_TYPES, HorizonType

# the former folder name -> the name it takes now
RENAMES: dict[str, str] = {old: HorizonType[member].value for old, member in LEGACY_HORIZON_TYPES.items()}


def folders_to_rename(root: Path) -> list[tuple[Path, Path]]:
    """``(from, to)`` for every horizon-type folder under ``root``, deepest first."""
    pairs: list[tuple[Path, Path]] = []
    for old, new in RENAMES.items():
        for path in root.rglob(old):
            if path.is_dir() and not path.is_symlink():
                pairs.append((path, path.with_name(new)))
    return sorted(pairs, key=lambda pair: len(pair[0].parts), reverse=True)


def rename_folders(root: Path, *, check: bool = False) -> tuple[int, int]:
    """Rename the horizon-type folders under ``root``; returns ``(renamed, skipped)``."""
    renamed = skipped = 0
    for source, target in folders_to_rename(root):
        if target.exists():
            print(f"skipped {source}: {target.name} already exists next to it", file=sys.stderr)
            skipped += 1
            continue
        print(f"{'would rename' if check else 'renamed'} {source} -> {target.name}")
        if not check:
            source.rename(target)
        renamed += 1
    return renamed, skipped


def rewrite_case(path: Path, *, check: bool = False) -> bool:
    """Rewrite ``project.horizon_type`` of one case YAML; ``True`` when it changed."""
    document = read_yaml(path)
    project = document.get("project")
    if not isinstance(project, dict) or project.get("horizon_type") not in RENAMES:
        return False
    old = project["horizon_type"]
    if check:
        print(f"would rewrite {path}: {old} -> {RENAMES[old]}")
        return True
    project["horizon_type"] = RENAMES[old]
    comment = next((line[2:] for line in path.read_text(encoding="utf-8").splitlines()[:3]
                    if line.startswith("# eforecast configuration part:") or line.startswith("# whole eforecast")), None)
    write_yaml(path, document, comment=comment)
    print(f"rewrote {path}: {old} -> {RENAMES[old]}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="model store roots, case YAMLs or case directories")
    parser.add_argument("--check", action="store_true", help="report what would change and write nothing")
    args = parser.parse_args(argv)

    changed = 0
    for raw in args.paths:
        path = Path(raw).expanduser()
        if not path.exists():
            print(f"skipped {path}: does not exist", file=sys.stderr)
            continue
        if path.is_file():
            changed += rewrite_case(path, check=args.check)
            continue
        renamed, _ = rename_folders(path, check=args.check)
        changed += renamed
        for case in sorted(path.rglob("project.yaml")) + sorted(path.rglob("main.yaml")):
            changed += rewrite_case(case, check=args.check)
    if not changed:
        print("nothing to change")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
