# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Read additive fuzzy generations while keeping all legacy RBF folders readable."""
import json
from pathlib import Path

POINTER = 'fuzzy_bundle.json'
GENERATIONS = '.fuzzy-generations'


def bundle_root(base):
    base = Path(base)
    pointer = base / POINTER
    if not pointer.exists():
        return base
    record = json.loads(pointer.read_text())
    generation = record.get('generation')
    if record.get('version') != 1 or not isinstance(generation, str) or not generation.isalnum():
        raise ValueError(f'invalid fuzzy generation pointer {pointer}')
    target = base / GENERATIONS / generation
    if target.is_symlink() or not target.is_dir():
        raise ValueError(f'missing/unsafe fuzzy generation {target}')
    return target


def current_files(base):
    base = Path(base)
    root = bundle_root(base)
    if root == base:
        return []
    record = json.loads((base / POINTER).read_text())
    files = record.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('fuzzy generation has no named payload')
    for name, item in files.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not isinstance(item, dict):
            raise ValueError('invalid fuzzy bundle payload record')
    return [base / POINTER, root / 'bundle_manifest.json', *[root / name for name in files]]
