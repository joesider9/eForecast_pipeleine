# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Neutral site-graph generation/commit readers; legacy folders remain readable."""
import hashlib
import json
from pathlib import Path
from contextvars import ContextVar
from contextlib import contextmanager

POINTER = 'site_graph_bundle.json'
GENERATIONS = '.graph-generations'
_retained = ContextVar('retained_graph_locations', default={})


def location_key(path):
    return hashlib.sha256(str(Path(path).absolute()).encode()).hexdigest()


@contextmanager
def retained_locations(locations):
    """Relocate verified bundle copies without rewriting their original producer records."""
    token = _retained.set(locations or {})
    try:
        yield
    finally:
        _retained.reset(token)


def retained_path(kind, path):
    return Path(_retained.get().get(kind, {}).get(location_key(path), path))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def publication_error(payload):
    stamp = payload.get('graph_publication')
    if stamp is None:
        return None
    if not isinstance(stamp, dict) or not all(isinstance(stamp.get(k), str) and stamp[k]
                                             for k in ('id', 'commit', 'logical_path')):
        return 'invalid graph publication identity'
    try:
        commit = json.loads(retained_path('commits', stamp['commit']).read_text())
    except FileNotFoundError:
        return 'graph group publication is not committed'
    except (OSError, ValueError):
        return 'graph group commit cannot be read'
    if (not isinstance(commit, dict) or type(commit.get('version')) is not int or commit['version'] != 1 or commit.get('id') != stamp['id']
            or not isinstance(commit.get('artifacts'), dict)
            or commit['artifacts'].get(stamp['logical_path']) != identity(payload)):
        return 'graph artifact differs from its group commit'
    return None


def bundle_root(base):
    base = retained_path('bundles', base)
    pointer = base / POINTER
    if not pointer.exists():
        return base
    record = json.loads(pointer.read_text())
    if not isinstance(record, dict) or type(record.get('version')) is not int or record['version'] != 1:
        raise ValueError(f'invalid site-graph pointer {pointer}')
    generation = record.get('generation')
    if not isinstance(generation, str) or not generation.isalnum():
        raise ValueError('invalid site-graph generation')
    target = base / GENERATIONS / generation / 'bundle'
    commit = base / GENERATIONS / generation / 'commit.json'
    if not commit.exists():
        fallback = record.get('fallback')
        if not isinstance(fallback, str) or not Path(fallback).is_absolute():
            raise ValueError('uncommitted site graph has no legacy/current fallback')
        previous = Path(fallback).resolve()
        if previous != base.resolve() and not previous.is_relative_to((base / GENERATIONS).resolve()):
            raise ValueError('site-graph fallback escapes its group')
        return previous
    if (target.is_symlink() or (base / GENERATIONS).is_symlink()
            or not target.resolve().is_relative_to((base / GENERATIONS).resolve()) or not target.is_dir()):
        raise ValueError(f'missing/unsafe site-graph generation {target}')
    return target
