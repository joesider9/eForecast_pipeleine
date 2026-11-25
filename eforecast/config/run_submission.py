# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Submission input envelopes: cheap case-file capture, deferred A07 materialization.

Case bytes live in the accepted database record, so preparing never rereads edited YAML.
External experiment sources are guarded until A07 materializes them; a changed dependency
fails preparation instead of silently substituting a new experiment. Datasets/libraries are
outside this configuration envelope, as in A07.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import os
from pathlib import Path
import sys
import sysconfig

from .config_files import config_paths, existing_parts
from .service import ServiceError

EXECUTION_ENV = ('EFORECAST_OPTUNA_STORAGE', 'EFORECAST_METRICS_PORT', 'EFORECAST_WORKER_IMAGE',
    'EFORECAST_STUDY_ACTION', 'EFORECAST_STUDY_REVISION',
    'EFORECAST_MODEL_CACHE_MODE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
    'EFORECAST_WORKER_FOLDERS', 'EFORECAST_WORKER_MOUNTS', 'EFORECAST_TRIAL_CMD', 'CUDA_VISIBLE_DEVICES',
    'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONPATH')


def environment(values):
    result = dict(os.environ)
    for key, value in values.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive(source):
    single, parts = config_paths(source)
    files = list(existing_parts(parts))
    if single.is_file() and single not in files:
        files.append(single)
    return {('main.yaml' if path == single else path.name):
            base64.b64encode(path.read_bytes()).decode('ascii') for path in files}


def restore(files, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    for name, payload in files.items():
        if Path(name).name != name or name in {'.', '..'}:
            raise ServiceError('invalid accepted configuration filename')
        (folder / name).write_bytes(base64.b64decode(payload, validate=True))


def _module(name, roots):
    """Find source without importing a package in the API process."""
    relative = Path(*name.split('.'))
    for root in roots:
        candidates = (root / relative.with_suffix('.py'), root / relative / '__init__.py') if name else (root / '__init__.py',)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


def experiment_sources(document, source):
    spec = document.get('experiments_module')
    if not spec:
        return {}
    parent = config_paths(source)[0].resolve().parent
    roots = [parent, *parent.parents, *(Path(p or os.getcwd()) for p in sys.path)]
    if str(spec).endswith('.py'):
        named = Path(spec)
        first = named.resolve() if named.is_absolute() else next(
            ((root / named).resolve() for root in [parent, *parent.parents] if (root / named).is_file()), None)
    else:
        first = _module(str(spec), roots)
    if first is None:
        # A07 retains its explicit degraded route for configurations that do not normalize.
        return {}
    pending, seen = [first], {}
    library = Path(__file__).resolve().parent.parent
    standard_library = Path(sysconfig.get_path('stdlib'))
    while pending:
        path = pending.pop()
        if str(path) in seen or (path != first and path.is_relative_to(library)):
            continue
        payload = path.read_bytes()
        seen[str(path)] = digest(payload)
        ancestor = path.parent
        while (ancestor / '__init__.py').is_file():
            pending.append(ancestor / '__init__.py')
            ancestor = ancestor.parent
        try:
            tree = ast.parse(payload)
        except (SyntaxError, ValueError):
            continue
        search = [path.parent, *roots]
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    anchor = path.parent
                    for _ in range(node.level - 1):
                        anchor = anchor.parent
                    relative = _module(node.module or '', [anchor])
                    if relative:
                        pending.append(relative)
                    prefix = f'{node.module}.' if node.module else ''
                    for alias in node.names:
                        relative = _module(prefix + alias.name, [anchor])
                        if relative:
                            pending.append(relative)
                names = [node.module] if node.module else []
                names += [f'{node.module}.{alias.name}' if node.module else alias.name for alias in node.names]
            for name in names:
                # Installed dependencies and standard-library source are deliberately not
                # configuration snapshots. Capture only workspace/repository Python code.
                found = _module(name, search)
                if found and not any(part in {'site-packages', 'dist-packages', 'lib-dynload'} for part in found.parts):
                    if found.is_relative_to(Path(sys.base_prefix)) or found.is_relative_to(standard_library):
                        continue
                    pending.append(found)
    return seen


def verify_sources(sources):
    for path, expected in sources.items():
        if digest(Path(path).read_bytes()) != expected:
            raise ServiceError(f'experiment source changed after acceptance: {path}; submit a new request')
