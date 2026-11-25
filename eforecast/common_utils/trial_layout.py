# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Neutral, additive trial generations shared by writers and inference readers.

No marker means the historical layout. New revisions never write in that layout.
Only the publisher of usable work switches the per-method current marker.
"""
import json
import logging
import os
from pathlib import Path
import re
import uuid

from . import lineage

log = logging.getLogger(__name__)
MARKER = 'trial_generation.json'
GENERATIONS = '.generations'
CLUSTERS = 'clusters.pickle'
CLUSTER_ROOT = 'cluster_organizer'
_told = set()


class ForeignTree(RuntimeError):
    """A model tree whose cluster folders are those of another tree, and that has none of its own."""


def cluster_folders(static):
    """``{cluster: folder}`` of a clustered case, every folder inside the case's own model tree.

    ``clusters.pickle`` holds the folders as absolute paths. A model tree that was copied, or
    moved by hand, therefore still names the folders of the tree it came from, and whatever
    predicts, selects or trains "in the copy" writes into the original: on 2026-11-14 a
    measurement on a copy of a finished case rewrote 448 prediction files of the case itself.

    So a folder outside ``path_model`` is read as this tree's own folder of the same name --
    ``cluster_organizer/<method>/<rule>`` under ``path_model`` -- and said once. A tree that has
    no such folder is refused: there is nothing of its own to work in. The file is left as it
    is; its content is part of what the trained models are bound to, and training refreshes it
    by itself (``ClusterOrganizer.update_cluster_folders``).
    """
    import joblib
    root = Path(static['path_model']).absolute()
    real = root.resolve()
    own, foreign = {}, []
    for name, folder in joblib.load(root / CLUSTERS).items():
        folder = Path(folder)
        if folder.absolute().is_relative_to(root) or folder.resolve().is_relative_to(real):
            own[name] = str(folder)
            continue
        parts = folder.parts
        at = len(parts) - 1 - parts[::-1].index(CLUSTER_ROOT) if CLUSTER_ROOT in parts else None
        local = root.joinpath(*parts[at:]) if at is not None else None
        if local is None or not local.is_dir():
            raise ForeignTree(
                f'{root / CLUSTERS} names the cluster {name} at {folder}, outside this model tree, and the tree '
                f'has no folder of its own for it{f" ({local})" if local is not None else ""}. It was copied or moved '
                'without its cluster folders; nothing is read from or written into another case. Copy the '
                f'{CLUSTER_ROOT} folder as well, or run the clustering stage for this case.')
        own[name] = str(local)
        foreign.append(str(folder))
    if foreign and str(root) not in _told:
        _told.add(str(root))
        log.warning('%s names %d cluster folder(s) of another model tree (%s ...); this tree was copied or moved, '
                    'and its own folders under %s are used', root / CLUSTERS, len(foreign),
                    foreign[0], root / CLUSTER_ROOT)
    return own


def revision(static):
    value = static.get('study_revision') or os.environ.get('EFORECAST_STUDY_REVISION')
    if not value and (static.get('study_action') or os.environ.get('EFORECAST_STUDY_ACTION')) == 'new':
        value = 'new'
    if not value or value == 'continue':
        return None
    if value in ('new', 'auto'):
        # Action/scope/resource controls do not change the learned inputs.
        data = {k: v for k, v in static.items() if k not in ('study_action', 'study_revision')}
        return lineage.comparability_id(data)[:8]
    return re.sub(r'[^A-Za-z0-9_.-]', '', str(value))[:16].strip('.') or lineage.dependency_id(value)[:16]


def generation(static, scope, method):
    if method == 'GRAPH':
        return None  # GRAPH publishes through its separate immutable group bundle contract.
    key = str(Path(scope).absolute() / method)
    pinned = static.get('_trial_generations', {})
    if key in pinned:
        return pinned[key]
    explicit = revision(static)
    if explicit:
        return explicit
    file = Path(scope) / method / MARKER
    if not file.exists():
        return None
    record = json.loads(file.read_text())
    value = record.get('generation') if isinstance(record, dict) else None
    if not isinstance(record, dict) or record.get('version') != 1 or not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}', value):
        raise ValueError(f'invalid trial generation marker: {file}')
    return value


def pin(static, scope, method, cluster=None):
    """Freeze a unit's layout before opening its history; never mutate a saved case."""
    key = str(Path(scope).absolute() / method)
    return {**static, '_trial_generations': {**static.get('_trial_generations', {}),
                                            key: generation(static, scope, method)},
            '_trial_scopes': {**static.get('_trial_scopes', {}),
                              **({cluster: str(scope)} if cluster is not None else {})}}


def method_directory(static, scope, method):
    base = Path(scope) / method
    selected = generation(static, scope, method)
    return base / GENERATIONS / selected if selected else base


def trial_directory(static, scope, method, number):
    return method_directory(static, scope, method) / f'test_{number}'


def results_file(static, scope, cluster, method):
    directory = method_directory(static, scope, method)
    # Retain the legacy table path; generated histories keep their tables apart.
    return (directory if directory.parent.name == GENERATIONS else Path(scope)) / f'results_{cluster}_{method}.csv'


def activate(path):
    """Select the generation this trial belongs to; called once its payload is validated and in place."""
    path = Path(path)
    if path.parent.parent.name != GENERATIONS:
        return
    folder = path.parent.parent.parent
    marker = folder / MARKER
    expected = {'version': 1, 'generation': path.parent.name}
    if marker.is_file():
        try:
            if json.loads(marker.read_text()) == expected:
                return
        except ValueError:
            pass  # an explicitly requested, verified new publication can repair its marker
    temporary = folder / f'.{MARKER}.{uuid.uuid4().hex}.tmp'
    try:
        with temporary.open('x') as stream:
            json.dump({'version': 1, 'generation': path.parent.name}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
        descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def scope_directory(static, cluster):
    if cluster in static.get('_trial_scopes', {}):
        return Path(static['_trial_scopes'][cluster])
    if cluster == 'global':
        return Path(static.get('path_model', '')) / 'global'
    mapping = Path(static.get('path_model', '')) / CLUSTERS
    return Path(cluster_folders(static)[cluster]) if mapping.is_file() else Path(static.get('path_model', '')) / cluster
