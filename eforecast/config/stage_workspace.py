# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A private view of a case's model tree, for a worker that must not write into it.

A model worker -- the image encoder, the site-graph builder, the foundation adapter -- builds its
result in its own folder and the host copies the verified files into the case afterwards. This
gives such a worker the case's layout with the large inputs linked rather than copied, and maps
paths between the two. It is not a stage boundary: stages write in place (``direct_stage``).
"""
from copy import deepcopy
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import shutil
import uuid

from eforecast.common_utils import lineage
from eforecast.common_utils.stage_context import current_stage as _current
from eforecast.datasets.data_preprocessing.partition_report import atomic_json
from eforecast.folder_lock import folder_lock as file_critical_section
from . import stage_artifacts as artifacts

log = logging.getLogger(__name__)


def current():
    return _current.get()


class Observation(dict):
    def __init__(self, workspace):
        super().__init__(kind='success', detail='stage outputs await verification')
        self.workspace = workspace
        self.static_data = workspace.static

    def paths(self, value, *, canonical=False):
        return self.workspace.rebase(value, canonical=canonical)


class Workspace:
    def __init__(self, static, plan, key, *, attempt_uid=None):
        self.original = static
        self.static = deepcopy(static)
        self.plan, self.key = deepcopy(plan), key
        self.node = next(n for n in self.plan['nodes'] if n['key'] == key)
        self.base = Path(static['path_model']).absolute()
        self.uid = attempt_uid or uuid.uuid4().hex
        self.root = self.view = None            # the worker names its own folder
        self.maps = [(self.base, self.view)]
        self.writes, self.trees = self._writes()
        self.copied = {}
        self.rebuilding, self.replacing = False, set()

    def _writes(self):
        key, base = self.key, self.base
        files = {'transformers.pickle', 'DATA/day_light_hours.csv', 'DATA/daylight_images.csv',
                 'DATA/clear_sky.csv', 'partition_report.json',
                 'partition_report.json.lock', 'Results/partition_clusters.csv'}
        if self.original.get('is_Fuzzy') and key in {'clustering', 'features', 'predict', 'select', 'combine', 'calibrate', 'evaluate'}:
            files.add('clusters.pickle')
        trees = set()
        datasets = {'target': 'dataset_target_data.csv', 'row': 'dataset_row_data.pickle',
                    'lstm': 'dataset_lstm_data.pickle', 'nwp': 'dataset_nwps.pickle', 'images': 'dataset_images.pickle'}
        if key in datasets:
            files.add('DATA/' + datasets[key])
            if key == 'row':
                files.add('DATA/dataset_row_data.csv')
        if key in {'split', 'clustering'}:
            files.update({'DATA/cv_mask.pickle', 'partition.json', 'global/cv_mask.pickle',
                          'global/feature_selectors.partition.json'})
        if key == 'preprocess':
            files.update({'scalers.pickle', 'scalers_provenance.json', 'scalers_fingerprint.json', 'imputers.pickle'})
        if key == 'clustering':
            trees.add('cluster_organizer')
        if key == 'features':
            files.add('feature_selectors.pickle')
            for scope in artifacts._scopes(self.original):
                relative = scope.absolute().relative_to(base)
                files.update(str(relative / name) for name in ('feature_selectors.pickle', 'feature_selectors.partition.json'))
        if key in {'predict', 'select', 'combine', 'calibrate', 'evaluate'}:
            trees.add('Results')
            if key == 'combine':
                trees.add('combine_models')
            files.add('combined_eval.pickle')
            for scope in artifacts._scopes(self.original):
                relative = scope.absolute().relative_to(base)
                if key == 'combine':
                    trees.add(str(relative / 'combine'))
                for method in scope.iterdir() if scope.is_dir() else []:
                    if not method.is_dir():
                        continue
                    from eforecast.common_utils.trial_layout import method_directory
                    method = method_directory(self.original, scope, method.name)
                    rel_method = method.absolute().relative_to(base)
                    files.add(str(rel_method / 'selected_trials.json'))
                    for trial in method.glob('test_*'):
                        for name in ('predictions_eval.pickle', 'predictions_eval.pickle.key.json',
                                     'quantiles_eval.pickle', 'quantiles_eval.pickle.key.json'):
                            files.add(str(trial.absolute().relative_to(base) / name))
        # Content identity sidecars are part of a producer's own writes, not aliases to old claims.
        files |= {name + lineage.SIDECAR for name in tuple(files) if not name.endswith('.lock')}
        return {Path(p) for p in files}, {Path(p) for p in trees}


    def owned(self, relative):
        return relative in self.writes or any(relative == p or p in relative.parents for p in self.trees)

    def _copy(self, source, target, relative=Path('.')):
        """Build only writable ancestors; unrelated model trees remain a single read link."""
        if source.is_dir():
            # Existing trial payloads under a replaced cluster are immutable inputs, not copied
            # checkpoints. Deleting their private parent removes links only, never the old models.
            if relative.name.startswith('test_') and self.key == 'clustering':
                target.symlink_to(source, target_is_directory=True)
                return
            needed = (relative == Path('.') or self.owned(relative) or
                      any(relative in p.parents for p in self.writes | self.trees))
            if not needed:
                target.symlink_to(source, target_is_directory=True)
                return
            target.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                if relative == Path('.') and child.name == artifacts.DIRECTORY:
                    continue
                self._copy(child, target / child.name, relative / child.name)
        elif relative in self.replacing:
            return                       # authorized rebuild: this output is not carried forward
        elif self.owned(relative):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            self.copied[source] = target
        else:
            target.symlink_to(source)

    def rebase(self, value, *, canonical=False):
        pairs = [(str(b), str(a)) if canonical else (str(a), str(b)) for a, b in self.maps]
        if isinstance(value, str):
            if not canonical:
                for key in ('path_model', 'path_group_nwp', 'path_image'):
                    raw = self.original.get(key)
                    if isinstance(raw, str) and not os.path.isabs(raw) and (value == raw or value.startswith(raw + os.sep)):
                        value = str(Path(value).absolute())
                        break
            for source, target in sorted(pairs, key=lambda pair: len(pair[0]), reverse=True):
                if value == source or value.startswith(source + os.sep):
                    return target + value[len(source):]
            return value
        if isinstance(value, dict):
            return {self.rebase(k, canonical=canonical): self.rebase(v, canonical=canonical) for k, v in value.items()}
        if isinstance(value, list):
            return [self.rebase(v, canonical=canonical) for v in value]
        if isinstance(value, tuple):
            return tuple(self.rebase(v, canonical=canonical) for v in value)
        return value


    def _canonicalize_metadata(self):
        import joblib
        for base, tree in self.maps:
            for path in self._files(tree):
                original = base / path.relative_to(tree)
                if original.is_file() and artifacts.content_digest(path) == artifacts.content_digest(original):
                    continue  # unrelated retained metadata need not be parsed or rewritten
                if path.suffix == '.json':
                    value = json.loads(path.read_text())
                    rebased = self.rebase(value, canonical=True)
                    if rebased != value:
                        atomic_json(path, rebased)
                elif path.name == 'clusters.pickle':
                    value = joblib.load(path)
                    joblib.dump(self.rebase(value, canonical=True), path)

    @staticmethod
    def _files(root):
        # Path.rglob follows no directory symlinks on supported Python versions. Explicit walk
        # also refuses leaf symlinks so an immutable input is never moved into publication.
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not Path(directory, d).is_symlink()]
            for name in files:
                path = Path(directory, name)
                if not path.is_symlink():
                    yield path


