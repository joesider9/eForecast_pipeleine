# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Content receipts for stage boundaries. No receipt means legacy, never verified history.

Receipts are published last. A publishing marker survives a failed multi-file promotion and
prevents readers from falling back to file presence. Expensive content checks run at stage
boundaries, not on scheduler ticks or in every model trial.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import logging
import os
import pickle
import re
import stat
import time
from pathlib import Path

from eforecast.common_utils import artifact, lineage
from eforecast.datasets.data_preprocessing.partition_report import atomic_json

VERSION = 1
DIRECTORY = '.stage-artifacts'
#: External inputs a run may continue without. Raw weather off a provider feed belongs here: an
#: empty file is a missing hour, and refusing does not make the hour appear. Nothing produced by
#: a stage of this pipeline belongs here -- an image encoder that is absent means the stage that
#: trains it has not run, and continuing cannot produce it.
RECOVERABLE = frozenset({'external:nwp_raw'})
log = logging.getLogger(__name__)
_digests = ContextVar('stage_content_digests', default=None)
#: Digests of files this process has hashed, or that a receipt observed, for as long as each
#: stays the same file: (path, inode, size, modification time, change time) -> sha256.
_known = {}
KNOWN_LIMIT = 400_000
#: How long a file must have been at rest before its digest is remembered. A write in the same
#: clock tick as the one before it leaves every timestamp where it was, so a file that changed
#: a moment ago cannot be told from the file it is about to become.
SETTLED_NS = 3_000_000_000
#: ``EFORECAST_REHASH=1`` reads every payload again at every check, as every run used to.
REHASH = 'EFORECAST_REHASH'
#: Receipts already taken into ``_known``: {receipt path: what it was when it was read}.
_remembered = {}
#: Gaps in an external input that this process has already described.
_described = set()
#: Identities of payloads whose shape was checked: by this process, or by the stage whose
#: receipt records them as its output (``output_records`` checks before it records).
_valid = set()
#: Files of one bundle hashed side by side. sha256 leaves the interpreter lock while it works,
#: so eight threads read eight times as fast from memory; from a cold disk the disk decides.
HASHERS = 8
#: A bundle of at least this many payloads is shape-checked by the stage's reader pool.
POOLED_FROM = 32


class IncompleteStage(ValueError):
    pass


class ConfigurationChanged(IncompleteStage):
    """A stage whose output is whole, and was built from settings the case no longer has.

    Its own kind of rejection because it is the one a person caused by editing the case, and
    the one they are asked about before a run replaces what the stage made (``stage_resume``).
    """
    cause = 'configuration'


@contextmanager
def verification_boundary():
    token = _digests.set({}) if _digests.get() is None else None
    try:
        yield
    finally:
        if token is not None:
            _digests.reset(token)


def boundary(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with verification_boundary():
            return function(*args, **kwargs)
    return wrapped


def _identity(path, observation):
    """What makes a file the same file: where it is, its inode, size, and both its clocks.

    The change time is the one nothing short of the system clock can put back, so content
    cannot move under an identity that still matches. The device number is left out on purpose:
    a disk can come back from a reboot under another one, and every digest would be lost for it.
    """
    return (_absolute(path), *observation[1:])


def _absolute(path):
    """``str(Path(path).absolute())`` without building a ``Path`` for what already is one.

    A run names every raw provider file several times over; for a three-year case that is the
    better part of a million ``Path`` objects made only to be turned back into the string they
    came from.
    """
    text = os.fspath(path)
    if (isinstance(text, str) and text.startswith('/') and '//' not in text and '/./' not in text
            and not text.endswith(('/', '/.'))):
        return text
    return str(Path(text).absolute())


def _keep(identity, value):
    if len(_known) >= KNOWN_LIMIT:                  # a long-lived server: drop the oldest half
        for stale in list(_known)[:KNOWN_LIMIT // 2]:
            del _known[stale]
    _known[identity] = value


def content_digest(path):
    """The sha256 of a file, read once for as long as it stays the same file.

    A case's weather is tens of gigabytes that no run changes, and every stage names it: as
    its own output, as the next stage's input, and again when the next run asks what it may
    reuse. Reading it each time is what made a run that had nothing to do take minutes on one
    core. So a digest is kept against the file's identity -- within a boundary, as before; for
    the life of the process once the file has been at rest; and across runs through the
    receipts, which already record the identity each digest was taken from (``_remember``).
    A file that was written, replaced or merely touched has another identity and is read again.
    """
    before = _signature(path)
    cache = _digests.get()
    if cache is not None and before in cache:
        return cache[before]
    value = None if os.environ.get(REHASH) == '1' else _known.get(_identity(path, before))
    if value is None:
        value = _read_digest(path, before)
    if cache is not None:
        cache[before] = value
    return value


def _signature(path):
    value = os.stat(path)
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_digest(path, before):
    """Hash ``path``, which had the signature ``before``; remembered when the file was at rest.

    The file must be the same file from the first byte read to the last. A sync client that
    marks it as uploaded meanwhile moves its change time and not its content: that is told
    apart from a file that was written to (``file_stamps``), and the digest is then true of
    the bytes but is not kept, because the identity it would be kept under is gone.
    """
    from eforecast.common_utils import file_stamps
    started = time.time_ns()
    first = file_stamps.stamp(path)
    value = artifact.digest(path)
    last = file_stamps.stamp(path)
    if tuple(first[:4]) != tuple(before[:4]) or not file_stamps.same(first, last):
        raise IncompleteStage(f'payload changed while hashing: {path}')
    if tuple(last[:5]) == tuple(before) and started - max(before[3], before[4]) >= SETTLED_NS:
        _keep(_identity(path, before), value)
    return value


def prime(paths, what='payloads'):
    """Hash the files of a bundle side by side, so recording them one by one finds each ready.

    A bundle is often a thousand daily weather files or the raw provider files behind them. One
    after another they are read at the speed of one core; nothing about them depends on order.
    Whatever goes wrong with a file here is left for the caller's own read of it to report.
    """
    from concurrent.futures import ThreadPoolExecutor
    from eforecast.bounded_pools import cancel_check
    cache, rehash = _digests.get(), os.environ.get(REHASH) == '1'
    todo, size = [], 0
    for path in dict.fromkeys(str(Path(p).absolute()) for p in paths):
        try:
            before = _signature(path)
        except OSError:
            continue
        if (cache is not None and before in cache) or (not rehash and _identity(path, before) in _known):
            continue
        todo.append((path, before))
        size += before[2]
    if len(todo) < 2:
        return
    loud = size >= 2 ** 30
    if loud:
        log.info('reading %d %s, %.1f GB, to record their content', len(todo), what, size / 2 ** 30)

    def read(item):
        try:
            return _read_digest(*item)
        except (OSError, IncompleteStage):
            return None
    started, check = time.monotonic(), cancel_check.get() or (lambda: None)
    pool = ThreadPoolExecutor(max_workers=min(HASHERS, len(todo)), thread_name_prefix='digest')
    try:
        for (path, before), value in zip(todo, pool.map(read, todo)):
            check()
            if value is not None and cache is not None:
                cache[before] = value
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if loud:
        took = max(time.monotonic() - started, 1e-3)
        log.info('read %.1f GB in %.0f s (%.0f MB/s)', size / 2 ** 30, took, size / 2 ** 20 / took)


def _remember(receipt, value):
    """Take a receipt's word for every file it observed that nothing has touched since.

    A receipt holds, for each input and output, the digest and the identity of the file it was
    taken from. Where a file still has that identity, the digest is its digest and reading the
    file again can only confirm it. Records the receipt wrote within ``SETTLED_NS`` of the
    file's last change are passed over; so are records of receipts older than the identity.
    """
    try:
        stamp = os.stat(receipt)
    except OSError:
        return
    published, seen = stamp.st_mtime_ns, (stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, value.get('id'))
    if _remembered.get(str(receipt)) == seen and os.environ.get(REHASH) != '1':
        return                      # this very receipt has been taken in already
    _remembered[str(receipt)] = seen
    outputs = list((value.get('outputs') or {}).values()) if isinstance(value.get('outputs'), dict) else []
    bundles = [(records, True) for records in outputs]
    if isinstance(value.get('inputs'), dict):
        bundles += [(bundle.get('files'), False) for bundle in value['inputs'].values() if isinstance(bundle, dict)]
    for records, produced in bundles:
        for record in records if isinstance(records, list) else ():
            if not isinstance(record, dict):
                continue
            observation, digest, path = record.get('observation'), record.get('sha256'), record.get('path')
            if (isinstance(path, str) and isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
                    and isinstance(observation, list) and len(observation) == 5
                    and all(type(item) is int for item in observation)
                    and published - max(observation[3], observation[4]) >= SETTLED_NS):
                _keep(_identity(path, observation), digest)
                if produced:        # a stage checks the shape of an output before it records it
                    _vouch(_identity(path, observation))


def directory(static):
    return Path(static['path_model']) / DIRECTORY


def receipt_path(static, key):
    # Stage names are catalogue keys, never arbitrary filenames supplied by a worker.
    from .pipeline_stages import ALL_DATASET_STAGES, COMBINE_STAGES
    if key not in {*ALL_DATASET_STAGES, *COMBINE_STAGES, 'training'}:
        raise ValueError('unknown receipt stage')
    return directory(static) / 'receipts' / f'{key}.json'


RUNNING = '.running'
PREVIOUS = '.previous'


def running_marker(static, key):
    """The file a stage keeps while it works. It is removed only when the stage has finished."""
    return directory(static) / f'{key}{RUNNING}'


def finished_despite_marker(static, key):
    """Whether the marker was left by a run that had already recorded its receipt.

    The receipt is written before the marker is removed, so a run killed between the two leaves
    both. That stage is finished: the receipt names the attempt the marker names.
    """
    try:
        attempt = json.loads(running_marker(static, key).read_text()).get('attempt_uid')
        receipt = json.loads(receipt_path(static, key).read_text())
    except (OSError, ValueError, AttributeError):
        return False
    return bool(attempt) and isinstance(receipt, dict) and receipt.get('state') in {'complete', 'legacy'} \
        and (receipt.get('producer') or {}).get('attempt_uid') == attempt


def superseded_path(static, key):
    """Where the receipt that a publication is replacing is kept while it is replaced.

    The publishing marker is written over the canonical receipt, so an interruption used to
    destroy the only description of the bundle that is still on disk. Keeping the previous one
    beside it lets A09-08 decide afterwards whether the promotion had already changed anything.
    """
    return directory(static) / 'receipts' / f'{key}.superseded.json'


#: What every dataset stage is built from, and what each of the model-input stages adds to it.
COMMON_SETTINGS = ('type', 'horizon_type', 'horizon', 'ts_resolution', 'target_variable', 'variables',
                   'coord', 'local_timezone', 'site_timezone', 'evaluation_start', 'Evaluation_start',
                   'remove_night_hours', 'rated', 'NWP', 'use_image', 'img_data', 'project_name')
STAGE_SETTINGS = {
    'split': ('split_seed', 'val_test_ratio', 'split_policy'),
    'preprocess': ('scale_target_method', 'scale_row_method', 'scale_nwp_method', 'scale_img_method',
                   'nwp_data_merge', 'compress_data'),
    'clustering': ('is_Fuzzy', 'clustering', 'split_policy', 'val_test_ratio', 'split_seed'),
    'features': ('feature_selection_methods', 'feature_selection', 'nwp_data_merge', 'compress_data',
                 'scale_row_method', 'scale_nwp_method'),
}
WHOLE_CONFIGURATION = ('training', 'predict', 'select', 'combine', 'calibrate', 'evaluate')


def settings(static, key):
    """The settings a dataset stage is built from, as ``configuration`` hashes them."""
    values = {name: static.get(name) for name in (*COMMON_SETTINGS, *STAGE_SETTINGS.get(key, ()))}
    # Provider source locations can move without changing their content or tensor contract.
    values['NWP'] = [{k: v for k, v in provider.items() if not k.startswith('path_')}
                     for provider in static.get('NWP') or []]
    return values


def configuration(static, key):
    """The producer's semantic inputs, excluding model generations and execution budgets."""
    if key in WHOLE_CONFIGURATION:
        return lineage.config_identity(static, budgets=False)
    return lineage.dependency_id(settings(static, key))


def changed_settings(before, after, key):
    """The names of the settings of stage ``key`` that differ between two configurations.

    The answer to *which* setting made ``configuration`` change. For the stages that hash the
    whole configuration it is ``lineage.config_differences``.
    """
    if key in WHOLE_CONFIGURATION:
        return lineage.config_differences(before, after)
    old, new = settings(before, key), settings(after, key)
    return [name for name in new if lineage.dependency_id({'value': old.get(name)})
            != lineage.dependency_id({'value': new.get(name)})]


def transformer_state(static, key):
    """Bind only this producer's fitted transformation parameters in the shared aggregate.

    Adding unrelated NWP/image parameters must not invalidate an already published row view.
    Stateless transformations retain a null entry; no fitted history is invented for them.
    """
    import joblib
    types = {'row': {'timeseries', 'calendar'}, 'lstm': {'timeseries', 'calendar'},
             'nwp': {'nwp'}, 'images': {'image'}}.get(key, set())
    names = [v['name'] for v in static.get('variables', []) if v.get('type') in types and v.get('transformer')]
    if not names:
        return None
    file = Path(static['path_model']) / 'transformers.pickle'
    state = joblib.load(file) if file.is_file() else {}
    if not isinstance(state, dict):
        raise IncompleteStage('fitted transformation state must be a mapping')
    return {'algorithm': 'joblib-sha1', 'values': {name: joblib.hash(state[name], hash_name='sha1') if name in state else None
                                                for name in sorted(names)}}


def _record(path, *, verify=True):
    from eforecast.common_utils import file_stamps
    path = _absolute(path)
    # A stamp: the five numbers of the stat and, read at the same moment, the inode's own
    # attributes -- what tells a change time moved by a sync client apart from a file that was
    # rewritten. The observation keeps its five numbers, receipts are read by that shape, and
    # the attributes are recorded beside it (``stamp_of`` puts the two together again).
    try:
        regular = stat.S_ISREG(os.stat(path).st_mode)
        *observation, attributes = file_stamps.stamp(path)
    except OSError:
        regular = False
    if not regular or observation[2] == 0:
        raise IncompleteStage(f'missing or empty payload: {path}')
    # Sidecars are a cheap observation on previews. Acceptance checks the bytes themselves.
    digest = content_digest(path) if verify else (lineage.file_identity(path) or {}).get('sha256')
    return {'path': path, 'bytes': observation[2], 'mtime_ns': observation[3], 'sha256': digest,
            'observation': observation, 'attributes': attributes}


def stamp_of(record):
    """The stamp a file record holds: its observation, with the attributes when it recorded any."""
    observation = (record or {}).get('observation')
    if isinstance(observation, list) and record.get('attributes') is not None:
        return [*observation, record['attributes']]
    return observation


def _semantic(records):
    """Locations/mtimes are observations; equal bytes at a new location are reusable inputs."""
    return [{'bytes': item['bytes'], 'sha256': item['sha256']} for item in records]


def _read(path):
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError('receipt must be a mapping')
        return value
    except (OSError, ValueError) as error:
        raise IncompleteStage(f'unreadable stage receipt {path}: {error}') from error


def _scopes(static):
    model = Path(static['path_model'])
    scopes = [model / 'global']
    if static.get('is_Fuzzy') and (model / 'clusters.pickle').is_file():
        import joblib
        mapping = joblib.load(model / 'clusters.pickle')
        if not isinstance(mapping, dict) or any(not isinstance(p, str) for p in mapping.values()):
            raise IncompleteStage('clusters.pickle must map cluster names to folders')
        scopes += [Path(p) for p in mapping.values()]
    return list(dict.fromkeys(scopes))


@boundary
def model_files(static, *, cpu=None, verify=True, requested=None):
    """Inventory admitted trials with full manifest checks at this boundary, including GRAPH."""
    from eforecast.common_utils.completion import is_finished
    from .capabilities import CPU_METHODS
    model = Path(static['path_model'])
    current_dependencies = None
    found = []
    for scope in _scopes(static):
        section = ('global_methods',) if scope == model / 'global' else ('cluster_methods', 'methods_cpu')
        methods = {m for s in section for m, on in (static.get(s) or {}).items() if on}
        if requested is not None:
            methods = set(requested[0] if scope == model / 'global' else requested[1])
        for method in sorted(methods):
            if cpu is not None and (method in CPU_METHODS) != cpu:
                continue
            from eforecast.common_utils.trial_layout import method_directory
            for trial in sorted(method_directory(static, scope, method).glob('test_*')):
                if not trial.is_dir() or not (trial / 'net_weights.pickle').is_file():
                    continue
                usable, _ = is_finished(str(trial), method)
                if not usable:
                    continue
                manifest = artifact.read(str(trial))
                if manifest is not None:
                    verdict, reason = artifact.inspect(str(trial))
                    if verdict != 'verified':
                        continue  # an explicit weights-only discovery policy cannot waive a modern rejection
                    if verify:
                        for name, record in manifest['files'].items():
                            if content_digest(trial / name) != record['sha256'].lower():
                                raise IncompleteStage(f'{trial}/{name}: content differs from manifest')
                    recorded = manifest.get('depends_on')
                    if recorded is not None and verify:
                        if not isinstance(recorded, dict):
                            raise IncompleteStage(f'{trial}: invalid recorded input lineage')
                        if current_dependencies is None:
                            current_dependencies = lineage.dependencies(static)
                        for name, identity in (recorded.get('datasets') or {}).items():
                            current = current_dependencies['datasets'].get(name)
                            if not isinstance(identity, dict) or current is None:
                                raise IncompleteStage(f'{trial}: missing recorded dataset {name}')
                            if identity.get('sha256'):
                                from eforecast.datasets.files_manager import FilesManager
                                path = FilesManager(static, train=True).dataset_files().get(name)
                                if not path or content_digest(path) != identity['sha256']:
                                    raise IncompleteStage(f'{trial}: dataset {name} differs from model inputs')
                            elif (identity.get('bytes'), identity.get('mtime_ns')) != (current['bytes'], current['mtime_ns']):
                                raise IncompleteStage(f'{trial}: legacy dataset observation changed: {name}')
                        if recorded.get('partition') is not None and recorded['partition'] != current_dependencies['partition']:
                            raise IncompleteStage(f'{trial}: global partition differs from model inputs')
                        scaler = recorded.get('scalers')
                        transformer = recorded.get('transformers')
                        for label, identity, path in (('global scalers', scaler, model / 'scalers.pickle'),
                                                      ('fitted transformations', transformer, model / 'transformers.pickle')):
                            if identity:
                                if not path.is_file():
                                    raise IncompleteStage(f'{trial}: missing {label}')
                                matches = (content_digest(path) == identity['sha256']) if identity.get('sha256') else (
                                    identity.get('bytes'), identity.get('mtime_ns')) == (path.stat().st_size, path.stat().st_mtime_ns)
                                if not matches:
                                    raise IncompleteStage(f'{trial}: {label} differ from model inputs')
                        for section in ('memberships', 'selectors'):
                            for name, identity in (recorded.get(section) or {}).items():
                                path = model / name
                                if not path.is_file() or not isinstance(identity, dict):
                                    raise IncompleteStage(f'{trial}: missing recorded {section}: {name}')
                                if identity.get('sha256'):
                                    matches = content_digest(path) == identity['sha256']
                                else:
                                    matches = (identity.get('bytes'), identity.get('mtime_ns')) == (
                                        path.stat().st_size, path.stat().st_mtime_ns)
                                if not matches:
                                    raise IncompleteStage(f'{trial}: changed {section}: {name}')
                        if recorded.get('image_pixels') is not None:
                            from eforecast.common_utils.image_assets import paths as pixel_paths
                            current_pixels = {str(p.relative_to(Path(static['path_image']))): p for p in pixel_paths(static)}
                            if set(current_pixels) != set(recorded['image_pixels']):
                                raise IncompleteStage(f'{trial}: raw-image pixel roster changed')
                            for name, record in recorded['image_pixels'].items():
                                identity = record.get('identity') or {}
                                path = current_pixels[name]
                                if not path.is_file() or (identity.get('sha256') and content_digest(path) != identity['sha256']):
                                    raise IncompleteStage(f'{trial}: raw-image pixels changed: {name}')
                                if not identity.get('sha256'):
                                    stat = path.stat()
                                    observed = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
                                    if record.get('observation') != observed:
                                        raise IncompleteStage(f'{trial}: legacy pixel observation changed: {name}')
                    scope_file = trial / 'training_scope.json'
                    masks_file = scope / 'cv_mask.pickle'
                    if verify and scope_file.is_file() and masks_file.is_file():
                        import joblib
                        from eforecast.datasets.data_preprocessing.partition_report import masks_identity
                        recorded_scope = _read(scope_file)
                        supplied = recorded_scope.get('supplied_masks_id')
                        if supplied and supplied != masks_identity(joblib.load(masks_file)):
                            raise IncompleteStage(f'{trial}: supplied cluster/global masks changed')
                    found += [trial / name for name in manifest['files']]
                    found.append(trial / artifact.MANIFEST_FILE)
                else:
                    found.append(trial / 'net_weights.pickle')
                    for name in ('trial_result.json', 'predictions_eval.pickle', 'parameters.pickle'):
                        if (trial / name).is_file():
                            found.append(trial / name)
    return list(dict.fromkeys(found))


def measured(static):
    """The timestamps of a case's series that hold a measurement, as the dataset builder takes them.

    It drops the rows with nothing in them before anything else, and every stage works from what
    is left. A series that opens with empty rows, or has a week of them in the middle, has days in
    its file that no stage ever asks weather or images for.
    """
    import pandas as pd
    series = pd.read_csv(static['filename'], header=0, index_col=0, parse_dates=True)
    return series.dropna(how='all', axis=0).index


def weather_days(static):
    """The days a case needs weather for, which are the days the ``extract`` stage extracts."""
    return measured(static).round('D').unique().sort_values()


def product_paths(static, plan, product, *, verify=True):
    """Resolve physical bundles rather than treating one directory entry as completion."""
    import joblib
    model = Path(static['path_model'])
    data = Path(static['path_data'])
    if product == 'source':
        return [Path(static['filename'])]
    if product == 'prepared_trials':
        from eforecast.common_utils.trial_layout import method_directory
        return sorted(p for scope in _scopes(static) if scope.is_dir() for method in scope.iterdir() if method.is_dir()
                      for p in method_directory(static, scope, method.name).glob('test_*/parameters.pickle'))
    if product == 'models':
        node = next(n for n in plan['nodes'] if n['key'] == 'training')
        if node['selected']:
            from .stage_catalogue import _methods
            return model_files(static, verify=verify, requested=_methods(static, node['options']))
        return model_files(static, verify=verify)
    if product == 'extracted_nwp':
        # the same days the extract stage extracts: a day with no measurement is not one of them
        return [Path(static['path_group_nwp']) / f"{p['model']}_{day.strftime('%d%m%y')}.pickle"
                for p in static.get('NWP') or [] for day in weather_days(static)]
    if product == 'global_masks':
        return [model / 'global/cv_mask.pickle']
    if product == 'image_pixels':
        from eforecast.common_utils.image_assets import paths
        return paths(static)
    if product == 'memberships':
        paths = [model / 'clusters.pickle']
        if paths[0].is_file():
            clusters = joblib.load(paths[0])
            if not isinstance(clusters, dict) or not clusters:
                raise IncompleteStage('cluster membership mapping is empty or invalid')
            paths += [Path(p) / 'cv_mask.pickle' for p in clusters.values()]
            for method in static.get('clustering', {}).get('methods', []):
                folder = model / 'cluster_organizer' / method
                from eforecast.common_utils.fuzzy_bundle import bundle_root, current_files
                paths.extend(current_files(folder))
                folder = bundle_root(folder)
                file = folder / 'fuzzy_model.pickle'
                if file.is_file():
                    paths.append(file)
                    state = joblib.load(file)
                    networks = state.get('fuzzy_models') if isinstance(state, dict) else None
                    if isinstance(networks, list):
                        paths += [folder / f'RBF_fuzzy_net_{index}' / name for index in range(len(networks))
                                  for name in ('net_weights.pickle', 'net_weights.pt')
                                  if (folder / f'RBF_fuzzy_net_{index}' / name).is_file()]
        return paths
    if product == 'features':
        scopes = [s for s in _scopes(static) if s != model / 'global' or any((static.get('global_methods') or {}).values())]
        return [*([model / 'feature_selectors.pickle'] if (model / 'feature_selectors.pickle').is_file() else []),
                *[s / 'feature_selectors.pickle' for s in scopes],
                *[s / 'feature_selectors.partition.json' for s in scopes if (s / 'feature_selectors.partition.json').is_file()]]
    if product == 'predictions':
        requested = set(next(n for n in plan['nodes'] if n['key'] == 'predict')['options'].get('stage', []))
        selected_only = not bool(requested & {'predict', 'select', 'all'})
        paths = []
        for p in model_files(static, verify=verify):
            if p.name != 'net_weights.pickle':
                continue
            selection = p.parent.parent / 'selected_trials.json'
            if selected_only and selection.is_file() and p.parent.name not in _read(selection).get('trials', []):
                continue
            cache = p.parent / 'predictions_eval.pickle'
            paths.append(cache)
            if p.parent.parent.name != 'GRAPH':
                point_key = Path(str(cache) + '.key.json')
                paths.append(point_key)
                quantiles = p.parent / 'quantiles_eval.pickle'
                quantile_key = Path(str(quantiles) + '.key.json')
                if quantiles.is_file() and quantile_key.is_file() and point_key.is_file():
                    expected = _read(point_key).get('key')
                    if isinstance(expected, str) and _read(quantile_key).get('key') == expected:
                        paths.extend([quantiles, quantile_key])
        return paths
    if product == 'selection':
        methods = set()
        for path in product_paths(static, plan, 'predictions', verify=verify):
            if path.name == 'predictions_eval.pickle' and path.is_file() and len(joblib.load(path)):
                methods.add(path.parent.parent / 'selected_trials.json')
        return [model / 'Results/selection.csv', *sorted(methods)]
    if product == 'combined':
        configured = set((static.get('combining') or {}).get('methods') or []) & {'bcp', 'kmeans', 'elastic_net'}
        fitted = sorted((model / 'combine_models').rglob('*_model.pickle'))
        fitted += [p for scope in _scopes(static) for p in sorted((scope / 'combine').rglob('*_model.pickle'))]
        fitted = [path for path in fitted if path.name.removesuffix('_model.pickle') in configured]
        return [model / 'combined_eval.pickle', *fitted]  # mean/single need only the result checkpoint
    if product == 'image_source':
        import pandas as pd
        from eforecast.common_utils.resolutions import pandas_freq
        from eforecast.common_utils.image_assets import paths as image_paths
        specification = plan['products'][product]
        # Raw and processed caches can both be used by one producer. An empty raw folder
        # must not hide an existing processed cache. Pin only pre-existing processed inputs;
        # files produced by this attempt are outputs, not newly discovered inputs.
        processed = (Path(static['path_image']) / 'processed').absolute()
        raw = [Path(p).absolute() for p in specification['paths'] if Path(p).absolute() != processed]
        files = [p for base in raw if base.is_dir() for p in base.rglob('*')
                 if p.is_file() and not p.is_relative_to(processed) and DIRECTORY not in p.parts
                 and not p.name.endswith(('.tmp', '.identity.json'))]
        retained = specification.get('observed_processed')
        if retained is None:
            dates = measured(static).round(pandas_freq(static['ts_resolution'])).unique()
            mapping = {f'{v["source"]}_{v["name"]}': dates for v in static.get('variables', []) if v.get('type') == 'image'}
            retained = [p for p in image_paths(static, dates=mapping) if p.is_file()]
        return sorted(set(files + list(map(Path, retained))))
    candidates = [Path(p) for p in plan['products'][product]['paths'] if p]
    # row has a documented CSV compatibility reader; these are alternatives, not two outputs.
    if product == 'row':
        return [next((p for p in candidates if p.is_file()), candidates[0])]
    if product == 'partition':
        return [data / 'cv_mask.pickle']
    if product == 'scalers':
        return [model / 'scalers.pickle', *[model / name for name in ('scalers_fingerprint.json', 'scalers_provenance.json')
                                          if (model / name).is_file()]]
    return candidates


@boundary
def input_records(static, plan, key, *, verify=True, stack=(), check_sources=True):
    node = next(n for n in plan['nodes'] if n['key'] == key)
    out = {}
    for name in node.get('execution_inputs', node['inputs']):
        product = plan['products'][name]
        producer = product['producer']
        if not check_sources and (producer is None or name == 'extracted_nwp'):
            continue  # consuming a published dataset does not re-read its historical raw inputs
        evidence = None
        if producer and producer != key and name != 'models':
            evidence = inspect_stage(static, producer, verify=verify, stack=(*stack, key), product=name, check_sources=False)
            if evidence['status'] in {'rejected', 'incomplete'}:
                raise IncompleteStage(f'{key} requires {name}: {evidence["reason"]}')
        paths, gaps = required_paths(static, plan, name, verify=verify)
        if not paths:
            raise IncompleteStage(f'{key} requires {name}: no physical payload')
        if verify:
            prime(paths, f'{name} file(s)')
        records = [_record(p, verify=verify) for p in dict.fromkeys(paths)]
        provenance = evidence['status'] if evidence else 'external'
        policy = None
        if name == 'models':
            from eforecast.common_utils.completion import discovery_policy
            from eforecast.common_utils.image_assets import trial_uses_pixels
            policy = discovery_policy()
            trials = [Path(p).parent for p in paths if Path(p).name == 'net_weights.pickle']
            manifests = [artifact.read(str(path)) for path in trials]
            provenance = 'legacy_trials' if policy == 'legacy_weights' or any(
                not manifest or not manifest.get('depends_on') or (trial_uses_pixels(static, path) and
                    not manifest['depends_on'].get('image_pixels')) for path, manifest in zip(trials, manifests)) else 'verified_trials'
        out[name] = {'files': records, 'identity': lineage.dependency_id(_semantic(records)),
                     'provenance': provenance, **({'admission_policy': policy} if policy else {}),
                     **({'gaps': [str(path) for path in gaps]} if gaps else {})}
    extras = {}
    if key == 'extract' and check_sources:
        extras['external:nwp_raw'] = raw_nwp_sources(static)
    if check_sources and key == 'images' and ('image_source' in node.get('execution_inputs', node['inputs']) or node.get('validate_image_encoders')):
        extras['external:image_encoders'] = [Path(static['path_group']) / f'best_ae_{v["name"]}_model_hor_{h}.pt'
            for v in static.get('variables', []) if v.get('type') == 'image' for h in range(1, int(static['horizon']) + 1)]
    for name, paths in extras.items():
        if name in node.get('unobservable_inputs', []):
            continue
        records, unusable = [], []
        if verify:
            prime(paths, f'{name} file(s)')
        for path in sorted(set(map(str, paths))):
            try:
                records.append(_record(path, verify=verify))
            except IncompleteStage:
                unusable.append(path)
        if unusable and name not in RECOVERABLE:
            raise IncompleteStage(_missing_inputs(name, unusable))
        if unusable and (name, tuple(unusable)) not in _described:
            _described.add((name, tuple(unusable)))     # every check of a run finds the same gaps
            log.warning('%s', _raw_gaps(name, unusable))
        out[name] = {'files': records, 'identity': lineage.dependency_id(_semantic(records)),
                     'provenance': 'external', **({'gaps': unusable} if unusable else {})}
    return out


def raw_nwp_sources(static):
    """Every raw provider file the extract stage would read for the days this case covers."""
    from eforecast.nwp_extraction import sources
    files = []
    for provider in static.get('NWP') or []:
        for day in weather_days(static):
            folder = sources.day_folder(sources.roots(provider.get('path_nwp_source')
                                                      or static['nwp_folder']), day)
            if folder:
                files += sources.grib_files(folder)
    return files


def weatherless(static, paths):
    """Split declared daily NWP payloads into the required ones and the days with no weather.

    One payload is one day of one provider, and a day the provider never supplied cannot
    become one: its folder is absent, or holds nothing but empty files and steps past the
    extractor's horizon. The dataset already treats such a day as a sample without weather --
    ``read_nwp_pickle`` finds no file, says so and carries on -- exactly as it does for the
    empty files ``_raw_gaps`` describes. So the stage reports the day instead of failing on
    it; refusing to publish 1005 extracted days because the provider skipped one makes the
    other 1005 unreachable and no weather appears for the missing one.

    A payload that exists is never questioned here: cached weather deliberately outlives the
    raw inputs it was made from, and a day whose name cannot be read is left to the ordinary
    check to speak about.
    """
    import pandas as pd
    from eforecast.nwp_extraction import sources
    providers = {p['model']: p for p in static.get('NWP') or []}
    required, absent = [], []
    for path in map(Path, paths):
        if path.is_file():
            required.append(path)
            continue
        model, _, stamp = path.stem.rpartition('_')
        provider = providers.get(model)
        try:
            day = pd.to_datetime(stamp, format='%d%m%y')
        except ValueError:
            provider = None
        if provider is None:
            required.append(path)
            continue
        folder = sources.day_folder(sources.roots(provider.get('path_nwp_source')
                                                 or static['nwp_folder']), day)
        (required if folder and sources.holds_weather(folder) else absent).append(path)
    return required, absent


def required_paths(static, plan, product, *, verify=True):
    """The declared payloads a stage must really have, and the ones nothing can produce.

    ``product_paths`` names what the case declares: for daily weather, one payload per provider
    per day the target series covers. Some of those days the provider never supplied, and
    ``weatherless`` says which. That reduction has to be the same fact in every place that
    compares a receipt against the declaration -- when the stage publishes, when a consumer
    checks its prerequisites, when a receipt is accepted, and when it is read back later. When
    one place disagrees the stage publishes happily and then nothing can consume it: the receipt
    is read as omitting a required payload, and the next stage refuses with a message about a
    foreign output location that no amount of extracting can clear.
    """
    paths = list(dict.fromkeys(product_paths(static, plan, product, verify=verify)))
    if product == 'extracted_nwp':
        return weatherless(static, paths)
    return paths, []

def _no_weather(name, paths):
    """What a run says about days the provider never supplied. It does not stop."""
    from eforecast.nwp_extraction import sources
    return (f'{_listing(name, paths)}\n'
            f'Their provider folders are absent, or hold only empty files and steps past '
            f'{sources.MAX_STEP_HOURS}h, so there is nothing to extract for those days and no '
            f'pickle can exist. The dataset is built without them, as it is for any sample with '
            f'no weather. Fetch those days from the provider to include them.')


def _listing(name, paths):
    """One message naming every unusable input, not just the first one found.

    These inputs are a whole date range of raw provider files -- for ``external:nwp_raw``, every
    GRIB of every day the target series covers. Stopping at the first empty one meant learning
    about the rest a run at a time: a dataset blocked on fifteen empty GRIBs needed fifteen runs
    to finish describing itself.
    """
    shown = paths[:20]
    listing = '\n  '.join(shown)
    rest = len(paths) - len(shown)
    tail = f'\n  ... and {rest} more' if rest else ''
    days = sorted({day.name for day in (_day_of(path) for path in paths) if day is not None})
    where = f' spanning {days[0]}..{days[-1]}' if len(days) > 1 else (f' in {days[0]}' if days else '')
    return f'{len(paths)} missing or empty payload(s) for {name}{where}:\n  {listing}{tail}'


def _day_of(path):
    from eforecast.nwp_extraction import sources
    return sources.day_of(path)[0]


def _missing_inputs(name, paths):
    return (f'{_listing(name, paths)}\n'
            f'These are produced by a stage of this pipeline, not fetched: run the stage that '
            f'publishes them before the one that reads them.')


def _raw_gaps(name, paths):
    """What a run says about raw provider files that hold no weather. It does not stop.

    An empty file is missing weather, and no amount of refusing makes it appear. What matters is
    that the dataset is not built *around* it as though the hour were ordinary: the extractor
    drops the step, and the first readable step after it as well, because an accumulated field is
    a total since the analysis time and differencing across a gap would publish two intervals of
    radiation as one. Another analysis cycle usually covers the dropped hours, and a sample with
    no weather at all is a case the dataset already handles. So the run continues, and says once
    which files were empty rather than stopping on each in turn.
    """
    from eforecast.nwp_extraction import refill, sources
    return (f'{_listing(name, paths)}\n'
            f'The dataset is built without those steps, and without the readable step after each '
            f'one: an accumulated field differenced across a gap would hold two intervals and '
            f'claim to hold one. Steps past the extractor\'s max_step_hours '
            f'({sources.MAX_STEP_HOURS}h) are never read, '
            f'so an empty file beyond that costs nothing. Fetching them again is tried first and '
            f'is reported above; {refill.MODE}=off stops the run from reaching the network. '
            f'Deleting the empty files changes nothing -- they are already skipped. '
            f'See docs/troubleshooting.md section 6.')


def _validate_shape(path, product):
    """Read trusted workspace payloads with their producer formats before promotion."""
    import joblib
    import pandas as pd
    path = Path(path)
    if path.suffix == '.json':
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise IncompleteStage(f'{path}: JSON payload must be an object')
    elif path.suffix == '.csv':
        value = pd.read_csv(path)
        if not len(value.columns) or product in {'target', 'row', 'evaluation'} and value.empty:
            raise IncompleteStage(f'{path}: empty {product} table')
        if product in {'target', 'row'} and not isinstance(pd.read_csv(path, index_col=0, parse_dates=True).index, pd.DatetimeIndex):
            raise IncompleteStage(f'{path}: dataset dates are not readable')
    elif path.suffix in {'.pickle', '.pkl'}:
        try:
            # Mapped, not read: what follows asks for types and lengths, which the header of
            # each array answers. Reading an 18 GB weather dataset to count its rows took as
            # long as loading it for training, and as much memory. A compressed pickle cannot
            # be mapped; joblib says so and reads it, which is what happened before.
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='mmap_mode', category=UserWarning)
                value = joblib.load(path, mmap_mode='r')
        except AttributeError:
            if path.name != 'net_weights.pickle':
                raise
            # Use the model reader's existing normalization for old saved runtime monitors.
            from eforecast.deep_models.pytorch_2x.network import load_tolerant
            value = load_tolerant(path)
        if product in {'partition', 'global_masks'} or path.name == 'cv_mask.pickle':
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise IncompleteStage(f'{path}: expected three partition masks')
            masks = [pd.DatetimeIndex(item) for item in value]
            if any(masks[i].intersection(masks[j]).size for i, j in ((0, 1), (0, 2), (1, 2))):
                raise IncompleteStage(f'{path}: partition masks overlap')
        elif not isinstance(value, (dict, pd.DataFrame)) or (len(value) == 0 and product != 'predictions'):
            raise IncompleteStage(f'{path}: empty or invalid {product} payload')
        if product == 'row' and isinstance(value, dict):
            if any(v is not None and not isinstance(v, pd.DataFrame) for v in value.values()) or not any(
                    isinstance(v, pd.DataFrame) and len(v) for v in value.values()):
                raise IncompleteStage(f'{path}: row views contain no readable samples')
        if product == 'lstm':
            if not isinstance(value.get('data'), dict) or not isinstance(value.get('metadata'), dict):
                raise IncompleteStage(f'{path}: LSTM data/metadata are missing')
            frames = [frame for branch in value['data'].values() for frame in branch.values()]
            if not frames or any(not isinstance(frame, pd.DataFrame) or frame.empty for frame in frames):
                raise IncompleteStage(f'{path}: incomplete LSTM views')
        if product == 'nwp':
            leaves = [provider for area in value.values() for variable in area.values() for provider in variable.values()]
            if not leaves or any(not isinstance(leaf, dict) or 'data' not in leaf or 'dates' not in leaf
                                 or not len(leaf['dates']) or len(leaf['data']) != len(leaf['dates']) for leaf in leaves):
                raise IncompleteStage(f'{path}: incomplete NWP tensors/dates')
        if product == 'images':
            if (not isinstance(value.get('dates'), dict) or not isinstance(value.get('stats'), dict) or
                    not any(isinstance(frame, pd.DataFrame) and len(frame) for frame in value['stats'].values())):
                raise IncompleteStage(f'{path}: image dates/statistics are missing')
        if product == 'image_pixels' and (not isinstance(value, dict) or not any(
                getattr(array, 'size', 0) > 0 for array in value.values())):
            raise IncompleteStage(f'{path}: processed image has no pixels')
        if product == 'combined' and path.name == 'combined_eval.pickle':
            def has_samples(item):
                return any(has_samples(v) for v in item.values()) if isinstance(item, dict) else isinstance(item, pd.DataFrame) and not item.empty
            if not has_samples(value):
                raise IncompleteStage(f'{path}: combined checkpoint has no usable predictions')


def _vouch(identity):
    if len(_valid) >= KNOWN_LIMIT:
        _valid.clear()
    _valid.add(identity)


def _unchecked(paths, validated):
    """The payloads whose shape nobody has vouched for: no worker, no receipt, not this process."""
    out = []
    for path in paths:
        if str(Path(path).absolute()) in (validated or {}):
            continue
        try:
            if os.environ.get(REHASH) != '1' and _identity(path, _signature(path)) in _valid:
                continue
        except OSError:
            pass                    # the ordinary check says what is wrong with it
        out.append(path)
    return out


def _checked(path):
    """Remember that this file has the shape of its product, once it has been at rest."""
    try:
        before = _signature(path)
    except OSError:
        return
    if time.time_ns() - max(before[3], before[4]) >= SETTLED_NS:
        _vouch(_identity(path, before))


def shape_error(path, product):
    """``_validate_shape`` for a reader pool: what is wrong with the payload, or nothing."""
    try:
        _validate_shape(path, product)
    except (OSError, ValueError, TypeError, EOFError, KeyError, AttributeError, IndexError,
            pickle.UnpicklingError) as error:
        return f'{error}'
    return None


def _shapes(paths, product):
    """``{path: error or None}`` for a large bundle, checked by the stage's readers.

    A day of weather is a compressed pickle and can only be checked by reading it: a fifth of
    a second each, three and a half minutes for three years of them, one after another. The
    stage has readers for exactly this kind of work; without them (no CPU budget, a debugger)
    the bundle is checked inline as before and this returns nothing.
    """
    from eforecast.bounded_pools import map_bounded, workers
    if len(paths) < POOLED_FROM or workers(None) < 2:
        return {}
    log.info('checking %d %s payloads with the stage readers', len(paths), product)
    return dict(zip(paths, map_bounded(shape_error, [str(p) for p in paths], [product] * len(paths), n_jobs=None)))


@boundary
def output_records(static, plan, key, *, validated=None):
    node = next(n for n in plan['nodes'] if n['key'] == key)
    if key == 'features' and not (Path(static['path_model']) / 'feature_selectors.pickle').is_file():
        raise IncompleteStage('features did not publish its feature_selectors.pickle backup')
    if key == 'training' and not node['options'].get('prepare_only'):
        from .stage_catalogue import _methods
        glob, clusters = _methods(static, node['options'])
        admitted = {p.parent for p in product_paths(static, plan, 'models') if p.name == 'net_weights.pickle'}
        root = Path(static['path_model'])
        for scope in _scopes(static):
            for method in glob if scope == root / 'global' else clusters:
                budget = static.get(method, {}).get('n_trials', 1)
                if isinstance(budget, (int, float)) and budget <= 0:
                    continue
                from eforecast.common_utils.trial_layout import method_directory
                if not any(path.parent == method_directory(static, scope, method) for path in admitted):
                    raise IncompleteStage(f'training has no usable {scope.name}/{method} output')
    if key == 'clustering' and static.get('is_Fuzzy'):
        import joblib
        for method in static.get('clustering', {}).get('methods', []):
            folder = Path(static['path_model']) / 'cluster_organizer' / method
            from eforecast.common_utils.fuzzy_bundle import bundle_root
            folder = bundle_root(folder)
            file = folder / 'fuzzy_model.pickle'
            if not file.is_file():
                raise IncompleteStage(f'clustering is missing {file}')
            state = joblib.load(file)
            networks = state.get('fuzzy_models') if isinstance(state, dict) else None
            if not isinstance(networks, list) or not networks:
                raise IncompleteStage(f'{file}: missing trained fuzzy networks')
            for index in range(len(networks)):
                for name in ('net_weights.pickle', 'net_weights.pt'):
                    _record(folder / f'RBF_fuzzy_net_{index}' / name)
    out = {}
    for name in node['outputs']:
        paths, without = required_paths(static, plan, name)
        if without:
            log.warning('%s', _no_weather(name, [str(path) for path in without]))
        if not paths:
            raise IncompleteStage(f'{key} did not publish required {name}')
        out[name] = []
        pending = _unchecked(paths, validated)
        failures, unchecked = _shapes(pending, name), set(pending)
        prime(paths, f'{name} file(s)')
        for path in paths:
            try:
                checksum = (validated or {}).get(str(Path(path).absolute()))
                if checksum is not None:
                    if checksum != content_digest(path):
                        raise IncompleteStage(f'validated worker payload changed: {path}')
                elif path in unchecked:
                    if path in failures:
                        if failures[path]:
                            raise IncompleteStage(failures[path])
                    else:
                        _validate_shape(path, name)
                    _checked(path)
                out[name].append(_record(path))
            except (OSError, ValueError, TypeError, EOFError, KeyError, AttributeError, IndexError, pickle.UnpicklingError) as error:
                raise IncompleteStage(f'{key}: invalid {name} payload {path}: {error}') from error
        if name == 'predictions':
            if paths and all(str(Path(path).absolute()) in (validated or {}) for path in paths):
                # The common worker checked nonempty coverage and every current
                # cache key while its arrays/models still belonged to a grant.
                # Above, their exact bytes were checked against that proof.
                continue
            import joblib
            if not any(len(joblib.load(path)) for path in paths if path.name == 'predictions_eval.pickle'):
                raise IncompleteStage('no usable evaluation-period predictions were produced')
            from eforecast.combine_predictions.ensemble import TrialPredictor
            predictor = TrialPredictor(static)
            for path in paths:
                if path.name == 'predictions_eval.pickle' and path.parent.parent.name != 'GRAPH':
                    expected = predictor.cache_key({'path_weights': str(path.parent)})
                    if _read(str(path) + '.key.json').get('key') != expected:
                        raise IncompleteStage('prediction cache was not published for the current model and inputs')
    return out


@boundary
def inspect_stage(static, key, *, verify=True, stack=(), product=None, check_sources=True):
    """Read-only compatibility/verification path; malformed modern claims never become legacy."""
    path = receipt_path(static, key)
    if not path.exists():
        if running_marker(static, key).exists():
            # Output with no receipt is normally old work of unknown provenance, and is used as
            # such. Beside the marker of a stage that never finished it is something else: what
            # an interrupted run left, which may be half of a file.
            return {'status': 'rejected', 'path': str(path),
                    'reason': f'{key} did not finish: it was interrupted, or is still running. What it left is '
                              f'not trusted; run the stage again, or recover the case'}
        return {'status': 'legacy', 'reason': 'no stage receipt; historical producer/input provenance is unknown'}
    try:
        if key in stack:
            raise IncompleteStage('cyclic receipt dependency')
        value = _read(path)
        if type(value.get('version')) is not int or value['version'] != VERSION or value.get('stage') != key:
            raise IncompleteStage('unknown receipt version or stage')
        supplied = value.get('id')
        if supplied != lineage.dependency_id({k: v for k, v in value.items() if k != 'id'}):
            raise IncompleteStage('receipt identity does not match its contents')
        _remember(path, value)      # whatever else is wrong with it, what it observed stays true
        if value.get('state') not in {'complete', 'legacy'}:
            raise IncompleteStage(f'stage publication is {value.get("state")!r}')
        if value.get('configuration') != configuration(static, key):
            raise ConfigurationChanged(f'{key} configuration changed; rebuild this stage')
        producer = value.get('producer')
        if (not isinstance(producer, dict) or not isinstance(producer.get('attempt_uid'), str) or
                not re.fullmatch('[0-9a-f]{32}', producer['attempt_uid']) or
                type(producer.get('fence')) is not int or producer['fence'] <= 0):
            raise IncompleteStage('missing producer attempt identity')
        if not isinstance(value.get('outputs'), dict) or not value['outputs']:
            raise IncompleteStage('missing expected output bundles')
        plan = value.get('contract')
        if not isinstance(plan, dict):
            raise IncompleteStage('missing receipt input/output contract')
        node = next((n for n in plan['nodes'] if n['key'] == key), None)
        if not node or set(node['outputs']) != set(value['outputs']):
            raise IncompleteStage('receipt omits an expected output bundle')
        missing_product = product and product not in value['outputs']
        bundles = {product: value['outputs'][product]} if product and not missing_product else value['outputs']
        changed_observation = False
        for name, records in bundles.items():
            declared, _ = required_paths(static, plan, name, verify=verify)
            if {str(p.absolute()) for p in declared} != {r['path'] for r in records}:
                raise IncompleteStage('receipt omits a required payload or names a foreign output location')
        for records in bundles.values():
            if not isinstance(records, list) or not records:
                raise IncompleteStage('invalid output bundle')
            for record in records:
                if (not isinstance(record, dict) or not isinstance(record.get('sha256'), str) or
                        not re.fullmatch('[0-9a-f]{64}', record['sha256']) or
                        type(record.get('bytes')) is not int or record['bytes'] < 0 or
                        type(record.get('mtime_ns')) is not int or record['mtime_ns'] < 0 or
                        not isinstance(record.get('path'), str)):
                    raise IncompleteStage('invalid output metadata')
                current = _record(record.get('path'), verify=verify)
                if current['bytes'] != record.get('bytes') or (verify and current['sha256'] != record['sha256']):
                    raise IncompleteStage(f'changed output {record["path"]}')
                # The five numbers, strictly: this compares with a receipt of another run, and what
                # it decides is only whether the content is read again. (Within one run, a change
                # time that a sync client moved is told apart from a write: ``file_stamps``.)
                if not verify and (current['mtime_ns'] != record.get('mtime_ns') or
                                   record.get('observation') is not None and
                                   current['observation'] != record['observation']):
                    changed_observation = True  # a touch is not proof that the bytes changed
        degraded = []
        if value.get('transformers') is not None:
            if transformer_state(static, key) != value['transformers']:
                if not (Path(static['path_model']) / 'transformers.pickle').is_file():
                    degraded.append('fitted_transformers')
                else:
                    raise IncompleteStage('fitted transformation parameters changed; rebuild this dataset view')
        # Cached weather/image datasets have always been usable after their raw caches were
        # pruned. Keep that route explicit: verify retained outputs and present raw inputs, and
        # report unobservable historical raw inputs as legacy rather than inventing verification.
        raw = ({'extract': {'external:nwp_raw'}, 'nwp': {'extracted_nwp'},
                'images': {'image_source', 'external:image_encoders'}}.get(key, set()) if check_sources else set())
        recorded_inputs = value.get('inputs', {})
        for name in raw & set(recorded_inputs):
            records = recorded_inputs[name]['files']
            if any(not Path(record['path']).is_file() for record in records):
                for record in records:
                    if Path(record['path']).is_file():
                        current = _record(record['path'], verify=verify)
                        if ((verify and current['sha256'] != record['sha256']) or
                                not verify and current['bytes'] != record['bytes']):
                            raise IncompleteStage('retained raw input changed beside pruned inputs')
                        if not verify and current['mtime_ns'] != record['mtime_ns']:
                            changed_observation = True
                degraded.append(name)
        # Legacy inspection records observed bytes, never an invented historical input binding.
        if value['state'] == 'complete':
            current_plan = deepcopy(plan)
            if degraded:
                current_node = next(n for n in current_plan['nodes'] if n['key'] == key)
                current_node['execution_inputs'] = [n for n in current_node.get('execution_inputs', current_node['inputs']) if n not in degraded]
                current_node['unobservable_inputs'] = degraded
                if key == 'images' and 'external:image_encoders' in recorded_inputs and 'external:image_encoders' not in degraded:
                    current_node['validate_image_encoders'] = True
            current = input_records(static, current_plan, key, verify=verify, stack=stack, check_sources=check_sources)
            for name in degraded:
                current.pop(name, None)
            recorded = {k: v for k, v in recorded_inputs.items() if k not in degraded}
            if not check_sources:
                recorded = {k: v for k, v in recorded.items() if k in current}
            if verify:
                same = {k: v['identity'] for k, v in current.items()} == {k: v['identity'] for k, v in recorded.items()}
            else:
                # Raw source files need not have sidecars. A cheap preview checks their retained
                # observations instead of comparing an unknown digest with a known one.
                same = (set(current) == set(recorded) and all(
                    [(r['bytes'], r['mtime_ns']) for r in current[k]['files']] ==
                    [(r['bytes'], r['mtime_ns']) for r in recorded[k]['files']] for k in current))
            if not same:
                if verify:
                    raise IncompleteStage(f'{key} input content changed; rebuild this stage')
                changed_observation = True
        return {'status': 'legacy' if degraded or missing_product else 'verified' if value['state'] == 'complete' and verify else
                         'recorded' if value['state'] == 'complete' else 'legacy',
                'reason': f'historical receipt does not bind {product}; inspect its payloads' if missing_product else
                          f'cached output retained; historical raw inputs unavailable: {degraded}' if degraded else
                          'content and input identities match' if verify and value['state'] == 'complete' else
                          ('metadata changed; full content verification required' if changed_observation else
                           'receipt observed; full content verification occurs at execution') if value['state'] == 'complete' else
                          'inspected legacy payload; original producer/input binding remains unknown',
                # 'recorded' alone cannot say whether a cheap check was reassuring or merely
                # inconclusive; a reader that must decide needs the observation itself.
                'metadata': 'changed' if changed_observation else 'matches',
                'id': value['id'], 'producer': value['producer'], 'path': str(path)}
    except (IncompleteStage, OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, IndexError, pickle.UnpicklingError) as error:
        cause = getattr(error, 'cause', None)
        return {'status': 'rejected', 'reason': str(error), 'path': str(path), **({'cause': cause} if cause else {})}


def seal(value):
    value = {k: v for k, v in value.items() if k != 'id'}
    return {**value, 'id': lineage.dependency_id(value)}


def publish_receipt(static, value):
    value = seal(value)
    atomic_json(receipt_path(static, value['stage']), value)
    return value


def completion(static, key, plan, producer, inputs, outputs, *, legacy=False):
    # A minimal contract is enough to re-inspect input bundles without loading mutable case YAML.
    contract = {k: deepcopy(plan[k]) for k in ('products', 'nodes')}
    return seal({'version': VERSION, 'stage': key, 'state': 'legacy' if legacy else 'complete',
        'producer': producer, 'plan_id': plan['identity'], 'configuration': configuration(static, key),
        'effective_revision': producer['effective_revision'], 'inputs': inputs, 'outputs': outputs,
        'contract': contract, 'code': lineage.code_identity(), 'transformers': transformer_state(static, key)})


@boundary
def verify_completion(run, task, event):
    """Collector acceptance, using the accepted snapshot and canonical receipt location."""
    from . import run_inputs
    from eforecast.init.initialize import initializer
    snapshot = json.loads(run.snapshot_json)
    value = event.get('completion')
    if not isinstance(value, dict) or value.get('status') not in {'verified', 'legacy', 'reused'}:
        raise IncompleteStage('successful stage has no verified/legacy completion receipt')
    static = initializer(run_inputs.load(snapshot['config']), dry_run=True)
    evidence = inspect_stage(static, task.stage)
    if evidence['status'] not in {'verified', 'legacy'} or evidence.get('id') != value.get('id'):
        raise IncompleteStage(f'stage completion was not verified: {evidence["reason"]}')
    receipt = _read(receipt_path(static, task.stage))
    plan = snapshot['stage_plan']
    producer = receipt['producer']
    if value['status'] == 'reused':
        # An earlier run produced this artifact and this run was given permission to keep it.
        # The bytes were just verified above; what remains is that the permission was real and
        # that the receipt is the one the stage says it reused.
        from .stage_resume import permitted
        if not permitted(plan, task.stage):
            raise IncompleteStage('stage reported reuse without a pinned resume decision')
        if producer != value.get('producer') or receipt['id'] != value.get('id'):
            raise IncompleteStage('reused receipt is not the one this stage names')
        return {**value, 'version': VERSION}
    expected = {'run_id': run.id, 'task_id': run.tasks[0].id,
        'attempt_id': run.tasks[0].attempts[-1].id, 'executor_id': run.tasks[0].attempts[-1].executor.id,
        'stage_task_id': task.id, 'stage_attempt_id': task.attempts[-1].id}
    if (receipt['plan_id'] != plan['identity'] or producer != value.get('producer') or
            any(producer.get(k) != v for k, v in expected.items())):
        raise IncompleteStage('stage receipt belongs to a foreign plan or producer attempt')
    if receipt.get('effective_revision') != snapshot.get('effective_revision'):
        raise IncompleteStage('stage receipt differs from the accepted effective input revision')
    node = next(n for n in plan['nodes'] if n['key'] == task.stage)
    if set(receipt['outputs']) != set(node['outputs']):
        raise IncompleteStage('stage receipt differs from the expected payload contract')
    # No omitted file can be smuggled through a short, otherwise valid bundle.
    for product, records in receipt['outputs'].items():
        declared, _ = required_paths(static, plan, product)
        if {str(p.absolute()) for p in declared} != {r['path'] for r in records}:
            raise IncompleteStage('stage receipt omits required physical payloads')
    return {**value, 'version': VERSION}
