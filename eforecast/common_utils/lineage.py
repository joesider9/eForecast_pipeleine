# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""What an artifact depends on, as identities a later reader can compare.

A trained model is only reusable against the inputs it was trained on: the dataset files, the
partition, the globally fitted scalers, the selectors, the configuration and the code that ran.
Those were implicit -- a folder's existence meant "current" -- so a changed input silently kept
old models in service. Each of them has an identity here, and the artifact records them.

The cost is bounded by never hashing a dataset on the path of a trial: a dataset's content
identity is published once, when the stage that wrote it finishes, into a sidecar beside the file.
A reader that finds no sidecar reports size and modification time instead, and says so, rather
than reading gigabytes to answer a bookkeeping question.
"""
import hashlib
import json
import logging
import os
import subprocess
import sys

log = logging.getLogger(__name__)

SIDECAR = '.identity.json'
VERSION = 1
_code = {}


def digest_file(path, blocks=1024 * 1024):
    sha = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(blocks), b''):
            sha.update(block)
    return sha.hexdigest()


def sidecar_path(path):
    return f'{path}{SIDECAR}'


def stat_of(path):
    info = os.stat(path)
    return {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns}


def publish_identity(path, *, sha256=None):
    """Hash a file once and keep the answer beside it; returns the identity."""
    identity = {'version': VERSION, 'sha256': sha256 or digest_file(path), **stat_of(path)}
    temporary = f'{sidecar_path(path)}.{os.getpid()}.tmp'
    try:
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(identity, stream)
        os.replace(temporary, sidecar_path(path))
    except OSError as error:
        log.warning('cannot publish the identity of %s (%r)', path, error)
        if os.path.exists(temporary):
            os.unlink(temporary)
    return identity


def file_identity(path, compute=False):
    """The content identity of a file, from its sidecar when that still matches it.

    ``compute`` is for the stage that produced the file. A reader leaves it false: it gets the
    published identity, or size and modification time with ``confidence`` saying which it is.
    """
    if not path or not os.path.isfile(path):
        return None
    current = stat_of(path)
    try:
        with open(sidecar_path(path), encoding='utf-8') as stream:
            stored = json.load(stream)
        if (stored.get('bytes'), stored.get('mtime_ns')) == (current['bytes'], current['mtime_ns']) \
                and isinstance(stored.get('sha256'), str):
            return {**current, 'sha256': stored['sha256'], 'confidence': 'content'}
    except (OSError, ValueError):
        pass
    if compute:
        return {**publish_identity(path), 'confidence': 'content'}
    return {**current, 'sha256': None, 'confidence': 'size_mtime'}


def dataset_identities(files_manager, compute=False):
    """One identity per dataset product this case has on disk."""
    out = {}
    for name, path in files_manager.dataset_files().items():
        identity = file_identity(path, compute=compute)
        if identity is not None:
            out[name] = identity
    return out


def publish_dataset_identities(static_data):
    """Called when a dataset stage finishes: the hashing happens here, once, and not in a trial."""
    from eforecast.datasets.files_manager import FilesManager

    identities = dataset_identities(FilesManager(static_data, train=True), compute=True)
    log.info('dataset identities published for %d product(s): %s', len(identities),
             ', '.join(f"{name} {value['sha256'][:12]}" for name, value in identities.items()))
    return identities


def code_identity():
    """The revision that ran, the image it ran in, and the libraries that decide the numbers."""
    if _code:
        return _code
    revision, dirty = None, None
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        revision = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True,
                                  text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(['git', '-C', root, 'status', '--porcelain'], capture_output=True,
                                    text=True, timeout=10).stdout.strip()) if revision else None
    except (OSError, subprocess.SubprocessError):
        pass
    libraries = {'python': sys.version.split()[0]}
    for module in ('torch', 'numpy', 'pandas', 'sklearn', 'optuna'):
        found = sys.modules.get(module)
        if found is not None and getattr(found, '__version__', None):
            libraries[module] = found.__version__
    _code.update({'revision': revision, 'dirty': dirty, 'image_id': os.environ.get('EFORECAST_IMAGE_ID'),
                  'libraries': libraries})
    return _code


# Execution knobs and machine paths are not what a model learns: two runs that differ only in
# how many workers they were given must produce comparable artifacts.
VOLATILE = ('n_jobs', 'n_cpus', 'n_gpus', 'Docker', 'static_data_base', 'config_revision',
            'execution_profile', 'profile_resources', '_execution_profile',
            'array_reader', 'cache_max_bytes', 'cache_max_entries', 'cache_batch_rows', 'cache_buffer_mb', 'model_cache_mode',
            'cpu_workers', 'cpu_study_threads', 'cpu_study_ram_mb', 'cpu_wait_timeout', 'cpu_study_vram_mb', 'cpu_study_device',
            'cpu_scope', 'study_action', 'study_revision', '_trial_generations', '_trial_scopes')


#: Where a foundation family computes is not part of what it is. Nothing of such a model is
#: learned, so a trial is the same model on a card and on the CPUs, and its numbers differ in
#: the last digits a float holds. Its ``device`` was once hashed like any parameter, and every
#: case then carried the value below. The identity still carries that value, whatever the case
#: says now: a case is moved to a card, or off one, without its studies, its predictions and
#: its stages becoming those of another configuration -- and no identity computed before this
#: was written changes.
PLACEMENT = {'Chronos2': 'cpu', 'TimesFM': 'cpu', 'TiRex': None}


#: Which families are switched on. Membership, not parameters: a case that gains or loses a
#: family has to combine, select and evaluate again, and those stages hash the whole
#: configuration -- but the families already trained are unaffected by a *sibling* arriving,
#: so a per-family identity leaves these out.
MEMBERSHIP = ('global_methods', 'cluster_methods', 'methods_cpu')


#: Settings no dataset is built from and no family learns from: how the trained models are
#: selected, combined and given intervals afterwards, and knobs of how the work is run. They
#: belong to the configuration of the case -- the combination steps hash all of it and are
#: computed again when one changes -- but not to what makes a *family's* trained work its own.
#: A case that tries another combiner, or keeps three models of each family instead of two,
#: has trained nothing differently. Each name was checked against every reader in the code
#: before it was put here; a setting this list does not name counts, which is the safe side.
#:
#: The identity of a family carried them until 17 November 2026. What was recorded under that
#: formula is still recognised: ``comparability_id`` gives the earlier name beside the new one.
NOT_LEARNED = ('combining', 'keep_best_models', 'interval_coverage', 'cluster_coverage_policy',
               'n_jobs_rbfnn', 'n_jobs_lstm', 'n_jobs_cnn_3d', 'n_jobs_cnn', 'n_jobs_mlp', 'intra_op',
               'use_polars')


def family_keys(static_data):
    """The top-level keys that hold one family's own parameters.

    Each family is its own entry in the effective configuration (`static_data['MLP']` and so
    on), so they can be told apart from the shared sections without a list of family names to
    keep in step with the catalogue: a mapping whose name is switched on somewhere is one.
    """
    enabled = set()
    for name in MEMBERSHIP:
        value = static_data.get(name)
        if isinstance(value, dict):
            enabled.update(str(key) for key in value)
    return {name for name in enabled if isinstance(static_data.get(name), dict)}


def config_identity(static_data, budgets=True, *, method=None, not_learned=False):
    """The effective configuration this artifact was trained under, however it was produced.

    ``budgets`` false drops how many trials each method was asked for: raising a budget does not
    make the work already done incomparable with the work still to do.

    ``method`` narrows it to one family: that family's own parameters plus everything shared,
    and **not** the other families' parameters or which families are switched on. Changing the
    learning rate of one family then leaves its siblings' trials comparable, which is the
    difference between retraining one family and retraining all of them. Everything a family
    actually reads -- the variables, the NWP providers, the scaling, the split, the clustering,
    the target -- stays in, so narrowing cannot make a stale trial look reusable. Default
    ``None`` is the whole configuration, byte-identical to before this argument existed.

    What no family learns from (``NOT_LEARNED``) is left out of a family's identity as well.
    ``not_learned`` true keeps it in, which is the formula families were recorded under before
    those settings stopped counting: the earlier name of the same inputs. The whole
    configuration always carries them.
    """
    return dependency_id(config_payload(static_data, budgets, method=method, not_learned=not_learned))


#: What a family is asked to do, as opposed to what it is: how many trials, and how many at
#: once. Neither is hashed when ``budgets`` is false.
BUDGET = ('n_trials', 'n_jobs')


def config_payload(static_data, budgets=True, *, method=None, not_learned=False):
    """What ``config_identity`` hashes, as the mapping itself.

    The identity says that two configurations differ; this is what lets a reader say in which
    setting (``config_differences``). The two are one definition: the identity is the hash of
    exactly this.
    """
    payload = {key: value for key, value in static_data.items()
               if key not in VOLATILE and not key.startswith('path_') and not key.endswith('_folder')
               and key not in ('pycharm_path', 'env_name', 'filename')}
    # Where a provider's files are downloaded from is not part of what they hold: a case that
    # starts fetching its weather from the shared folder, or stops, keeps what it trained. No
    # identity computed before this key existed carries it, so none of them changes.
    if isinstance(payload.get('NWP'), list):
        payload['NWP'] = [{k: v for k, v in provider.items() if k != 'path_nwp_gdrive'}
                          if isinstance(provider, dict) else provider for provider in payload['NWP']]
    for family, device in PLACEMENT.items():
        if isinstance(payload.get(family), dict) and 'device' in payload[family]:
            payload[family] = {**payload[family], 'device': device}
    if method is not None:
        families = family_keys(static_data)
        if method not in families:
            # An unknown family is not an invitation to narrow anything: a name that is not
            # switched on anywhere would silently drop every family's parameters below.
            raise ValueError(f'config_identity: {method!r} is not a configured family')
        payload = {key: value for key, value in payload.items()
                   if key not in MEMBERSHIP and (key not in families or key == method)}
        if not not_learned:
            payload = {key: value for key, value in payload.items() if key not in NOT_LEARNED}
    if not budgets:
        payload = {key: ({k: v for k, v in value.items() if k not in BUDGET}
                         if isinstance(value, dict) else value) for key, value in payload.items()}
    return payload


def _same(left, right):
    return dependency_id({'value': left}) == dependency_id({'value': right})


def config_differences(before, after, *, method=None):
    """The settings in which two configurations differ, as one family -- or the case -- is judged.

    ``['variables', 'MLP.learning_rate']``: a shared setting by its name, a family's own
    parameter under the family. Budgets are left out, as they are from what makes trained work
    comparable. A family one side does not switch on is compared as the whole configuration,
    which is how ``comparability_id`` treats a name it cannot place.
    """
    def narrowed(static):
        return config_payload(static, budgets=False, method=method if method in family_keys(static) else None)
    old, new = narrowed(before), narrowed(after)
    out = []
    for key in [*new, *(key for key in old if key not in new)]:
        left, right = old.get(key), new.get(key)
        if _same(left, right):
            continue
        if isinstance(left, dict) and isinstance(right, dict) and key in (family_keys(before) | family_keys(after)):
            out += [f'{key}.{name}' for name in [*right, *(name for name in left if name not in right)]
                    if not _same(left.get(name), right.get(name))]
        else:
            out.append(str(key))
    return out


class Inputs(str):
    """The comparability id of one family, with the other names the same inputs were recorded under.

    It is the id, and compares, hashes and is written as that string. ``also`` holds what a
    trial trained on exactly these inputs may carry instead: the id as it was computed while the
    settings no family learns from still counted (``NOT_LEARNED``). ``names`` asks with both.
    """

    def __new__(cls, value, also=()):
        self = super().__new__(cls, value)
        self.also = frozenset(item for item in also if item and item != value)
        return self


def names(inputs, recorded):
    """Whether ``recorded`` -- the id a trial carries -- names the inputs ``inputs``."""
    return recorded is not None and (recorded == inputs or recorded in getattr(inputs, 'also', ()))


def comparability_id(static_data, full=None, *, method=None):
    """Whether two results may be compared, continued and reused: the inputs.

    Not the code revision or the library versions -- splitting a search at every commit would be
    unusable -- and not the budget: asking for more trials does not invalidate the ones already
    trained, nor the predictions made from them. Both are recorded with the artifact instead.

    ``method`` asks the same question of one family (`config_identity`). Without it, every
    trial of every family became incomparable the moment any family's parameter changed. The
    answer for a family is an ``Inputs``: the id a new trial records, and beside it the names
    the same inputs had under the earlier formula -- this configuration's own, and those the
    model remembers of the configurations it was run under before (``earlier_names``).
    """
    full = full or dependencies(static_data)
    # A name this configuration does not switch on falls back to the whole configuration rather
    # than raising: this is on the training path, and the safe direction for a name we cannot
    # place is to compare *more* strictly, never less. `config_identity` stays strict for
    # callers that mean to narrow deliberately.
    if method is not None and method not in family_keys(static_data):
        method = None

    def under(**formula):
        return dependency_id({'config': config_identity(static_data, budgets=False, method=method, **formula),
                              'datasets': full['datasets'],
                              'partition': full['partition'], 'scalers': full['scalers']})
    if method is None:
        return under()
    inputs = under()
    return Inputs(inputs, {under(not_learned=True), *earlier_names(static_data, method, inputs)})


# ------------------------------------------------------------- the earlier names of a family's inputs
#
# A trial records the id of the inputs it was trained on. Until 17 November 2026 that id also
# hashed settings no family learns from (``NOT_LEARNED``), so a trial trained before carries a
# name that moves when, say, another combiner is tried -- although nothing it was trained on
# did. The name cannot be recomputed once the configuration it was computed from is gone. So
# while that configuration can still be read -- the frozen copy of the run, the configuration
# the model keeps -- the model learns the name and keeps it, filed under the inputs it names:
#
#     <path_model>/.stage-artifacts/identities.json
#         {"version": 1, "families": {"<family>": {"<inputs id>": ["<earlier name>", ...]}}}
#
# Every entry is a fact about two hashes of one configuration, not a judgement about what is
# on disk: it says that the earlier formula called these inputs by that name. Nothing else is
# read from it, and a trial is admitted by it only when the family's inputs -- its own
# parameters, everything it learns from and the data -- are the ones the name was filed under.

#: ``stage_artifacts.DIRECTORY``; this module is imported by that one.
STAGE_FOLDER = '.stage-artifacts'
LEDGER = 'identities.json'
#: How many different inputs of one family the model keeps earlier names for, and how many
#: names for one of them: the most recent. A name that is dropped vouches for nothing any more,
#: which costs a question (a trial that carries it is asked about), never a wrong answer.
LEDGER_DEPTH = 16
LEDGER_NAMES = 64
_learned = {}          # path_model -> {family: {inputs: {names}}}: what this process worked out itself


def ledger_path(static_data):
    return os.path.join(str(static_data.get('path_model') or ''), STAGE_FOLDER, LEDGER)


def _read_ledger(path):
    try:
        with open(path, encoding='utf-8') as stream:
            families = json.load(stream).get('families')
        return families if isinstance(families, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def earlier_names(static_data, method, inputs):
    """The names ``inputs`` of family ``method`` had under the earlier formula, as far as they are known."""
    root = str(static_data.get('path_model') or '')
    if not root:
        return set()
    out = set((_learned.get(root, {}).get(method) or {}).get(inputs, ()))
    kept = (_read_ledger(ledger_path(static_data)).get(method) or {})
    if isinstance(kept, dict) and isinstance(kept.get(inputs), list):
        out.update(str(name) for name in kept[inputs])
    return out


def learn_names(static_data, earlier, *, persist=False):
    """Learn what the earlier formula called this configuration's families in ``earlier`` configurations.

    ``earlier`` are configurations this model was run under, resolved as a run resolves them.
    For each family whose inputs are the same there as here -- its own parameters, everything
    it learns from, and the data as it is now -- the name those inputs had there is kept.
    Returns ``{family: {inputs: [names]}}`` of what was learned.

    It is known to this process from then on. ``persist`` also writes it into the model
    folder, which is what a run does before the configurations it learned from are removed;
    a plan, a preview or a dry run only learns.
    """
    root = str(static_data.get('path_model') or '')
    if not root:
        return {}
    try:
        full = dependencies(static_data)
    except Exception as error:                                   # noqa: BLE001 -- lineage never blocks a run
        log.debug('no earlier names learned for %s (%r)', root, error)
        return {}

    def under(configuration, method, **formula):
        return dependency_id({'config': config_identity(configuration, budgets=False, method=method, **formula),
                              'datasets': full['datasets'],
                              'partition': full['partition'], 'scalers': full['scalers']})
    found = {}
    for method in sorted(family_keys(static_data)):
        inputs = under(static_data, method)
        for configuration in [static_data, *earlier]:
            try:
                if method not in family_keys(configuration) or under(configuration, method) != inputs:
                    continue
                name = under(configuration, method, not_learned=True)
            except Exception as error:                           # noqa: BLE001
                log.debug('an earlier configuration of %s could not be read for %s (%r)', root, method, error)
                continue
            if name != inputs:
                found.setdefault(method, {}).setdefault(inputs, set()).add(name)
    known = _learned.setdefault(root, {})
    for method, entries in found.items():
        for inputs, values in entries.items():
            known.setdefault(method, {}).setdefault(inputs, set()).update(values)
    if persist:
        _keep_names(static_data, found)
    return {method: {inputs: sorted(values) for inputs, values in entries.items()}
            for method, entries in found.items()}


def _keep_names(static_data, found):
    """Add ``found`` to the model's ledger. A model that has no stage folder has trained nothing."""
    path = ledger_path(static_data)
    folder = os.path.dirname(path)
    if not found or not os.path.isdir(folder):
        return
    import fcntl
    try:
        with open(os.path.join(folder, 'identities.lock'), 'a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)       # two runs of one model start side by side
            families = _read_ledger(path)
            before = json.dumps(families, sort_keys=True)
            for method, entries in found.items():
                kept = families.get(method) if isinstance(families.get(method), dict) else {}
                for inputs, values in entries.items():
                    earlier = kept.pop(inputs, [])            # filed again as the newest entry
                    earlier = [str(name) for name in earlier] if isinstance(earlier, list) else []
                    kept[inputs] = [*earlier, *sorted(set(values) - set(earlier))][-LEDGER_NAMES:]
                families[method] = dict(list(kept.items())[-LEDGER_DEPTH:])
            if json.dumps(families, sort_keys=True) == before:
                return
            temporary = f'{path}.{os.getpid()}.tmp'
            with open(temporary, 'w', encoding='utf-8') as stream:
                json.dump({'version': VERSION, 'families': families}, stream, indent=1)
            os.replace(temporary, path)
    except OSError as error:
        log.warning('the earlier names of the families of %s were not kept (%r); nothing else is affected',
                    static_data.get('path_model'), error)


def partition_identity(static_data):
    """The partition the masks on disk describe, and the policy that drew it."""
    from eforecast.datasets.files_manager import FilesManager
    from eforecast.datasets.data_preprocessing.data_split import partition_fingerprint

    try:
        cv_mask = FilesManager(static_data, train=True).check_if_exists_cv_data()
        return partition_fingerprint(static_data, cv_mask) if cv_mask else None
    except Exception as error:                                   # noqa: BLE001 -- lineage never blocks training
        log.debug('no partition identity (%r)', error)
        return None


def canonical(value):
    """A stable form for things JSON has no form for.

    ``str`` is not good enough for two of them. A set prints in its iteration order, which is not
    the same for two sets holding the same values -- so the identity of one configuration could
    differ between a process that built it and a process that unpickled it. And a large array
    prints abbreviated, so two different arrays can print the same.
    """
    if isinstance(value, (set, frozenset)):
        try:
            return ['set', sorted(value)]
        except TypeError:
            return ['set', sorted(str(item) for item in value)]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def dependency_id(dependencies):
    payload = json.dumps(dependencies, sort_keys=True, separators=(',', ':'), default=canonical)
    return hashlib.sha256(payload.encode()).hexdigest()


def dependencies(static_data, extra=None, *, scope=None, selectors=True, raw_pixels=False, method=None):
    """What this artifact was made from, as identities, plus one hash over all of them."""
    from eforecast.datasets.files_manager import FilesManager

    out = {'version': VERSION, 'code': code_identity(), 'config': config_identity(static_data),
           'config_revision': static_data.get('config_revision'),
           'partition': partition_identity(static_data), 'datasets': {}, 'scalers': None}
    try:
        out['datasets'] = dataset_identities(FilesManager(static_data, train=True))
    except Exception as error:                                   # noqa: BLE001
        log.debug('no dataset identities (%r)', error)
    scalers = os.path.join(static_data.get('path_model', ''), 'scalers.pickle')
    out['scalers'] = file_identity(scalers)
    out['transformers'] = file_identity(os.path.join(static_data.get('path_model', ''), 'transformers.pickle'))
    # Stage publication hashes these files once. Trials read the same cheap sidecar contract;
    # new manifests retain cluster membership and selector identities as well as global inputs.
    # The older comparability formula is intentionally unchanged for stored study budgets.
    root = static_data.get('path_model', '')
    scopes = [scope] if scope is not None else [os.path.join(root, 'global')]
    mapping = os.path.join(root, 'clusters.pickle')
    if scope is None and static_data.get('is_Fuzzy') and os.path.isfile(mapping):
        from . import trial_layout          # the folders of this tree, also when the file names another's
        scopes.extend(trial_layout.cluster_folders(static_data).values())
    out['memberships'] = {}
    out['selectors'] = {}
    memberships = ([mapping] if scope is None else []) + [os.path.join(p, 'cv_mask.pickle') for p in scopes]
    selected = (([os.path.join(root, 'feature_selectors.pickle')] if scope is None else []) +
                [os.path.join(p, 'feature_selectors.pickle') for p in scopes]) if selectors else []
    for section, paths in (('memberships', memberships), ('selectors', selected)):
        for path in paths:
            identity = file_identity(path)
            if identity is not None:
                out[section][os.path.relpath(path, root)] = identity
    if raw_pixels:
        from .image_assets import identities
        out['image_pixels'] = identities(static_data)
    out.update(extra or {})
    out['dependency_id'] = dependency_id({k: v for k, v in out.items() if k != 'dependency_id'})
    # what may be compared with what, beside the whole provenance: the inputs without the budget.
    # `method` narrows it to one family, so a trial records the id its own family is judged by.
    out['comparability_id'] = str(comparability_id(static_data, full=out, method=method))
    # Whose id that is: the family's when it could be narrowed to one, else nobody's in
    # particular. A reader that finds no name here is looking at the id of the whole case.
    out['comparability_scope'] = method if method is not None and method in family_keys(static_data) else None
    return out


def input_observations(static_data, scope, *, raw_pixels=False):
    """Cheap before/after fitting guard, including ctime and replacement identity.

    Dataset content is hashed by its stage. A trial never hashes gigabytes to guard a fit;
    ordinary writes, replacements and restored mtimes still change this observation.

    Each file is a ``file_stamps`` stamp. Compare two of these with ``inputs_changed``, never
    with ``!=``: a file whose change time moved because a sync client marked it as uploaded is
    the same input, and a plain comparison ended the fit it was guarding.
    """
    missing = [key for key in ('path_model', 'path_data') if not static_data.get(key)]
    if missing:
        return {'status': 'unknown', 'missing': missing}  # legacy/custom producers have no invented input binding
    from eforecast.datasets.files_manager import FilesManager
    manager = FilesManager(static_data, train=True)
    paths = [*manager.dataset_files().values(), manager.file_cv_data(),
             os.path.join(static_data['path_model'], 'scalers.pickle'),
             os.path.join(static_data['path_model'], 'transformers.pickle'),
             os.path.join(scope, 'cv_mask.pickle'), os.path.join(scope, 'feature_selectors.pickle')]
    from . import file_stamps
    result = {path: file_stamps.observe(path) for path in paths}
    if raw_pixels:
        from .image_assets import stamps
        result.update(stamps(static_data))
    return result


def inputs_changed(before, after):
    """The observed inputs of a fit that are not the same file any more; empty when none moved."""
    from . import file_stamps
    return file_stamps.differing(before, after)
