# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Versioned, runnable configuration snapshots, separate from archived raw case files.

Experiment code is evaluated at submission against its original location. Its resolved
definitions (including values obtained from helpers) become data: consumers never import that
code again. This pins configuration, not datasets, installed libraries or worker attempt paths.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import uuid
from zoneinfo import ZoneInfo

from .case_revision import revision
from .config_files import config_paths, load_document

log = logging.getLogger(__name__)
VERSION = 1
MARKER = '.eforecast-inputs.json'
PAYLOAD = 'effective_config.json'
REPO_ROOT = Path(__file__).resolve().parents[2]
_parser_warmup_lock = threading.Lock()
_parser_warmup = None


#: The file a process holds a lock on for as long as it runs from a frozen copy.
IN_USE = '.in-use'
#: A copy taken by a command line is a folder named by a random 32-digit hex; one taken for a
#: run the service started is named by that run's id, and the service knows when it has ended.
OWN_COPY = re.compile(r'[0-9a-f]{32}')
_executing = None
_held = []


def executing():
    """The frozen configuration this process runs from, or None.

    ``{'folder', 'source', 'raw_revision', 'effective_revision'}``: the case folder inside the
    copy, the case it was taken from, and the two revisions its marker records. It is what the
    model keeps as its own configuration when a stage finishes (``model_configuration``).
    """
    return _executing


def _run_from(folder, metadata, source=None):
    global _executing
    folder = Path(folder)
    _executing = {'folder': str(folder), 'source': str(source or metadata.get('source') or '') or None,
                  'raw_revision': metadata.get('raw_revision'),
                  'effective_revision': metadata.get('effective_revision')}
    # A process runs from one frozen copy at a time. One that starts another run -- a script
    # that calls two entry points in turn -- has ended the run before it, and lets its copy go.
    while _held:
        _held.pop().close()
    hold(folder.parent)


def hold(root):
    """Keep a frozen copy from being pruned for as long as this process lives.

    A shared lock on a file inside it, which the kernel lets go when the process ends, however
    it ends. ``prune`` leaves alone whatever is held; nothing else reads the lock.
    """
    import fcntl
    try:
        stream = open(Path(root) / IN_USE, 'a')
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
    except OSError as error:
        log.debug('the frozen configuration %s is not held (%r)', root, error)
        return
    _held.append(stream)


def in_use(root):
    """Whether a live process runs from this frozen copy."""
    import fcntl
    path = Path(root) / IN_USE
    if not path.is_file():
        return False
    try:
        with open(path, 'a') as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            return False
    except OSError:
        return True                     # it cannot be told: leave it


def copies(inputs):
    """The case folders of every frozen configuration under ``inputs``, the newest first.

    Both shapes a case holds: ``<id>/<case>`` of a command-line run and
    ``<run id>/<token>/<case>`` of a run the service started. The resolved configuration of
    each lies beside its case folder (``PAYLOAD``).
    """
    inputs = Path(inputs)
    found = []
    if inputs.is_dir():
        for pattern in (f'*/*/{MARKER}', f'*/*/*/{MARKER}'):
            for marker in inputs.glob(pattern):
                try:
                    found.append((marker.stat().st_mtime, str(marker.parent)))
                except OSError:
                    continue
    return [Path(folder) for _, folder in sorted(found, reverse=True)]


def prune(inputs, keep=()):
    """Remove the frozen copies of command-line runs that have ended; returns how many went.

    A run freezes its configuration so that an edit cannot reach it while it works. Once it
    has ended the copy is nobody's input any more. What the model was built from is kept
    once, with the model (``model_configuration``), so a case no longer collects one copy
    for every run that was ever started.

    Only folders this module names itself are touched (``OWN_COPY``), never one a live
    process still runs from, and never one in ``keep``. The copies of runs the service
    started are its own to remove, when it knows they have ended (``runs.RunManager``).
    """
    inputs = Path(inputs)
    if not inputs.is_dir():
        return 0
    kept = {Path(item).resolve() for item in keep}
    removed = 0
    for entry in sorted(inputs.iterdir()):
        if not entry.is_dir() or not OWN_COPY.fullmatch(entry.name) or entry.resolve() in kept or in_use(entry):
            continue
        shutil.rmtree(entry, ignore_errors=True)
        removed += not entry.exists()
    if removed:
        log.info('removed the frozen configuration of %d earlier run(s) of this case from %s', removed, inputs)
    return removed


def _parser(script):
    return importlib.import_module('eforecast.config.entrypoint_parsers').PARSERS[script]


def _reset_parser_warmup():
    # A child process cannot inherit a live thread (or a lock it might hold).
    global _parser_warmup_lock, _parser_warmup
    _parser_warmup_lock = threading.Lock()
    _parser_warmup = None


def _finish_parser_warmup_before_fork():
    # fork cannot copy an in-progress import owned by a thread that vanishes in the child.
    # This waits only at a fork boundary, never during normal manager construction.
    with _parser_warmup_lock:
        thread = _parser_warmup
    if thread is not None and thread is not threading.current_thread():
        thread.join()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(before=_finish_parser_warmup_before_fork, after_in_child=_reset_parser_warmup)


def warm_parsers():
    """Import shared CLI parsers once per process without delaying manager construction.

    Only modules are warmed: parsing and environment-dependent defaults still happen for
    each request. A failed speculative import never changes normal foreground error handling.
    """
    global _parser_warmup
    with _parser_warmup_lock:
        if _parser_warmup is None:
            def warm():
                for script in ('datasets', 'training', 'combine'):
                    try:
                        _parser(script)
                    except Exception:
                        log.exception('background parser warm-up failed for %s; submission will retry', script)

            thread = threading.Thread(target=warm, name='eforecast-parser-warmup', daemon=True)
            try:
                thread.start()
            except RuntimeError:
                log.exception('cannot start background parser warm-up; submission will import on demand')
                return None
            _parser_warmup = thread
        return _parser_warmup


def encode(value):
    """Preserve runtime containers; do not turn tuples/sets into display strings."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if type(value) is dict:
        return {'type': 'dict', 'items': [[encode(k), encode(v)] for k, v in value.items()]}
    if type(value) in (list, tuple, set, frozenset):
        items = [encode(item) for item in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=_json)
        return {'type': type(value).__name__, 'items': items}
    if type(value) is datetime:
        if value.tzinfo is not None and type(value.tzinfo) not in (timezone, ZoneInfo):
            raise TypeError(f'configuration contains an unsupported timezone: {type(value.tzinfo).__name__}')
        return {'type': 'datetime', 'value': value.isoformat(), 'fold': value.fold,
                'zone': value.tzinfo.key if type(value.tzinfo) is ZoneInfo else None}
    if type(value) is date:
        return {'type': 'date', 'value': value.isoformat()}
    # Legacy configuration packages can supply numpy values, unlike typed YAML.
    import numpy as np
    if type(value) is np.ndarray and not value.dtype.hasobject:
        return {'type': 'ndarray', 'dtype': value.dtype.str, 'shape': list(value.shape),
                'items': encode(value.tolist())}
    if isinstance(value, np.generic):
        return encode(value.item())
    raise TypeError(f'configuration contains an unsupported runtime value: {type(value).__name__}')


def decode(value):
    if not isinstance(value, dict):
        return value
    kind = value['type']
    if kind == 'dict':
        return {decode(k): decode(v) for k, v in value['items']}
    if kind in ('list', 'tuple', 'set', 'frozenset'):
        constructors = {'list': list, 'tuple': tuple, 'set': set, 'frozenset': frozenset}
        return constructors[kind](decode(item) for item in value['items'])
    if kind == 'date':
        return date.fromisoformat(value['value'])
    if kind == 'datetime':
        result = datetime.fromisoformat(value['value'])
        if value.get('zone'):
            result = result.astimezone(ZoneInfo(value['zone']))
        return result.replace(fold=value.get('fold', 0))
    if kind == 'ndarray':
        import numpy as np
        return np.asarray(decode(value['items']), dtype=value['dtype']).reshape(value['shape'])
    raise ValueError(f'unknown configuration value type {kind!r}')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


#: Options of what this pipeline no longer has: throughput profiles, the managed dataset cache
#: with its mapped readers, and the allowances and deadline of a reservation. A client built
#: before they went may still send them, and a run accepted then recorded them; they are
#: dropped wherever options are read, never refused.
RETIRED_OPTIONS = frozenset({'execution_profile', 'profile_resources', 'array_reader', 'cache_max_bytes',
                             'cache_max_entries', 'cache_batch_rows', 'cache_buffer_mb',
                             'cpu_study_ram_mb', 'cpu_study_vram_mb', 'cpu_wait_timeout'})


def current_options(options):
    """``options`` without the ones that name something this pipeline no longer has."""
    return {key: value for key, value in (options or {}).items() if key not in RETIRED_OPTIONS}


#: What a command line carries that is not an option of the work: where the case is.
NOT_OPTIONS = ('config',)


def request_options(script, args, env=None):
    options = current_options({key: value for key, value in vars(args).items() if key not in NOT_OPTIONS})
    if options.get('model_cache_mode') is None:
        options.pop('model_cache_mode', None)
    # The new preview flag must not change an older training/combine snapshot's defaults.
    if script in {'training', 'combine'} and not options.get('dry_run'):
        options.pop('dry_run', None)
    if script == 'training':
        for key in ('cpu_scope', 'cpu_workers', 'cpu_study_threads', 'cpu_study_device'):
            if options.get(key) is None:
                options.pop(key, None)
        env = os.environ if env is None else env
        options['storage'] = options.get('storage') or env.get('EFORECAST_OPTUNA_STORAGE') or None
        options['metrics_port'] = (options['metrics_port'] if options.get('metrics_port') is not None
                                   else int(env.get('EFORECAST_METRICS_PORT', '9108') or 0))
        names = ('EFORECAST_WORKER_IMAGE', 'EFORECAST_WORKER_FOLDERS',
                 'EFORECAST_WORKER_MOUNTS', 'EFORECAST_TRIAL_CMD')
        options['worker_environment'] = {} if options.get('docker') else {key: env.get(key) or None for key in names}
    return options


def equivalent_options(script, left, right):
    """Normalize older optional empty environment/default flags without rewriting snapshots."""
    def normalize(value):
        value = current_options(deepcopy(value))
        if value.get('model_cache_mode') is None:
            value.pop('model_cache_mode', None)
        if script in {'training', 'combine'} and not value.get('dry_run'):
            value.pop('dry_run', None)
        if script == 'training':
            for key in ('cpu_scope', 'cpu_workers', 'cpu_study_threads', 'cpu_study_device'):
                if value.get(key) is None:
                    value.pop(key, None)
        if script == 'training' and isinstance(value.get('worker_environment'), dict):
            value['worker_environment'] = {key: item or None for key, item in value['worker_environment'].items()}
        return value
    return normalize(left) == normalize(right)


def options_for(script, argv, env=None):
    """Use the CLI's shared parser so implicit defaults also belong to the request."""
    args = _parser(script)(argv)
    return request_options(script, args, env)


def apply_overrides(static_data, script, options):
    result = deepcopy(static_data)
    if options.get('model_cache_mode') is not None:
        result['model_cache_mode'] = options['model_cache_mode']
    if script == 'datasets' and options.get('serial'):
        result['n_jobs'] = 1
    if script == 'training' and options.get('cpu_scope'):
        result['cpu_scope'] = options['cpu_scope']
    if script == 'training':
        for key in ('cpu_workers', 'cpu_study_threads', 'cpu_study_device'):
            if options.get(key) is not None:
                result[key] = options[key]
    # Training GPU ids/concurrency and method filters are executor arguments, not
    # replacements for unrelated CPU/method settings. They are pinned in options.
    return result


def identity(static_data, script, options):
    return _sha(_json({'version': VERSION, 'script': script, 'options': options,
                       'static_data': encode(apply_overrides(static_data, script, options))}).encode())


def read(config):
    folder = config_paths(config)[1]
    marker = folder / MARKER
    if not marker.exists():
        if (folder.parent / PAYLOAD).exists():
            raise ValueError(f'invalid run input snapshot at {folder}: snapshot marker is missing')
        return None
    try:
        metadata = json.loads(marker.read_text())
        if (type(metadata.get('version')) is not int or metadata['version'] != VERSION
                or metadata.get('script') not in ('datasets', 'training', 'combine')
                or not isinstance(metadata.get('options'), dict)):
            raise ValueError('unsupported snapshot version or request')
        payload = (folder.parent / PAYLOAD).read_bytes()
        if _sha(payload) != metadata.get('data_sha256'):
            raise ValueError('normalized configuration digest mismatch')
        static_data = decode(json.loads(payload))
        if not isinstance(static_data, dict):
            raise ValueError('normalized configuration must be a mapping')
        if identity(static_data, metadata['script'], metadata['options']) != metadata.get('effective_revision'):
            raise ValueError('effective request digest mismatch')
        if metadata.get('runtime_binding') is not None:
            from eforecast.runtime_registry import validate_binding, selections
            binding = validate_binding(metadata['runtime_binding'])
            if binding['selection'] != selections(static_data):
                raise ValueError('runtime selection differs from the snapshot')
            if binding['selection'] and static_data.get('_runtime_binding') != binding:
                raise ValueError('runtime definition differs from the effective configuration')
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise ValueError(f'invalid run input snapshot at {folder}: {error}') from error
    return metadata, static_data


def load(config):
    frozen = read(config)
    if frozen is None:
        return None
    metadata, static_data = frozen
    return apply_overrides(static_data, metadata['script'], metadata['options'])


def copy_raw(source, folder):
    single, parts = config_paths(source)
    folder.mkdir(parents=True, exist_ok=False)
    if source.suffix == '.py':
        shutil.copy2(source, folder / source.name)
        return
    if parts.is_dir():
        for entry in sorted(parts.iterdir()):
            if entry.is_file() and entry.name != MARKER:
                shutil.copy2(entry, folder / entry.name)
    if single.is_file():
        shutil.copy2(single, folder / 'main.yaml')


def normalize(raw, source):
    """Normalize the archived document with the original experiment lookup anchor."""
    from .loader import to_static_data
    from .schema import ExperimentConfig
    cfg = ExperimentConfig.model_validate(load_document(raw))
    object.__setattr__(cfg, '_yaml_path', str(config_paths(source)[0].resolve()))
    return to_static_data(cfg)


def resolve(raw, source, document, environment=None, runtime_binding=None):
    from eforecast.runtime_registry import bind_static
    if not document.get('experiments_module'):
        return bind_static(normalize(raw, source), environment, binding=runtime_binding)[0]
    # A fresh interpreter avoids serving cached imported experiment/helper modules
    # across submissions. A private, unwritten pycache prefix also avoids stale pyc.
    output = raw.parent / '.resolved.json'
    code = ('import json,sys; from pathlib import Path; '
            'from eforecast.config.run_inputs import encode,normalize; '
            'Path(sys.argv[3]).write_text(json.dumps(encode(normalize(sys.argv[1],sys.argv[2]))))')
    environment = os.environ if environment is None else environment
    env = dict(environment, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), environment.get('PYTHONPATH')])),
               PYTHONPYCACHEPREFIX=str(raw.parent / '.pycache'))
    try:
        child = subprocess.run([sys.executable, '-B', '-c', code, str(raw), str(source), str(output)],
                               cwd=Path.cwd(), env=env, capture_output=True, text=True, timeout=60)
        if child.returncode:
            raise ValueError(f'experiment resolution failed: {child.stderr[-4000:].strip()}')
        return bind_static(decode(json.loads(output.read_text())), environment, binding=runtime_binding)[0]
    finally:
        try:
            output.unlink(missing_ok=True)
        except OSError as error:
            log.warning('cannot remove resolved configuration temporary file %s: %s', output, error)


def capture(source, folder, script, options, lock=None, normalized_base=None, environment=None, accepted_files=None, runtime_binding=None):
    """Archive raw files, then publish runnable data last. Failure keeps legacy execution explicit."""
    source, folder = Path(source), Path(folder)
    if folder.parent.exists():
        raise FileExistsError(f'run input directory already exists: {folder.parent}')
    record = {'config': str(source), 'config_source': str(source), 'config_frozen': False,
              'input_snapshot_status': 'degraded', 'input_snapshot_version': VERSION,
              'effective_options': options}
    try:
        with lock or nullcontext():
            if accepted_files is None:
                copy_raw(source, folder)
            else:
                from .run_submission import restore
                restore(accepted_files, folder)
            document = load_document(folder)
            raw_revision = _sha((folder / source.name).read_bytes()) if source.suffix == '.py' else revision(document)
    except OSError as error:
        shutil.rmtree(folder.parent, ignore_errors=True)
        record['config_error'] = str(error)
        log.warning('cannot archive run inputs for %s: %s; %s', source, error,
                    'managed preparation incomplete' if accepted_files is not None else 'using live configuration')
        return record
    except Exception:
        shutil.rmtree(folder.parent, ignore_errors=True)
        raise
    record.update(config_frozen=True, config_raw=str(folder), config_revision=raw_revision)
    try:
        previous = read(source)
        from eforecast.runtime_registry import bind_static
        accepted_runtime = runtime_binding if runtime_binding is not None else (previous[0].get('runtime_binding') if previous else None)
        static_data = (normalized_base if normalized_base is not None else
                       previous[1] if previous else resolve(folder, source, document, environment, accepted_runtime))
        static_data, accepted_runtime = bind_static(static_data, environment, binding=accepted_runtime)
        if script == 'training' and static_data.get('transfer_learning') and 'static_data_base' not in static_data:
            import joblib
            static_data = deepcopy(static_data)
            static_data['static_data_base'] = joblib.load(static_data['transfer_learning_from']['configuration'])
        encoded = encode(static_data)
        # Prove the serialized form is consumable before publishing its marker.
        decode(encoded)
        payload = _json(encoded).encode()
        effective_revision = identity(static_data, script, options)
        metadata = {'version': VERSION, 'script': script, 'options': options,
                    'raw_revision': record['config_revision'], 'effective_revision': effective_revision,
                    'data_sha256': _sha(payload), 'source': str(source),
                    'experiments_module': document.get('experiments_module'),
                    'experiment_resolution': 'materialized', 'runtime_binding': accepted_runtime}
        (folder.parent / PAYLOAD).write_bytes(payload)
        # Human-readable evidence only; the typed payload above is the executable input.
        (folder.parent / 'static_data.json').write_text(json.dumps(
            apply_overrides(static_data, script, options), indent=1, default=str))
        temporary = folder / (MARKER + '.tmp')
        temporary.write_text(json.dumps(metadata, indent=1))
        os.replace(temporary, folder / MARKER)
        record.update(config=str(folder), input_snapshot_status='pinned', effective_revision=effective_revision,
                      input_snapshot=str(folder / MARKER), runtime_binding=accepted_runtime)
    except Exception as error:
        record['normalized_error'] = f'{type(error).__name__}: {error}'
        if accepted_files is not None:
            record['config'] = str(folder)
        log.warning('cannot pin effective inputs for %s: %s; %s', source, error,
                    'using accepted raw configuration' if accepted_files is not None else 'using live configuration')
    return record


def prepare(config, script, args):
    value = _prepare(config, script, args)
    from eforecast.runtime_registry import execution_binding
    execution_binding(value)
    if not getattr(args, 'dry_run', False):
        from eforecast import model_library
        model_library.use_mode(value)
    return value


def _prepare(config, script, args):
    """CLI counterpart of RunManager capture. Dry-run keeps its existing no-write behavior."""
    options = request_options(script, args)
    for key, value in options.items():
        if hasattr(args, key):
            setattr(args, key, value)
    managed = os.environ.get('EFORECAST_RUN_INPUT_CONFIG')
    if managed and Path(managed).resolve() == Path(config).resolve():
        if os.environ.get('EFORECAST_RUN_INPUT_STATUS') == 'degraded':
            from eforecast.init.initialize import load_static_data
            source = os.environ.get('EFORECAST_RUN_INPUT_SOURCE')
            if source:
                from .run_submission import verify_sources
                guards = json.loads(os.environ.get('EFORECAST_RUN_INPUT_GUARD', '{}'))
                verify_sources(guards)
                loaded = normalize(config, source)
                verify_sources(guards)
                log.warning('DEGRADED run inputs: accepted raw configuration at %s; normalization is not pinned', config)
                return apply_overrides(loaded, script, options)
            log.warning('DEGRADED run inputs: submission uses live configuration at %s', config)
            return apply_overrides(load_static_data(config), script, options)
    previous = read(config)
    if managed and Path(managed).resolve() == Path(config).resolve():
        if not previous or previous[0]['script'] != script or not equivalent_options(script, previous[0]['options'], options):
            raise ValueError('execution options differ from the submitted input snapshot')
    if previous and previous[0]['script'] == script and equivalent_options(script, previous[0]['options'], options):
        log.info('pinned inputs %s (%s)', previous[0]['effective_revision'], config)
        if not options.get('dry_run'):
            _run_from(config_paths(config)[1], previous[0])
        _learn_quietly(previous[1], _inputs_beside(config), persist=not options.get('dry_run'))
        return apply_overrides(previous[1], script, options)
    if options.get('dry_run'):
        from eforecast.init.initialize import load_static_data
        static_data = previous[1] if previous else load_static_data(config)
        log.info('in-memory dry-run inputs %s', identity(static_data, script, options))
        _learn_quietly(static_data, _inputs_beside(config) if previous
                       else config_paths(Path(config).resolve())[1] / 'runs' / 'inputs', persist=False)
        return apply_overrides(static_data, script, options)
    source = Path(config).resolve()
    parts = config_paths(source)[1]
    inputs = parts.parent.parent if previous else parts / 'runs' / 'inputs'
    folder = inputs / uuid.uuid4().hex / parts.name
    record = capture(source, folder, script, options)
    if record['input_snapshot_status'] == 'pinned':
        log.info('pinned inputs %s (%s)', record['effective_revision'], record['input_snapshot'])
        _run_from(folder, read(folder)[0], source=previous[0].get('source') if previous else source)
        loaded = load(record['config'])
        # What the earlier formula called each family's inputs can only be worked out from the
        # configurations themselves, and the copies of the runs before this one are about to go.
        _learn_quietly(loaded, inputs, persist=True)
        _prune_quietly(inputs, keep=[folder.parent, *([parts.parent] if previous else [])])
        return loaded
    from eforecast.init.initialize import load_static_data
    log.warning('DEGRADED run inputs: %s', record.get('normalized_error') or record.get('config_error'))
    return apply_overrides(load_static_data(config), script, options)


def _inputs_beside(config):
    """The folder that holds the frozen copies, seen from one of them (either shape, ``copies``)."""
    folder = config_paths(Path(config).resolve())[1]
    for parent in (folder.parent.parent, folder.parent.parent.parent):
        if parent.name == 'inputs':
            return parent
    return folder.parent.parent


def _learn_quietly(static_data, inputs, *, persist):
    """Learn the earlier names of the families' inputs before anything is decided with them."""
    try:
        from . import model_configuration
        model_configuration.learn(static_data, inputs, persist=persist)
    except Exception as error:                      # noqa: BLE001 -- what is not learned is asked about, not lost
        log.warning('the earlier names of the families were not learned from %s (%r)', inputs, error)


def _prune_quietly(inputs, keep):
    try:
        prune(inputs, keep=keep)
    except OSError as error:                        # a copy that cannot be removed fails no run
        log.warning('earlier frozen configurations in %s were not removed (%s)', inputs, error)


def prepare_legacy(static_data, source, script, args):
    """Materialize a legacy Python config once; keep its original source as raw evidence."""
    from eforecast.runtime_registry import bind_static, execution_binding
    static_data, _ = bind_static(static_data)
    execution_binding(static_data)
    options = request_options(script, args)
    if options.get('dry_run'):
        log.info('in-memory dry-run inputs %s', identity(static_data, script, options))
        return apply_overrides(static_data, script, options)
    source = Path(source).resolve()
    folder = source.parent / 'runs' / 'inputs' / uuid.uuid4().hex / source.parent.name
    record = capture(source, folder, script, options, normalized_base=static_data)
    if record['input_snapshot_status'] == 'pinned':
        log.info('pinned legacy inputs %s (%s)', record['effective_revision'], record['input_snapshot'])
        _run_from(folder, read(folder)[0], source=source)
        loaded = load(record['config'])
        _learn_quietly(loaded, folder.parent.parent, persist=True)
        _prune_quietly(folder.parent.parent, keep=[folder.parent])
        return loaded
    log.warning('DEGRADED run inputs: %s', record.get('normalized_error') or record.get('config_error'))
    return apply_overrides(static_data, script, options)
