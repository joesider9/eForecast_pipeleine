# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The configuration a model was built from: one copy, kept with the model.

A run freezes its configuration before it starts (``run_inputs``), because a case that is
edited under a run must not change what that run computes. Those copies used to stay, one for
every run that was ever started, and none of them said which one the model on disk was made
from. A case has one model, so it keeps one configuration: that of the run which last finished
a stage in it. It lies in the model folder, beside the receipts that say what each stage made,
and it goes with the model when the folder is copied, moved or backed up::

    <path_model>/.stage-artifacts/configuration/
        record.json             which run recorded it, when, and for which stages
        effective_config.json   the configuration as that run resolved it (``run_inputs``' form)
        case/                   the case's own files as they were: the part files and main.yaml

It decides nothing: receipts and trial records decide what may be reused, never this copy.
What it adds is the sentence those cannot give. "The nwp stage is rebuilt because its
configuration changed" becomes *which setting* changed since the model was built (``since``).

One more thing is read from it, and from the frozen copies of the case's runs before they are
removed (``earlier_configurations``, ``learn``): what the fingerprint in use until 17 November
2026 called each family's inputs. That name hashed settings no family learns from, so it can
only be worked out from a configuration that still has their old values
(``lineage.learn_names``). What is learned is a fact about two hashes of one configuration;
which trials are a family's own is still said by the trials.

A configuration takes the place of the one before only when the model is wholly its own
(``whole``): every dataset stage that left a receipt was built from it, and no family that is
switched on holds trials of another one. A change is worked off in steps -- the datasets are
built again, then each family is trained again -- and all the way through, the copy that is
kept is the last configuration the whole model was built from. So "what changed since it was
built" still names the setting behind the families that are waiting when the datasets are
already done.

Recording never fails a stage. A model whose configuration could not be written is exactly as
good as it was; the next stage records it again.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import time

log = logging.getLogger(__name__)

VERSION = 1
FOLDER = 'configuration'
RECORD = 'record.json'
PAYLOAD = 'effective_config.json'
CASE = 'case'
PREVIOUS = '.previous'
#: What a case's own configuration files are called (``config_files``), and a legacy package's.
SOURCES = ('.yaml', '.yml', '.py')


def directory(static_or_path):
    """Where the configuration of this model is kept."""
    from .stage_artifacts import DIRECTORY
    root = static_or_path['path_model'] if isinstance(static_or_path, dict) else static_or_path
    return Path(root) / DIRECTORY / FOLDER


def _atomic_json(path, value):
    temporary = Path(f'{path}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=1, default=str))
    os.replace(temporary, path)


def _sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _present(folder):
    """The folder that holds a whole record: the current one, or the one a killed swap set aside."""
    for candidate in (folder, Path(str(folder) + PREVIOUS)):
        if (candidate / RECORD).is_file():
            return candidate
    return None


def record(static, script, stage):
    """Keep the configuration this run executes from as the model's own. Returns its record.

    Called when a stage has finished, or was found finished and reused: either way the model
    now holds that stage as this configuration describes it.
    """
    try:
        return _record(static, script, stage)
    except Exception as error:                                   # noqa: BLE001 -- see the module text
        log.warning('the configuration of the model in %s was not recorded (%r); the stage is unaffected',
                    static.get('path_model'), error)
        return None


def _record(static, script, stage):
    from . import run_inputs
    target = directory(static)
    if not target.parent.is_dir():
        return None                              # a model that has no stage folder has no stages
    now = time.time()
    snapshot = run_inputs.executing()
    folder = Path(snapshot['folder']) if snapshot and snapshot.get('folder') else None
    if folder is not None and not folder.is_dir():
        folder = None
    payload = None
    if folder is not None and (folder.parent / run_inputs.PAYLOAD).is_file():
        payload = (folder.parent / run_inputs.PAYLOAD).read_bytes()
    if payload is None:
        # No frozen copy to take: a stage called in-process, or a run whose inputs could not be
        # pinned. The configuration it ran with is still what the model was made from.
        try:
            plain = {key: value for key, value in static.items() if not str(key).startswith('_')}
            payload = run_inputs._json(run_inputs.encode(plain)).encode()
        except TypeError:
            payload = None
    entry = {'version': VERSION, 'written': now, 'project': static.get('_id') or static.get('project_name'),
             'source': (snapshot or {}).get('source'), 'raw_revision': (snapshot or {}).get('raw_revision'),
             'effective_sha256': _sha(payload) if payload is not None else None,
             'case_files': folder is not None, 'recorded_by': {'script': script, 'stage': stage},
             'stages': {stage: now}}
    held = _present(target)
    if held is not None:
        try:
            before = json.loads((held / RECORD).read_text())
        except (OSError, ValueError):
            before = {}
        same = (before.get('version') == VERSION and before.get('effective_sha256') == entry['effective_sha256']
                and before.get('raw_revision') == entry['raw_revision']
                and before.get('case_files') == entry['case_files'] and held == target)
        if same:
            # The same configuration finished another stage: say so, and copy nothing again.
            entry['stages'] = {**(before.get('stages') or {}), stage: now}
            entry['written'] = before.get('written', now)
            entry['source'] = entry['source'] or before.get('source')
            _atomic_json(target / RECORD, entry)
            return entry
    if held is not None:
        # The configuration on record is about to be judged against this one, and may be
        # replaced by it: what it called each family's inputs is kept first.
        remember(static)
    if not whole(static):
        # Part of the model is still of the configuration before: that one stays the model's,
        # until the rest has been built again from this one.
        log.debug('the model in %s is not wholly of this configuration yet; the one on record stays',
                  static.get('path_model'))
        return None
    staging = Path(f'{target}.{os.getpid()}.tmp')
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        if payload is not None:
            (staging / PAYLOAD).write_bytes(payload)
        if folder is not None:
            (staging / CASE).mkdir()
            for item in sorted(folder.iterdir()):
                if item.is_file() and item.suffix in SOURCES:
                    shutil.copy2(item, staging / CASE / item.name)
        _atomic_json(staging / RECORD, entry)
        aside = Path(str(target) + PREVIOUS)
        shutil.rmtree(aside, ignore_errors=True)
        if target.exists():
            os.replace(target, aside)
        os.replace(staging, target)
        shutil.rmtree(aside, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log.info('the configuration of this model is the one of this run (%s)', target)
    return entry


def whole(static):
    """Whether everything the model holds was built from this configuration.

    Every dataset stage this configuration has, and that left a receipt, was built from its
    settings; and no family it switches on holds finished trials of another configuration.
    What was never built is not held against it, and neither is a stage it does not have.
    A model that cannot be asked is not said to be whole.
    """
    from . import stage_artifacts as artifacts
    from .pipeline_stages import ALL_DATASET_STAGES
    from .stage_catalogue import resolve
    try:
        from .run_inputs import options_for
        plan = resolve(static, 'datasets', options_for('datasets', ['--config', 'case', '--stage', 'all']),
                       inspect=False)
        enabled = {node['key'] for node in plan['nodes'] if node['enabled']}
        for key in ALL_DATASET_STAGES:
            receipt = artifacts.receipt_path(static, key)
            if key not in enabled or not receipt.is_file():
                continue
            if json.loads(receipt.read_text()).get('configuration') != artifacts.configuration(static, key):
                return False
        from eforecast.training.command_line import revisions
        return not revisions.retrained(revisions.plan(static, action='continue'))
    except Exception as error:                                  # noqa: BLE001
        log.debug('whether the model in %s is wholly of this configuration cannot be told (%r)',
                  static.get('path_model'), error)
        return False


def earlier_configurations(static, inputs=None, limit=32):
    """The configurations this model was run under that can still be read, as a run resolves them.

    The one the model keeps, and the frozen copies of the case's runs under ``inputs``
    (``<case>/runs/inputs``) for as long as they are there. Only those of this model folder,
    each configuration once. They are what the earlier names of a family's inputs are learned
    from (``lineage.learn_names``) before the copies are removed.
    """
    import contextlib
    import io
    from eforecast.init.initialize import initializer
    from . import run_inputs
    root = str(static.get('path_model'))
    seen, out = set(), []

    def add(payload, load):
        digest = _sha(payload)
        if digest in seen:
            return
        seen.add(digest)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                prepared = initializer(load(), dry_run=True)
        except Exception as error:                              # noqa: BLE001 -- one that cannot be read teaches nothing
            log.debug('an earlier configuration of %s cannot be resolved (%r)', root, error)
            return
        if str(prepared.get('path_model')) == root:
            out.append(prepared)
    kept = _present(directory(static))
    if kept is not None and (kept / PAYLOAD).is_file():
        try:
            add((kept / PAYLOAD).read_bytes(),
                lambda: run_inputs.decode(json.loads((kept / PAYLOAD).read_text())))
        except OSError:
            pass
    for folder in (run_inputs.copies(inputs) if inputs else [])[:limit]:
        payload = folder.parent / run_inputs.PAYLOAD
        try:
            if payload.is_file():
                add(payload.read_bytes(), lambda: run_inputs.load(folder))
        except OSError:
            continue
    return out


def learn(static, inputs=None, *, persist=False):
    """Learn the earlier names of this configuration's families from what can still be read.

    A plan, a preview and a dry run learn for themselves. A run that is about to remove the
    frozen copies of the runs before it, or to replace the configuration the model keeps,
    calls ``remember`` first, which also writes what was learned into the model folder. Never
    raises: a model that cannot be asked keeps the names it has.
    """
    from eforecast.common_utils import lineage
    try:
        if 'path_model' not in static:
            # A configuration that has not been given its folders yet: resolve a copy, as a
            # run resolves it, and leave the caller's as it is.
            import contextlib
            import copy
            import io
            from eforecast.init.initialize import initializer
            with contextlib.redirect_stdout(io.StringIO()):
                static = initializer(copy.deepcopy(static), dry_run=True)
        if not Path(static['path_model']).is_dir():
            return {}
        return lineage.learn_names(static, earlier_configurations(static, inputs), persist=persist)
    except Exception as error:                                  # noqa: BLE001
        log.warning('the earlier names of the families of %s were not learned (%r); nothing else is affected',
                    static.get('path_model') or static.get('project_name'), error)
        return {}


def remember(static, inputs=None):
    """``learn``, and keep what was learned with the model."""
    return learn(static, inputs, persist=True)


def read(path_model):
    """``{'record', 'document', 'static_data', 'folder'}`` of the model's configuration, or None.

    ``document`` is the case as it was written (None when the run had no frozen copy of it) and
    ``static_data`` the configuration as the run resolved it. Either can be missing; a model
    built before this was kept has neither, and then this returns None.
    """
    folder = _present(directory(path_model))
    if folder is None:
        return None
    try:
        entry = json.loads((folder / RECORD).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or entry.get('version') != VERSION:
        return None
    document = static = None
    if (folder / CASE).is_dir():
        try:
            from .config_files import existing_parts, load_document, read_yaml
            case = folder / CASE
            if existing_parts(case) or (case / 'main.yaml').is_file():
                document = load_document(case)
            else:
                single = sorted(path for path in case.iterdir() if path.suffix in ('.yaml', '.yml'))
                document = read_yaml(single[0]) if single else None
        except Exception as error:                              # noqa: BLE001 -- a reference, not an input
            log.debug('the case files of %s cannot be read (%r)', folder, error)
    if (folder / PAYLOAD).is_file():
        try:
            from .run_inputs import decode
            static = decode(json.loads((folder / PAYLOAD).read_text()))
        except Exception as error:                              # noqa: BLE001
            log.debug('the resolved configuration of %s cannot be read (%r)', folder, error)
    return {'record': entry, 'document': document or None, 'static_data': static, 'folder': str(folder)}


def since(path_model, document):
    """What a case document changes in the configuration its model was built from.

    ``{'record', 'changes'}``, or None when the model has no configuration on record or none
    that can be compared with a document. ``changes`` is ``document_changes.settings_changed``:
    the settings as they are read, so a default that was written down since is no change.
    """
    held = read(path_model)
    if held is None or held['document'] is None or not document:
        return None
    from .document_changes import settings_changed
    return {'record': held['record'], 'changes': settings_changed(held['document'], document)}
