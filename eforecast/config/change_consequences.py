# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""What saving a change to a case does to what the case has already built.

A case builds one model, and that model is the model of one configuration. So a setting that
is changed after something was built from it is not a note for later: the datasets that were
made from the old variables have to be made again, and a family whose parameters or search
space changed has to be trained again, before the model is this configuration's once more.

Nothing here decides that. The receipts decide it for the datasets and the studies for the
families, exactly as they do when a run is planned (``stage_resume``, ``revisions``). This
module asks them twice -- for the case as it is saved and for the case as it would be saved
-- and says what the change adds: which datasets are built again, which families are trained
from the beginning, which settings were changed. That is what a person reads, and accepts,
before the change is saved (``api``): the saving itself replaces nothing, the next run does,
and it is the save that makes the next run do it.

It reads. It resolves two configurations and inspects receipts and study journals; it writes
no file of the case or of the model, and it starts nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import tempfile
from typing import Any

log = logging.getLogger(__name__)

VERSION = 1


def _options(script, *argv):
    from . import run_inputs
    return run_inputs.options_for(script, ['--config', 'case', *argv])


def static_of(source, document):
    """The configuration a run of this document would execute from, resolved as a run resolves it."""
    from . import run_inputs
    from .config_files import save_document
    with tempfile.TemporaryDirectory(prefix='eforecast-consequences-') as temporary:
        folder = Path(temporary) / (Path(source).name if Path(source).suffix == '' else Path(source).stem)
        save_document(document, folder)
        return run_inputs.resolve(folder, Path(source), document)


def _inputs(source):
    """Where the frozen configurations of this case's runs are kept."""
    from .config_files import config_paths
    try:
        return config_paths(Path(source))[1] / 'runs' / 'inputs'
    except Exception:                                           # noqa: BLE001
        return None


def built(static, inputs=None, also=()) -> dict[str, Any]:
    """What a run of each step would replace in the model folder, were this the case's configuration.

    ``datasets``: the dataset stages built again because their settings changed, each with how
    many files it replaces. ``families``: the families trained from the beginning, with how
    many studies and trial folders that removes. ``results``: the steps of the combination
    that are computed again. ``unresolved`` names what could not be asked.

    ``inputs`` is where the frozen configurations of the case's runs are, and ``also`` further
    configurations of the case -- the saved one, when a draft is asked about. The earlier
    names of the families' inputs are learned from them first, as a run learns them, so that
    a setting no family learns from is not mistaken for a reason to train one again.
    """
    from eforecast.init.initialize import initializer
    from . import model_configuration, stage_resume
    from .stage_catalogue import resolve
    prepared = initializer(static, dry_run=True)
    out = {'path_model': prepared['path_model'], 'exists': Path(prepared['path_model']).is_dir(),
           'path_group_nwp': prepared.get('path_group_nwp'),
           'datasets': [], 'extract': None, 'families': {}, 'results': [], 'unresolved': []}
    if not out['exists']:
        return out
    model_configuration.learn(prepared, inputs)
    if also:
        try:
            from eforecast.common_utils import lineage
            lineage.learn_names(prepared, [initializer(other, dry_run=True) for other in also])
        except Exception as error:                              # noqa: BLE001 -- then it is asked the long way
            log.debug('nothing learned from the other configurations of %s (%r)', out['path_model'], error)
    try:
        report = stage_resume.decide(prepared, resolve(prepared, 'datasets', _options('datasets', '--stage', 'all')),
                                     action='continue')
        out['datasets'] = (stage_resume.changes(report) or {}).get('stages', [])
        for item in report['stages']:
            if item['key'] == 'extract' and item['decision'] == 'rebuild' and item.get('cause') == 'configuration':
                out['extract'] = {'kept': item.get('kept', 0)}
    except Exception as error:                                  # noqa: BLE001 -- said, not raised
        out['unresolved'].append(f'datasets: {type(error).__name__}: {error}')
    try:
        from eforecast.training.command_line import revisions
        for argv in ((), ('--cpu',)):
            units = revisions.plan(prepared, action='continue', options=_options('training', *argv))
            for method, entry in revisions.retrained(units).items():
                held = out['families'].setdefault(method, {'units': 0, 'trials': 0, 'clusters': []})
                held['units'] += entry['units']
                held['trials'] += entry['trials']
                held['clusters'] += entry['clusters']
    except Exception as error:                                  # noqa: BLE001
        out['unresolved'].append(f'training: {type(error).__name__}: {error}')
    try:
        report = stage_resume.decide(prepared, resolve(prepared, 'combine', _options('combine')), action='continue')
        out['results'] = [item['key'] for item in report['stages']
                          if item['decision'] == 'rebuild' and item.get('cause') == 'configuration'
                          and item.get('affected')]
    except Exception as error:                                  # noqa: BLE001
        out['unresolved'].append(f'combine: {type(error).__name__}: {error}')
    return out


def _same_work(then, now):
    """Whether two configurations build the same things in the same place.

    Everything a dataset stage or a family is built from is part of the configuration's
    identity without its budgets, so two configurations that share it -- and the model folder
    -- are told the same by every receipt and every study.
    """
    from eforecast.common_utils import lineage
    from eforecast.init.initialize import initializer
    try:
        one, other = initializer(then, dry_run=True), initializer(now, dry_run=True)
        return (one['path_model'] == other['path_model']
                and one.get('path_group_nwp') == other.get('path_group_nwp')
                and lineage.config_identity(one, budgets=False) == lineage.config_identity(other, budgets=False))
    except Exception:                                           # noqa: BLE001 -- then ask the long way
        return False


def _because(before, after, stage):
    """The settings of a stage that differ between the two configurations, by name."""
    try:
        from .stage_artifacts import changed_settings
        return changed_settings(before, after, stage)
    except Exception:                                           # noqa: BLE001
        return []


def _family_because(before, after, method):
    try:
        from eforecast.common_utils import lineage
        return lineage.config_differences(before, after, method=method)
    except Exception:                                           # noqa: BLE001
        return []


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def report(source, saved, draft, *, case=None) -> dict[str, Any]:
    """What saving ``draft`` in place of ``saved`` would do to what the case has built.

    ``consequential`` is true when the change makes a later run replace datasets or trained
    models that exist: that is what has to be accepted with the save, and ``digest`` names
    exactly what was accepted. Work that was already waiting to be done again before this
    change is listed apart, as ``pending``, and is not asked about a second time.
    """
    from .document_changes import differences, settings_changed
    out: dict[str, Any] = {
        'version': VERSION, 'case': case, 'status': 'resolved', 'reason': None,
        # The settings that change, as the case is read: a default that the draft merely
        # writes down is none. ``as_written`` counts the places where the two files differ,
        # which is what an editor means by "unsaved changes".
        'changes': settings_changed(saved, draft), 'as_written': len(differences(saved or {}, draft or {})),
        'datasets': [], 'extract': None, 'families': [], 'results': [], 'relocated': None, 'weather': None,
        'pending': {'datasets': [], 'families': []}, 'restored': {'datasets': [], 'families': []},
        'unresolved': [], 'consequential': False, 'digest': None}
    if not saved:
        out.update(status='new', reason='nothing is saved under this name yet, so nothing was built from it')
        return out
    if not out['changes']:
        # The saved case itself. Nothing is asked about, but whether what is built is this
        # configuration's is still worth saying: a change saved earlier may be waiting for the
        # runs that build the datasets again and train the families from the beginning.
        out.update(status='unchanged', reason='the draft is the saved case' if not out['as_written'] else
                   'the draft writes down values the case already has; no setting changes')
        try:
            waiting = built(static_of(source, saved), _inputs(source))
            out['path_model'] = waiting['path_model']
            out['unresolved'] = waiting['unresolved']
            out['pending']['datasets'] = [{'stage': item['key'], 'files': item['files'], 'because': []}
                                          for item in waiting['datasets']]
            out['pending']['families'] = [{'method': method, 'studies': held['units'], 'trials': held['trials'],
                                           'because': []} for method, held in waiting['families'].items()]
        except Exception as error:                              # noqa: BLE001
            out['unresolved'] = [f'{type(error).__name__}: {error}']
        return out
    try:
        then = static_of(source, saved)
    except Exception as error:                                  # noqa: BLE001
        out.update(status='unresolved', reason=f'the saved case does not resolve: {type(error).__name__}: {error}')
        return out
    try:
        now = static_of(source, draft)
    except Exception as error:                                  # noqa: BLE001
        # A draft that is not a case yet is refused by the save itself, with its own errors.
        out.update(status='unresolved', reason=f'the draft does not resolve: {type(error).__name__}: {error}')
        return out
    if _same_work(then, now):
        # A budget, a path, how many processes: nothing a dataset or a trial is built from.
        # The receipts and the studies would be asked the same question twice.
        out['digest'] = digest({'version': VERSION, 'unchanged': True})
        return out
    import copy
    before = built(copy.deepcopy(then), _inputs(source))
    after = built(copy.deepcopy(now), _inputs(source), also=[copy.deepcopy(then)])
    out['unresolved'] = after['unresolved']
    out['path_model'] = after['path_model']
    if after['path_model'] != before['path_model']:
        # Another name, group, owner, type or version: the draft is the case of another model
        # folder. What was built stays where it is, and is no longer this case's.
        if before['exists']:
            out['relocated'] = {'from': before['path_model'], 'to': after['path_model'],
                                'holds_work': after['exists']}
    waiting = {item['key'] for item in before['datasets']}
    for item in after['datasets']:
        entry = {'stage': item['key'], 'files': item['files'], 'because': _because(then, now, item['key'])}
        (out['pending']['datasets'] if item['key'] in waiting else out['datasets']).append(entry)
    wanted = {item['key'] for item in after['datasets']}
    out['restored']['datasets'] = sorted(waiting - wanted)
    if after['extract'] and not before['extract']:
        out['extract'] = after['extract']
    if (after['path_group_nwp'] and before['path_group_nwp'] and after['path_group_nwp'] != before['path_group_nwp']
            and after['path_model'] == before['path_model']):
        folder = Path(after['path_group_nwp'])
        out['weather'] = {'from': before['path_group_nwp'], 'to': after['path_group_nwp'],
                          'extracted': folder.is_dir() and any(folder.iterdir())}
    for method, held in after['families'].items():
        entry = {'method': method, 'studies': held['units'], 'trials': held['trials'],
                 'because': _family_because(then, now, method)}
        (out['pending']['families'] if method in before['families'] else out['families']).append(entry)
    out['restored']['families'] = sorted(set(before['families']) - set(after['families']))
    out['results'] = [key for key in after['results'] if key not in before['results']]
    out['consequential'] = bool(out['datasets'] or out['families'] or out['relocated'])
    out['digest'] = digest({'version': VERSION, 'path_model': after['path_model'],
                            'datasets': sorted(item['stage'] for item in out['datasets']),
                            'families': sorted(item['method'] for item in out['families']),
                            'relocated': bool(out['relocated'])})
    return out


def describe(found) -> str:
    """A report as the lines a person reads before saving."""
    from .document_changes import describe as line
    if found['status'] == 'unchanged':
        waiting = [item['stage'] for item in found['pending']['datasets']] + \
                  [item['method'] for item in found['pending']['families']]
        return ('what is built is not this configuration\'s yet; waiting to be done again: ' + ', '.join(waiting)
                if waiting else 'everything that is built was built from this configuration')
    if found['status'] != 'resolved':
        return f"consequences {found['status']}: {found['reason']}"
    lines = [f"{len(found['changes'])} setting(s) changed:"]
    lines += [f'  {line(change)}' for change in found['changes'][:20]]
    if found['relocated']:
        lines.append(f"the case becomes the case of another model folder: {found['relocated']['to']} "
                     f"(what is built stays in {found['relocated']['from']})")
    if found['datasets']:
        lines.append('datasets built again at the next datasets run:')
        lines += [f"  {item['stage']} ({item['files']} file(s)){': ' + ', '.join(item['because']) if item['because'] else ''}"
                  for item in found['datasets']]
    if found['families']:
        lines.append('families trained from the beginning at the next training run:')
        lines += [f"  {item['method']} ({item['studies']} studies, {item['trials']} trial folder(s) removed)"
                  f"{': ' + ', '.join(item['because']) if item['because'] else ''}" for item in found['families']]
    if found['results']:
        lines.append('results computed again at the next combine run: ' + ', '.join(found['results']))
    if not found['consequential']:
        lines.append('nothing that is built has to be built or trained again for this change')
    return '\n'.join(lines)
