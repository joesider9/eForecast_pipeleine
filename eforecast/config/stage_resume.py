# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""What a second run would reuse, verify, rebuild -- and what an interrupted publication left.

A09-06 says which stages a request contains and in what order; A09-07 says whether each stage's
artifacts are still the ones its receipt describes. This module turns those two facts into the
decision a person actually makes before starting a run again:

``reuse``     the artifact verifies and nothing it depends on is being rebuilt; do not recompute;
``resume``    it runs, and it keeps the compatible work it finds -- a training budget continues;
``verify``    the cheap check cannot tell -- its bytes must be hashed before anything may skip it;
``rebuild``   missing, rejected, reconfigured, or downstream of something that is being rebuilt;
``blocked``   an input nobody in this plan produces is missing;
``skip``      the configuration disables this branch.

It reads. It never deletes, never publishes and never changes a case: a rebuild decision names
the artifacts that a rebuild would replace, which is what the owner approves before it happens.
Recovery from an interrupted publication is the one place that may put a file back, and only the
receipt a publication had set aside for exactly that purpose.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from eforecast.common_utils import lineage
from eforecast.config import stage_artifacts as artifacts

VERSION = 1
ACTIONS = ('continue', 'new', 'rebuild')
DECISIONS = ('reuse', 'resume', 'verify', 'rebuild', 'blocked', 'skip')


def _products_of(plan, key):
    return [name for name, product in plan['products'].items() if product.get('producer') == key]


def payloads_of(static, plan, key):
    """The product files of this stage that exist right now, and what could not be resolved."""
    paths, unresolved = [], []
    for product in _products_of(plan, key):
        try:
            paths += [str(path) for path in artifacts.product_paths(static, plan, product, verify=False)]
        except (OSError, KeyError, ValueError, TypeError, AttributeError, IndexError) as error:
            unresolved.append(f'{product}: {type(error).__name__}: {error}')
    return sorted({path for path in paths if Path(path).exists()}), unresolved


def artifacts_of(static, plan, key):
    """Every existing file this stage would replace, and what could not be resolved.

    A rebuild preview is this list, so a product whose paths cannot be resolved is named as
    unresolved rather than quietly counted as nothing.
    """
    paths, unresolved = payloads_of(static, plan, key)
    receipt = artifacts.receipt_path(static, key)
    return sorted({*paths, *([str(receipt)] if receipt.is_file() else [])}), unresolved


def interrupted(static, plan=None):
    """Publications that were cut short: the marker is still there, the receipt is not.

    ``canonical`` says what the bundle on disk is now, judged against the receipt the publication
    set aside before it started replacing files: ``intact`` (nothing it names was changed yet),
    ``changed`` (promotion had begun), or ``unknown`` (nothing to judge it against).
    """
    found = []
    folder = artifacts.directory(static) / 'receipts'
    for path in sorted(folder.glob('*.json')) if folder.is_dir() else []:
        if path.name.endswith('.superseded.json'):
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if value.get('state') != 'publishing':
            continue
        key = value.get('stage') or path.stem
        superseded = artifacts.superseded_path(static, key)
        canonical, reason = 'unknown', 'no earlier receipt was kept; this stage has no previous bundle to compare'
        if superseded.is_file():
            try:
                before = json.loads(superseded.read_text())
                canonical, reason = _still_matches(before)
            except (OSError, ValueError) as error:
                canonical, reason = 'unknown', f'the superseded receipt is unreadable: {error}'
        found.append({'stage': key, 'producer': value.get('producer'), 'canonical': canonical,
                      'reason': reason, 'marker': str(path),
                      'superseded_receipt': str(superseded) if superseded.is_file() else None,
                      'scratch': None,
                      'recoverable': canonical == 'intact'})
    found.extend(_interrupted_in_place(static))
    return found


def _set_aside(static, key):
    """The outputs a rebuild of ``key`` set aside and never restored or let go.

    One run writes a case at a time, and a stage that ends -- in any way it can notice -- puts
    back or removes what it set aside. So whatever is still aside beside a stage's marker was
    left by that stage being killed.
    """
    from eforecast.common_utils import lineage
    candidates = []
    receipt = artifacts.receipt_path(static, key)
    if receipt.is_file():
        try:
            for records in (json.loads(receipt.read_text()).get('outputs') or {}).values():
                for record in records:
                    candidates += [record['path'], record['path'] + lineage.SIDECAR]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            candidates = []
    aside = [Path(str(path) + artifacts.PREVIOUS) for path in candidates]
    if not candidates:                          # no receipt names them: look where datasets live
        model = Path(static['path_model'])
        for folder in {model, Path(static['path_data']), model / 'global'}:
            if folder.is_dir():
                aside += sorted(folder.glob('*' + artifacts.PREVIOUS))
    return [path for path in dict.fromkeys(aside) if path.is_file()]


def _interrupted_in_place(static):
    """Stages that started writing in place and did not finish: their marker is still there."""
    found = []
    folder = artifacts.directory(static)
    for marker in sorted(folder.glob('*' + artifacts.RUNNING)) if folder.is_dir() else []:
        key = marker.name[:-len(artifacts.RUNNING)]
        if artifacts.finished_despite_marker(static, key):
            continue                    # its receipt is recorded: finished, with a little tidying left
        try:
            producer = json.loads(marker.read_text())
        except (OSError, ValueError):
            producer = None
        aside = _set_aside(static, key)
        found.append({'stage': key, 'producer': producer, 'marker': str(marker), 'in_place': True,
                      'canonical': 'previous_kept' if aside else 'unknown',
                      'reason': ('the stage was interrupted while it rebuilt; the outputs it replaced were set '
                                 'aside and can be put back' if aside else
                                 'the stage was interrupted and nothing was set aside: it has no previous '
                                 'outputs to return to'),
                      'superseded_receipt': None, 'scratch': None, 'set_aside': [str(path) for path in aside],
                      'recoverable': bool(aside)})
    return found


def _still_matches(receipt):
    """Does every output the superseded receipt names still have the bytes it recorded?"""
    if (not isinstance(receipt, dict) or receipt.get('version') != artifacts.VERSION or
            receipt.get('state') not in {'complete', 'legacy'} or not isinstance(receipt.get('id'), str)):
        return 'unknown', 'the kept receipt is not a complete publication of this stage'
    outputs = receipt.get('outputs')
    if not isinstance(outputs, dict) or not outputs or not any(records for records in outputs.values()):
        # An empty payload contract proves nothing: there is no bundle to recognise.
        return 'unknown', 'the kept receipt names no payload'
    for records in outputs.values():
        for record in records:
            path = Path(record.get('path', ''))
            if not path.is_file():
                return 'changed', f'{path} is no longer the file the previous receipt named'
            current = artifacts._record(str(path), verify=True)
            if current['sha256'] != record.get('sha256'):
                return 'changed', f'{path} was already replaced by the interrupted publication'
    return 'intact', 'every output still has the bytes of the receipt this publication replaced'


def recover(static, *, apply=False):
    """Finish deciding what an interrupted publication left behind.

    An ``intact`` bundle is the previous, complete one: its receipt is put back and the run can
    reuse it. A ``changed`` bundle is partial and no receipt describes it, so the stage must be
    rebuilt -- this never guesses which half of the promotion happened, and never deletes the
    partial payload, which is the owner's data to keep or overwrite.
    """
    from eforecast.folder_lock import Busy as AllocatorBusy, folder_lock as file_critical_section
    done = []
    for item in interrupted(static):
        key = item['stage']
        action, detail = 'rebuild_required', item['reason']
        if item.get('in_place'):
            done.append({**item, **_recover_in_place(static, item, apply)})
            continue
        if item['recoverable'] and item['superseded_receipt']:
            action = 'restored' if apply else 'restorable'
            detail = 'the previous complete receipt describes the bundle that is on disk'
            if apply:
                try:
                    # The publisher holds this lock for its whole promotion. A marker is not
                    # evidence of death: a live or slow publication must never be "recovered"
                    # from underneath, so recovery waits for nobody and simply refuses.
                    with file_critical_section(artifacts.directory(static), timeout=0.5):
                        current = interrupted(static)
                        still = next((entry for entry in current if entry['stage'] == key), None)
                        if still is None:
                            action, detail = 'no_longer_interrupted', 'the publication finished by itself'
                        elif not still['recoverable'] or still['superseded_receipt'] != item['superseded_receipt']:
                            action, detail = 'rebuild_required', still['reason']
                        else:
                            shutil.copyfile(still['superseded_receipt'], artifacts.receipt_path(static, key))
                            Path(still['superseded_receipt']).unlink(missing_ok=True)
                            evidence = artifacts.inspect_stage(static, key)
                            if evidence['status'] in {'verified', 'legacy'}:
                                detail = f"restored; the stage is {evidence['status']} again"
                            else:
                                # Saying "restored" about a bundle that still fails inspection
                                # would be the same false certificate this work exists to prevent.
                                action = 'rebuild_required'
                                detail = f'the restored receipt does not verify: {evidence["reason"]}'
                except AllocatorBusy:
                    action = 'publisher_active'
                    detail = 'a publication of this stage holds its lock; it is running, not interrupted'
                except OSError as error:
                    action, detail = 'rebuild_required', f'recovery could not take the stage lock: {error}'
        elif apply:
            # The marker stays: a partial bundle must keep failing verification until it is built
            # again. Saying so is the whole recovery; removing evidence is not.
            detail = f'{detail}; rebuild {key} to publish a complete bundle'
        done.append({**item, 'action': action, 'detail': detail})
    return done


def _recover_in_place(static, item, apply):
    """Put back what an interrupted rebuild set aside, under the case's lock, and say what is true after.

    Nothing is restored while a run holds the case: a marker beside a live stage is a stage at
    work, and the lock is the only thing that can tell the two apart.
    """
    from eforecast.folder_lock import Busy as AllocatorBusy, folder_lock as file_critical_section
    key = item['stage']
    if not item['recoverable']:
        return {'action': 'rebuild_required',
                'detail': f"{item['reason']}; run {key} again to build it" if apply else item['reason']}
    if not apply:
        return {'action': 'restorable', 'detail': item['reason']}
    try:
        with file_critical_section(artifacts.directory(static), timeout=0.5):
            still = next((entry for entry in _interrupted_in_place(static) if entry['stage'] == key), None)
            if still is None:
                return {'action': 'no_longer_interrupted', 'detail': 'the stage finished by itself'}
            for aside in still['set_aside']:
                os.replace(aside, aside[:-len(artifacts.PREVIOUS)])     # over whatever the killed run left there
            Path(still['marker']).unlink(missing_ok=True)
            evidence = artifacts.inspect_stage(static, key)
            if evidence['status'] in {'verified', 'legacy'}:
                return {'action': 'restored', 'detail': f"restored; the stage is {evidence['status']} again"}
            # Saying "restored" about outputs that still fail inspection would be a false certificate.
            return {'action': 'rebuild_required',
                    'detail': f'the outputs put back do not verify: {evidence["reason"]}'}
    except AllocatorBusy:
        return {'action': 'publisher_active',
                'detail': 'a run holds this case; the stage is being written, not interrupted'}
    except OSError as error:
        return {'action': 'rebuild_required', 'detail': f'recovery could not finish: {error}'}


def requested_action(static, options):
    """The Continue / New revision / Rebuild choice this request already carries.

    ``--recreate``, ``--recreate-extraction`` and ``--refit`` have always meant "do it again";
    everything else follows the case's own ``study_action``, which A07 introduced for training.
    """
    from eforecast.training.command_line.revisions import chosen_action
    if any(options.get(flag) for flag in ('recreate', 'recreate_extraction', 'refit')):
        return 'rebuild'
    return chosen_action(static)


def decide(static, plan, *, action='continue', verify=False):
    """One decision per selected stage, in the catalogue's order, with its reason."""
    if action not in ACTIONS:
        raise ValueError(f'action must be one of {", ".join(ACTIONS)}')
    # A preview resolves a case without creating its directories; the catalogue already knows
    # where the model tree is, so the planner never needs the case to have been initialized.
    static = {**static, 'path_model': static.get('path_model') or plan['path_model']}
    nodes = {node['key']: node for node in plan['nodes']}
    unverified = {entry['product'] for entry in plan.get('unverified', [])}
    pending = interrupted(static)
    halted = {item['stage'] for item in pending}
    stages, decided, causes = [], {}, {}
    for key in plan['execution_order']:
        node = nodes[key]
        upstream = [dep for dep in node['dependencies'] if decided.get(dep) == 'rebuild']
        # Why a stage is done again, beside the sentence that says so: ``configuration`` when
        # the case no longer says what the stage was built from -- directly, or through a stage
        # before it. That is the one cause a person is asked about (``changes``).
        cause = None
        if not node['enabled']:
            decision, reason, evidence = 'skip', node['skip_reason'] or 'disabled by the configuration', None
        elif node.get('blocked'):
            entry = node['blocked'][0]
            if all(item.get('rebuildable') for item in node['blocked']):
                # An invalid cached artifact is not a missing input: it is work to do again.
                decision, reason, evidence = 'rebuild', entry['reason'], entry.get('evidence')
                cause = (evidence or {}).get('cause') or 'content'
            else:
                decision, reason, evidence = 'blocked', entry['reason'], None
        elif upstream:
            decision = 'rebuild'
            reason = f'{", ".join(upstream)} will be rebuilt, so this stage\'s inputs change'
            evidence = None
            cause = 'configuration' if any(causes.get(dep) == 'configuration' for dep in upstream) else 'upstream'
        elif action == 'rebuild':
            decision, reason, evidence = 'rebuild', 'an explicit rebuild was requested', None
            cause = 'requested'
        else:
            # Deciding whether to rerun a producer is a question about *its* sources, not only
            # about the product a consumer would read: a changed series must rebuild its target.
            evidence = artifacts.inspect_stage(static, key, verify=verify)
            produced, _ = payloads_of(static, plan, key)
            decision, reason = _from_evidence(evidence, key in halted,
                                              bool(unverified & set(_products_of(plan, key))), bool(produced))
            if decision == 'rebuild':
                cause = ('interrupted' if key in halted else 'missing' if not produced
                         else evidence.get('cause') or 'content')
        units = None
        if key == 'training' and node['enabled']:
            # The trial plan is part of the answer even when the stage cannot run yet: what a
            # rebuild would delete does not depend on whether its inputs are ready today.
            refined, detail, units = _training(static, action, decision, reason, node['options'], cause=cause)
            if decision != 'blocked':
                decision, reason = refined, detail
                if decision != 'rebuild':
                    cause = None
                if any(unit.get('retrain') for unit in (units or {}).get('units', ())):
                    cause = 'configuration'
        decided[key], causes[key] = decision, cause
        entry = {'key': key, 'scope': node['scope'], 'decision': decision, 'reason': reason,
                 'dependencies': list(node['dependencies']),
                 'receipt': (evidence or {}).get('id'), 'status': (evidence or {}).get('status'),
                 'cache': {'effect':'new_generation' if action=='rebuild' and decision=='rebuild' else
                            'content_lookup' if decision not in ('blocked','skip') else 'none',
                           'accepted_references':'retained',
                           'reader_pins':'retained until accepted worker-tree quiescence'}}
        if cause:
            entry['cause'] = cause
        if units is not None:
            entry['units'] = units
        removed = sorted({path for unit in (units or {}).get('units', ()) for path in unit.get('would_delete', ())})
        if decision == 'rebuild':
            entry['affected'], unresolved = artifacts_of(static, plan, key)
            if key == 'extract' and not node['options'].get('recreate_extraction'):
                # The extractor keeps every day it finds and reads only the days that are
                # missing: doing this stage again replaces its receipt, not the archive.
                kept = [path for path in entry['affected'] if not _is_receipt(path)]
                entry['affected'] = [path for path in entry['affected'] if _is_receipt(path)]
                if kept:
                    entry['kept'] = len(kept)
            if units:
                # What a training rebuild would delete is its trials and studies, which the
                # existing trial planner already names unit by unit.
                entry['affected'] = sorted({*entry['affected'], *removed})
            if unresolved:
                entry['affected_unresolved'] = unresolved
        elif removed:
            # A study that is trained from the beginning loses what it holds, whatever word the
            # stage as a whole is given: those are files this run removes.
            entry['affected'] = removed
        stages.append(entry)
    report = {'version': VERSION, 'generated': time.time(), 'action': action, 'verified': bool(verify),
              'plan_id': plan['identity'], 'script': plan['script'], 'stages': stages,
              'interrupted': pending,
              'totals': {name: sum(1 for item in stages if item['decision'] == name) for name in DECISIONS}}
    report['totals']['artifacts_to_replace'] = sum(len(item.get('affected', ())) for item in stages)
    if action == 'new':
        from eforecast.training.command_line import revisions
        report['revision'] = {'training': revisions.study_revision({**static, 'study_action': 'new'}),
            'note': 'a new revision keeps a separate training history; dataset products are shared, '
                    'so they are reused when they verify and rebuilt when they do not',
            'limit': 'trial histories and artifacts use per-method generations; shared dataset products are reused or rebuilt according to their input identities'}
    return report


def _training(static, action, decision, reason, options=None, cause=None):
    """Continue, new revision or rebuild, as the trial planner already understands them.

    A09-07 pins the training branch contracts; the decision about the work *inside* a training
    run has belonged to ``revisions.plan`` since A07. This joins them: the stage's decision is
    what that trial plan implies, and the plan itself travels with it for the preview.
    """
    from eforecast.training.command_line import revisions
    try:
        units = revisions.plan(static, action='rebuild' if action == 'rebuild' else
                                              'new' if action == 'new' else 'continue', options=options)
    except Exception as error:                        # noqa: BLE001 -- a preview never fails on a study
        return decision, f'{reason} (the trial plan is unavailable: {type(error).__name__}: {error})', None
    totals = units['totals']
    # A study that is trained from the beginning wants its whole budget again.
    missing = sum((unit.get('budget') or {}).get('wanted' if unit.get('retrain') else 'missing', 0)
                  for unit in units['units'])
    again = revisions.retrained(units) if action == 'continue' else {}
    # The trials of a global study that each cluster trains again. No study wants them, so no
    # budget counts them: they are work to do all the same, and a stage that has some is not done.
    transfers = revisions.transfers_to_train(totals)
    if again:
        return 'resume', (f"trained from the beginning, because the configuration it was trained under has changed: "
                          f"{revisions.describe_retrained(again)} -- those trials are removed first; "
                          f"{missing} trial(s) wanted in all, {totals['reuse']} compatible trial(s) kept"
                          + transfers), units
    if action == 'rebuild':
        return 'rebuild', (f"an explicit rebuild of {totals['units']} unit(s): "
                           f"{totals['reuse'] + totals['redo'] + totals.get('left', 0)} trial(s) and their studies"), units
    if action == 'new':
        suffix = revisions.study_revision({**static, 'study_action': 'new'})
        return 'resume', (f'a new revision ({suffix}) uses separate study and trial generation directories; '
                          f'previous trial bytes are preserved'), units
    unreadable = [unit for unit in units['units'] if unit.get('budget') is None]
    if unreadable:
        # Without its study a method's budget is unknown, and unknown is not satisfied.
        return 'resume', (f"the search history of {len(unreadable)} method(s) cannot be read, so the "
                          f"budget is unknown; {totals['reuse']} compatible trial(s) would be kept"), units
    left = revisions.left_behind(totals.get('left', 0))
    if missing or totals['redo'] or totals['caches_to_recompute'] or totals.get('transfer'):
        return 'resume', (f"{missing} trial(s) still wanted, {totals['redo']} unfinished to redo"
                          + transfers + f", {totals['reuse']} compatible trial(s) kept" + left
                          + (f", {totals['caches_to_recompute']} cached prediction(s) to recompute"
                             if totals['caches_to_recompute'] else '')), units
    if decision in {'reuse', 'verify'}:
        return decision, (f"the budget is satisfied by {totals['reuse']} compatible trial(s){left}; " + reason), units
    if decision == 'rebuild' and cause == 'configuration' and totals['reuse']:
        # The receipt was written under a configuration the case no longer has, and every
        # study's own trials are still its own -- a sibling's parameter changed, or a setting
        # no trial reads. Nothing is trained and nothing is removed: the stage finds its
        # budget satisfied and records what is there under the configuration as it is now.
        return 'resume', (f"the budget is satisfied by {totals['reuse']} compatible trial(s){left}; nothing is "
                          f"trained or removed, and the stage is recorded under the configuration as it is now"), units
    return decision, reason, units


def _from_evidence(evidence, halted, needs_hashing, produced):
    if halted:
        return 'rebuild', 'a publication of this stage was interrupted; recover or rebuild it'
    status = evidence['status']
    if not produced:
        # No receipt and no payload: an unbuilt stage, not a historical one to inspect.
        return 'rebuild', 'this stage has produced nothing yet'
    if status == 'verified':
        return 'reuse', 'content and input identities match the receipt'
    if status == 'recorded':
        if evidence.get('metadata') == 'changed' or needs_hashing:
            return 'verify', evidence['reason']
        return 'reuse', 'receipt observed; the stage boundary hashes it before anything consumes it'
    if status == 'legacy':
        return 'reuse', f'legacy payload without verified provenance: {evidence["reason"]}'
    return 'rebuild', evidence['reason']


def accepted():
    """The permissions the run was accepted with, as its launch handed them over."""
    raw = os.environ.get('EFORECAST_STAGE_RESUME')
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


# A worker may always do more work than the plan hoped for; it may never do less.
NARROWING = {'reuse': ('reuse', 'resume', 'verify', 'rebuild'), 'resume': ('resume', 'rebuild'),
             'verify': ('verify', 'resume', 'rebuild'), 'rebuild': ('rebuild',),
             'skip': ('skip',), 'blocked': ('blocked', 'rebuild', 'resume')}


def reconcile(report, approved):
    """Keep the accepted decision unless this run's own evidence asks for strictly more work.

    The accepted plan is what the owner saw and what the collector will check. Fresh evidence
    may turn a hoped-for reuse into a rebuild -- an artifact can move between acceptance and
    execution -- but nothing here may turn an approved rebuild into a skip.
    """
    if not approved:
        return report
    for item in report['stages']:
        granted = approved.get(item['key'])
        if granted is None or granted == item['decision']:
            continue
        if item['decision'] in NARROWING.get(granted, ()) and item['decision'] != granted:
            item['reason'] = f"{item['reason']} (the accepted plan asked for {granted})"
            item['narrowed_from'] = granted
        else:
            item['reason'] = (f"the accepted plan requires {granted}; this run observed "
                              f"{item['decision']}: {item['reason']}")
            item['decision'] = granted
    report['totals'] = {name: sum(1 for item in report['stages'] if item['decision'] == name)
                        for name in DECISIONS}
    report['totals']['artifacts_to_replace'] = sum(len(item.get('affected', ())) for item in report['stages'])
    report['accepted'] = dict(approved)
    return report


def pin(plan, report):
    """Record the decisions in the plan the run is about to accept.

    Availability is not the contract: ``plan_identity`` ignores this, exactly as it ignores
    blocked messages, so a plan does not become a different plan because an artifact exists.
    """
    plan['resume'] = {item['key']: item['decision'] for item in report['stages']}
    # The whole report, including what a rebuild would replace: the effects the owner approved
    # must travel with the plan, not only their one-word summary.
    plan['resume_plan'] = report
    return plan


def permitted(plan, key):
    """Whether this run was given permission to reuse a stage instead of recomputing it."""
    return (plan.get('resume') or {}).get(key) == 'reuse'


def whole(report, key):
    """Whether this run found a stage's work whole, and so will not do any of it again.

    More than ``permitted``: the stage has a receipt, and its content and inputs were read and
    match it -- which is what the stage boundary asks before it reuses instead of working. A
    payload of unknown provenance is permitted too, and is done again when the boundary finds
    no receipt to hold it to; a plan made from sizes and dates alone has verified nothing.
    A run uses this to ask for nothing it would only need if it worked: ``run_training.py``
    does not ask for a card to reuse a finished stage.
    """
    for item in (report or {}).get('stages', ()):
        if item['key'] == key:
            return item['decision'] == 'reuse' and item.get('status') == 'verified' and bool(item.get('receipt'))
    return False


def _is_receipt(path):
    return f'{os.sep}{artifacts.DIRECTORY}{os.sep}' in str(path)


#: The scripts whose work is long to do again and cannot be had back: datasets and trained
#: models. What the combination step makes is derived from them in minutes and is made again
#: whenever they change; nothing is said about that.
ASKED_ABOUT = ('datasets', 'training')
#: The environment of a run the service started says ``1`` here when the person who started it
#: in the browser was shown what it replaces and accepted that (``runs``). A run from the
#: command line carries no answer and needs none: it says what it replaces and goes on.
ACCEPTED = 'EFORECAST_ACCEPT_CHANGES'


def changes(report):
    """What this run would replace because the case no longer says what it was built from.

    ``{'stages': [...], 'families': {...}}``, or None when there is nothing of the kind. A stage
    is listed when its configuration changed -- or that of a stage before it -- *and* it has
    output on disk that doing it again replaces. A family is listed when its studies hold
    trials of another configuration, which are removed before it is trained from the beginning.

    An explicit rebuild is not listed: whoever asked for it said so. Neither is what a run
    builds for the first time, finishes after an interruption, or repairs.
    """
    if report.get('script') not in ASKED_ABOUT or report.get('action') == 'rebuild':
        return None
    stages, families = [], {}
    for item in report['stages']:
        if report['script'] == 'datasets':
            replaced = [path for path in item.get('affected', ()) if not _is_receipt(path)]
            if item['decision'] == 'rebuild' and item.get('cause') == 'configuration' and replaced:
                stages.append({'key': item['key'], 'reason': item['reason'], 'files': len(replaced)})
            continue
        # Training removes nothing but the studies it starts over: its own word for the stage
        # as a whole (resume, rebuild) deletes no trial.
        units = item.get('units') or {}
        if units.get('units') and report.get('action') == 'continue':
            from eforecast.training.command_line import revisions
            families.update(revisions.retrained(units))
    if not stages and not families:
        return None
    return {'stages': stages, 'families': families}


def started_over(report):
    """The families a training plan trains from the beginning: what an acceptance of it covers."""
    return tuple(sorted((changes(report) or {}).get('families', {})))


def _plainly(reason):
    """A stage's reason without the planner's own vocabulary around it."""
    import re
    reason = re.sub(r'^\w+ cached completion is invalid: ', '', reason)
    reason = reason.replace('; select an explicit rebuild', '').replace('; rebuild this stage', '')
    return re.sub(r'^(\w+) configuration changed$', 'its settings changed', reason)


SHOWN = 20


def settings_changed(path_model, case_folder):
    """The settings that differ between what the model was built from and a case, as lines.

    Empty when the model keeps no configuration of its own yet (it was built before one was
    kept), or when either side cannot be read: this explains a decision, it never makes one.
    """
    try:
        from . import model_configuration
        from .config_files import load_document
        from .document_changes import describe as line
        found = model_configuration.since(path_model, load_document(case_folder))
        return [line(change) for change in (found or {}).get('changes', ())]
    except Exception:                                           # noqa: BLE001
        return []


def describe_changes(found, name=None, settings=()):
    """``changes`` as the lines a person reads before saying yes."""
    lines = [f"The configuration{f' of {name}' if name else ''} is not the one this was built from."]
    if settings:
        lines.append('  changed since it was built:')
        lines += [f'    {line}' for line in list(settings)[:SHOWN]]
        if len(settings) > SHOWN:
            lines.append(f'    ... and {len(settings) - SHOWN} more setting(s)')
    if found['stages']:
        lines.append('  built again, replacing what is there:')
        lines += [f"    {item['key']:<11} {item['files']} file(s) -- {_plainly(item['reason'])}"
                  for item in found['stages']]
    if found['families']:
        lines.append('  trained from the beginning, their trials removed first:')
        lines += [f"    {method:<11} {entry['units']} stud{'y' if entry['units'] == 1 else 'ies'}, "
                  f"{entry['trials']} trial folder(s)" for method, entry in found['families'].items()]
    return '\n'.join(lines)


class NotAccepted(SystemExit):
    """A run the service started that would replace built work, without its submitter's answer."""


def announce_changes(report, static):
    """Say what this run replaces because the configuration changed, before it replaces it.

    Returns whether there is anything of the kind.

    A run from the command line is its own decision. Whoever changed the case and started the
    run is told what that costs -- which datasets are built again, which families lose their
    trials and are trained from the beginning -- and the run goes on. Nothing is asked, and
    there is no flag to say yes with.

    A run the service started was asked about in the browser, where its submitter was shown
    the same list and typed the name of the case to accept it (``runs``). It carries that
    answer in its environment. One that arrives without it stops here, before it has touched
    a file: the question was the service's to ask, and it was not asked.
    """
    import logging
    found = changes(report)
    if not found:
        return False
    name = static.get('_id') or static.get('project_name') or 'this case'
    from . import run_inputs
    running = run_inputs.executing()
    text = describe_changes(found, name, settings_changed(static.get('path_model'), running['folder'])
                            if running and static.get('path_model') else ())
    answer = accepted_from_environment()
    if answer is False:
        raise NotAccepted(
            f'{text}\nNothing was changed. The service started this run without the answer of whoever '
            f'submitted it. Start it again from the Runs tab, which shows this list and asks before it '
            f'goes on; or put the configuration back.')
    logging.getLogger(__name__).warning(
        '%s\n  %s', text, 'accepted where the run was submitted: this run replaces them' if answer
        else 'this run replaces them now. Stop it within the next stage if that is not what you want; '
             '--dry-run shows the same list and changes nothing')
    return True


def accepted_from_environment():
    """``True``/``False`` when a managed run was started with an answer, None for a run nobody manages."""
    if accepted() is None:
        return None
    return os.environ.get(ACCEPTED) == '1'


def describe(report):
    """The decisions as lines a person reads before approving them."""
    from eforecast.training.command_line import revisions
    lines = [f"resume plan ({report['action']}) for {report['script']}, "
             f"plan {report['plan_id'][:12]}{'' if report['verified'] else ' -- metadata only, not hashed'}"]
    for item in report['stages']:
        lines.append(f"  {item['key']:<12} {item['decision']:<8} {item['reason']}")
        for unit in (item.get('units') or {}).get('units', ()):
            budget = unit.get('budget') or {}
            if unit.get('retrain'):
                lines.append(f"      {unit['cluster']}/{unit['method']}: trained from the beginning -- "
                             f"{unit.get('discard', 0)} trial folder(s) of another configuration are removed, "
                             f"{budget.get('wanted', '?')} trial(s) wanted" + revisions.transfers_of(unit))
                continue
            lines.append(f"      {unit['cluster']}/{unit['method']}: "
                         + (f"{budget['done']}/{budget['wanted']} trials done, " if budget else '')
                         + f"reuse {unit['reuse']}, redo {unit['redo']}" + revisions.transfers_of(unit)
                         + (f", {unit['left']} folder(s) of stopped trials left as they are" if unit.get('left') else '')
                         + (f", {unit['other_revision']} in another revision" if unit['other_revision'] else ''))
        if item.get('kept'):
            lines.append(f"      keeps the {item['kept']} file(s) it finds and makes only what is missing")
        for path in item.get('affected', ())[:8]:
            lines.append(f"      would replace {path}")
        if len(item.get('affected', ())) > 8:
            lines.append(f"      ... and {len(item['affected']) - 8} more file(s)")
    for item in report['interrupted']:
        lines.append(f"  interrupted publication of {item['stage']}: canonical bundle {item['canonical']}; "
                     f"{item['reason']}")
    totals = report['totals']
    lines.append('  total: ' + ', '.join(f'{totals[name]} {name}' for name in DECISIONS)
                 + f", {totals['artifacts_to_replace']} artifact(s) would be replaced")
    if any(item['decision'] == 'rebuild' for item in report['stages']):
        lines.append('  nothing was deleted: this is a plan. A rebuild replaces the files listed above.')
    found = changes(report)
    if found:
        lines += ['  ' + line for line in describe_changes(found).split('\n')]
        lines.append('  a run from the command line replaces them without asking; a run started in the '
                     'browser asks first')
    return '\n'.join(lines)
