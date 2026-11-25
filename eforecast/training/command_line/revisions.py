# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Continue, new revision, or rebuild -- and what each of them would do, before it is done.

A run reuses what it finds. Until now it never said what it was reusing or why, and the only way
to start again was ``--refit``, which deletes. These three actions are the same decision made
explicitly:

``continue``       satisfy the budget with compatible completed work, in the history it is in;
``new``            keep a separate search history for the current inputs; the old one stays;
``rebuild``        discard the work for these inputs and do it again.

``plan`` answers what each of them would reuse, recompute or delete. It reads; it never writes
anything but its own report, and never deletes.
"""
import json
import logging
import os
import time

from eforecast.common_utils import artifact, completion, lineage, trial_layout

log = logging.getLogger(__name__)

ACTIONS = ('continue', 'new', 'rebuild')
CACHE_FILES = ('predictions_eval.pickle', 'predictions_online.pickle',
               'quantiles_eval.pickle', 'quantiles_online.pickle')


def chosen_action(static_data=None):
    choice = (static_data or {}).get('study_action') or os.environ.get('EFORECAST_STUDY_ACTION') or 'continue'
    return choice if choice in ACTIONS else 'continue'


def study_revision(static_data):
    """The suffix that keeps this history apart, or ``None`` to stay in the one that exists."""
    return trial_layout.revision(static_data)



def model_groups(static_data):
    """``[(cluster name, cluster dir, [methods])]`` of this case, from the tree and the request."""
    import joblib

    gpu = [m for m, on in (static_data.get('cluster_methods') or {}).items() if on]
    cpu = [m for m, on in (static_data.get('methods_cpu') or {}).items() if on]
    glob = [m for m, on in (static_data.get('global_methods') or {}).items() if on]
    groups = [('global', os.path.join(static_data['path_model'], 'global'), glob)]
    clusters_file = os.path.join(static_data['path_model'], 'clusters.pickle')
    if static_data.get('is_Fuzzy') and os.path.isfile(clusters_file):
        for name, directory in trial_layout.cluster_folders(static_data).items():
            groups.append((name, directory, gpu + cpu))
    return groups


def trial_number(trial):
    """``('study', 3)`` for ``test_3``, ``('transfer', 3)`` for ``test_tl3``, else ``(None, None)``.

    A cluster's ``test_tl<n>`` is not a trial of its own study: it is the global study's trial
    ``n`` trained again on the cluster, and it has no record in any study.
    """
    name = trial[len('test_'):]
    if name.isdigit():
        return 'study', int(name)
    if name.startswith('tl') and name[2:].isdigit():
        return 'transfer', int(name[2:])
    return None, None


def transfer_sources(static_data, method):
    """The trials of the global study of ``method`` that every cluster trains again, as a run finds them.

    ``(sources, passed over)``: ``[(number, folder)]`` for every ``test_<number>`` of the global
    study that is finished work and kept the parameters it was trained with, and ``[(name,
    reason)]`` for the folders that hold parameters and are not finished -- a folder with the
    parameters that were sampled for it is not a trained model to start from.

    This is the one place that says what a transfer is trained from. The queue asks it when the
    global study has settled (``orchestrator.FixedUnit``) and the plan asks it before anything
    runs, so that the plan counts what the queue will train.
    """
    global_dir = str(trial_layout.method_directory(static_data, os.path.join(static_data['path_model'], 'global'), method))
    sources, passed = [], []
    for name in sorted(os.listdir(global_dir)) if os.path.isdir(global_dir) else []:
        folder = os.path.join(global_dir, name)
        if not name.startswith('test_') or not os.path.exists(os.path.join(folder, 'parameters.pickle')):
            continue
        usable, reason = completion.is_finished(folder)
        if usable:
            sources.append((name.split('_')[-1], folder))
        else:
            passed.append((name, reason))
    return sources, passed


def transfer_tasks(static_data, cluster_dir, method, sources, *, refit=False):
    """What one cluster has of ``sources`` and what it has still to train: ``(done, tasks)``.

    A source whose ``test_tl<number>`` in the cluster is finished work is done, unless
    everything is trained again (``refit``). Each task names the trial and the folder of the
    global trial it starts from.
    """
    done, tasks = [], []
    for number, source in sources:
        name = f'tl{number}'
        target = str(trial_layout.trial_directory(static_data, cluster_dir, method, name))
        if not refit and completion.is_finished(target)[0]:
            done.append(f'test_{name}')
        else:
            tasks.append({'trial_name': name, 'source': source, 'target': target})
    return done, tasks


def trains_transfers(static_data, cluster_name, method, options=None):
    """Whether a run trains, in this cluster, the trials of the method's global study again."""
    from eforecast.config.capabilities import CPU_METHODS
    return (cluster_name != 'global' and method not in CPU_METHODS and not (options or {}).get('cpu')
            and bool((static_data.get(method) or {}).get('transfer_learning_from_global')))


def trial_decisions(method_dir, comparability, inputs, unit_identity=None, *, view=None, sources=None):
    """One decision per trial folder: reuse it, redo it, or leave it to another revision.

    ``inputs`` is this family's own comparability id. ``comparability`` is the case-wide one,
    accepted as well so that trials recorded before the identity became per-family are not all
    declared another revision at once -- the same allowance the rest of the system makes for
    manifests an older build wrote. New trials record the narrow id.

    ``view`` is what the unit's study says of its own trials: ``{'kept', 'finished'}``, the
    numbers it counts toward the budget and the numbers it has finished. Where the study knows
    a trial, the study decides: it carries the family's id, while a folder written before
    folders recorded that id carries the case's, which every sibling's change makes look like
    another revision. ``sources`` are the trials the *global* study of the method counts as its
    own, which is what vouches for the transfers (``test_tl<n>``) a cluster trained from them.
    """
    out = []
    for trial in sorted(os.listdir(method_dir)) if os.path.isdir(method_dir) else []:
        path = os.path.join(method_dir, trial)
        if not trial.startswith('test_') or not os.path.isdir(path):
            continue
        manifest = artifact.read(path) or {}
        usable, reason = completion.is_finished(path)
        depends = manifest.get('depends_on') or {}
        recorded = depends.get('comparability_id')
        binding = depends.get('cpu_unit_identity')
        # ... and the names the family's inputs had under an earlier formula (``lineage.Inputs``).
        accepted = {str(value) for value in (inputs, comparability) if value is not None} \
            | set(getattr(inputs, 'also', ()))
        kind, number = trial_number(trial)
        if unit_identity is None or isinstance(unit_identity, str):
            bound = unit_identity is None or binding in (None, unit_identity)
        else:
            bound = unit_identity.admits(number, binding, same_family=recorded in accepted)
        compatible = (recorded is None or recorded in accepted) and bound
        if kind == 'study' and view is not None and number in view['finished']:
            compatible = number in view['kept']
        elif kind == 'transfer' and sources is not None and number in sources \
                and not depends.get('comparability_scope'):
            # A folder that names its own family's id speaks for itself. One written before
            # folders did is vouched for by the trial it was trained from.
            compatible = True
        caches = [name for name in CACHE_FILES if os.path.isfile(os.path.join(path, name))]
        # A trial's cached predictions are of other inputs when the trial is. Where the study (or
        # the trial a transfer was trained from) vouches for a folder that records an earlier
        # name of the same inputs, its predictions are not to be computed again either: a
        # finished case said "49 cached prediction(s) to recompute" at every run, which nothing
        # recomputed, and started its training stage each time to do nothing.
        stale = [name for name in caches
                 if not os.path.isfile(os.path.join(path, f'{name}.key.json'))
                 or (recorded is not None and accepted and recorded not in accepted and not compatible)]
        out.append({'trial': trial, 'finished': usable, 'reason': reason,
                    'dependency_id': recorded, 'same_inputs': None if recorded is None else recorded in accepted,
                    'attempt_id': manifest.get('attempt_id'), 'caches': caches, 'caches_to_recompute': stale,
                    'decision': 'reuse' if usable and compatible else
                                ('another revision' if usable else 'unfinished, redo')})
    return out


def plan(static_data, action=None, options=None):
    """What ``action`` would reuse, recompute and delete, unit by unit. Nothing is changed.

    Under ``continue`` a unit whose study holds finished trials of another configuration of its
    family is ``retrain``: it is trained from the beginning, and ``would_delete`` names what is
    removed first (``optuna_search.begins_again``). Nothing else of ``continue`` removes anything.
    """
    from eforecast.training.command_line import optuna_search
    from eforecast.common_utils.cpu_scope import Binding
    from eforecast.config.capabilities import CPU_METHODS

    action = action or chosen_action(static_data)
    if action == 'new':
        static_data = {**static_data, 'study_action': 'new'}
    full = lineage.dependencies(static_data)
    comparability = inputs = full['comparability_id']
    families = lineage.family_keys(static_data)
    report = {'version': 1, 'generated': time.time(), 'action': action, 'comparability_id': comparability,
              'dependency_id': full['dependency_id'], 'code': lineage.code_identity(), 'units': []}
    sources = {}        # method -> the trials its global study counts as its own
    parents = {}        # method -> the unit of its global study, when this plan has one
    for cluster_name, cluster_dir, methods in model_groups(static_data):
        if options is not None:
            from eforecast.config.stage_catalogue import _methods
            glob, clusters = _methods(static_data, options)
            methods = glob if cluster_name == 'global' else clusters
        for method in methods:
            method_dir = str(trial_layout.method_directory(static_data, cluster_dir, method))
            binding = Binding(static_data, cluster_dir, method) if method in CPU_METHODS else None
            # One family's parameter change must not make its siblings' trials incomparable.
            # The whole-configuration id stays in the report as the case-wide answer; each unit
            # is decided against its own family's inputs, which is the difference between
            # retraining one family and retraining all of them. A family this configuration
            # does not switch on keeps the case-wide id: narrowing an unknown name would drop
            # every family's parameters and make anything look reusable.
            unit_inputs = (lineage.comparability_id(static_data, full=full, method=method)
                           if method in families else inputs)
            name = budget = view = failure = None
            try:
                # Loading a study creates its storage: a plan that did that would write into the
                # model tree it is only supposed to describe, which a dry run must never do.
                if not os.path.exists(optuna_search.journal_path(cluster_dir, method, static_data)):
                    failure = 'no search history yet for this method'
                else:
                    # Read-only: loading a study for execution creates, adopts and stamps it,
                    # and a plan may do none of those to the history it is describing.
                    study = optuna_search.read_study(static_data, cluster_name, cluster_dir, method)
                    if study is None:
                        failure = 'no study of this name in the existing history'
                    else:
                        name = study.study_name
                        budget = optuna_search.budget_state(static_data, study, method, unit_inputs,
                                                            unit_identity=binding)
                        kept, finished, _ = optuna_search.completed_compatible(study, unit_inputs,
                                                                               unit_identity=binding)
                        # The study speaks for the trials it recorded an identity for. One it
                        # recorded none for -- a history from before the bookkeeping -- it merely
                        # does not object to, and the folder's own record is then all there is.
                        recorded = {trial.number for trial in study.get_trials(deepcopy=False)
                                    if trial.user_attrs.get(optuna_search.COMPARABILITY_KEY) is not None}
                        view = {'kept': set(kept), 'finished': set(finished) & recorded}
            except Exception as error:                     # noqa: BLE001 -- a plan never fails on a study
                name = budget = view = None
                failure = f'{type(error).__name__}: {error}'
            if cluster_name == 'global' and view is not None:
                sources[method] = view['kept']
            trials = trial_decisions(method_dir, comparability, unit_inputs, binding, view=view,
                                     sources=None if cluster_name == 'global' else sources.get(method))
            # A transfer (``test_tl<n>``) is no trial of this unit's study: the study neither
            # replaces it nor counts it, so it is counted on its own, further down.
            own = [t for t in trials if trial_number(t['trial'])[0] != 'transfer']
            unit = {'cluster': cluster_name, 'method': method, 'directory': method_dir,
                    'comparability_id': str(unit_inputs),
                    'study': name, 'budget': budget, 'trials': trials,
                    'reuse': sum(1 for t in trials if t['decision'] == 'reuse'),
                    'redo': sum(1 for t in own if t['decision'] == 'unfinished, redo'),
                    'other_revision': sum(1 for t in trials if t['decision'] == 'another revision'),
                    'caches_to_recompute': sum(len(t['caches_to_recompute']) for t in trials)}
            if failure:
                unit['study_error'] = failure
            # A folder a stopped trial left behind is not trained again: its study marks the trial
            # failed and asks for another under a new number. While the study still wants trials
            # that is work to do. Once it has all it wants nothing replaces the folder: it is
            # left as it is, and a plan that called it work would never come to an end.
            unit['left'] = 0
            if (unit['budget'] or {}).get('missing', 1) == 0:
                for trial in own:
                    if trial['decision'] == 'unfinished, redo':
                        trial['decision'] = 'stopped, left as it is'
                unit['left'], unit['redo'] = unit['redo'], 0
            # A study that holds finished trials of another configuration of its family is not
            # continued beside them. Everything of the unit goes and it is trained from the
            # beginning: the trials counted here are what that removes.
            unit['retrain'] = action == 'continue' and optuna_search.begins_again(budget)
            unit['discard'] = 0
            if unit['retrain']:
                unit['other_configuration'] = budget['incompatible']
                unit['discard'] = len(trials)
                for trial in trials:
                    trial['decision'] = 'removed, trained again'
                    trial['caches_to_recompute'] = []
                unit.update(reuse=0, redo=0, left=0, other_revision=0, caches_to_recompute=0)
                unit['would_delete'] = removed_with(static_data, cluster_dir, cluster_name, method, method_dir, trials)
            if action == 'rebuild':
                unit['would_delete'] = sorted(
                    [os.path.join(method_dir, t['trial']) for t in trials]
                    + [str(trial_layout.results_file(static_data, cluster_dir, cluster_name, method))]
                    + ([unit['study']] if unit['study'] else []))
            if cluster_name == 'global':
                parents[method] = unit
            unit['transfer'] = _transfers(static_data, action, unit, cluster_dir, parents.get(method), options)
            report['units'].append(unit)
    report['totals'] = {key: sum(unit[key] for unit in report['units'])
                        for key in ('reuse', 'redo', 'left', 'other_revision', 'caches_to_recompute', 'discard')}
    report['totals']['retrain'] = sum(1 for unit in report['units'] if unit['retrain'])
    report['totals']['units'] = len(report['units'])
    # Transfers are work of their own kind: no study wants them, so no budget counts them.
    for key, name in (('pending', 'transfer'), ('done', 'transfer_done'), ('later', 'transfer_later')):
        report['totals'][name] = sum((unit.get('transfer') or {}).get(key, 0) for unit in report['units'])
    return report


def _transfers(static_data, action, unit, cluster_dir, parent, options):
    """The transfers of one cluster's unit: ``{'sources', 'done', 'pending', 'later'}``, or None.

    ``pending`` is what a run trains now: every finished trial of the method's global study
    that this cluster has no finished ``test_tl<n>`` of. Until this was counted a plan said
    "0 trial(s) still wanted" of a case with dozens of them untrained, and a training stage
    whose studies were complete was reused without training any.

    ``later`` is what the global study will still add: each trial it finishes is trained once
    more in every cluster, so the trials it still wants are an upper bound.

    The folders of transfers that were started and never finished are decided here as well. One
    whose source is still there is trained again; one whose source is gone is left as it is.
    """
    transfers = [t for t in unit['trials'] if trial_number(t['trial'])[0] == 'transfer']
    unfinished = [t for t in transfers if t['decision'] == 'unfinished, redo']
    if not trains_transfers(static_data, unit['cluster'], unit['method'], options):
        for trial in unfinished:                    # nothing trains these: not work to do
            trial['decision'] = 'stopped, left as it is'
        unit['left'] += len(unfinished)
        return None
    # A global study that starts over (or is rebuilt) trains its trials again first: what is
    # there now is not what the clusters will start from.
    parent_over = parent is not None and (action == 'rebuild' or bool(parent.get('retrain')))
    starting_over = action == 'rebuild' or bool(unit.get('retrain'))
    ready = [] if parent_over else transfer_sources(static_data, unit['method'])[0]
    done, tasks = transfer_tasks(static_data, cluster_dir, unit['method'], ready, refit=starting_over)
    later = 0
    if parent is not None:
        budget = parent.get('budget')
        if budget:
            later = budget['wanted'] if parent_over else budget['missing']
        elif not parent['trials']:                  # no history and no folder: nothing was trained yet
            later = int(static_data[unit['method']]['n_trials'])
    if not starting_over and not parent_over:
        again = {f"test_{task['trial_name']}" for task in tasks}
        for trial in unfinished:
            if trial['trial'] not in again:
                trial['decision'] = 'stopped, left as it is'
                unit['left'] += 1
    return {'sources': len(ready), 'done': len(done), 'pending': len(tasks), 'later': later}


def removed_with(static_data, cluster_dir, cluster_name, method, method_dir, trials):
    """What is on disk of a unit that is trained from the beginning: what ``delete_study`` removes."""
    from eforecast.training.command_line import optuna_search
    journal = optuna_search.journal_path(cluster_dir, method, static_data)
    candidates = [os.path.join(method_dir, trial['trial']) for trial in trials]
    candidates += [os.path.join(method_dir, 'trials'), journal,
                   str(trial_layout.results_file(static_data, cluster_dir, cluster_name, method))]
    return sorted(path for path in candidates if os.path.exists(path))


def retrained(report):
    """``{method: {'units', 'trials', 'clusters'}}`` of the units a plan trains from the beginning."""
    out = {}
    for unit in report['units']:
        if unit.get('retrain'):
            entry = out.setdefault(unit['method'], {'units': 0, 'trials': 0, 'clusters': []})
            entry['units'] += 1
            entry['trials'] += unit.get('discard', 0)
            entry['clusters'].append(unit['cluster'])
    return out


def describe_retrained(families):
    """``TRANSFORMER (8 studies, 91 trial folders)``, family by family."""
    return ', '.join(f"{method} ({entry['units']} stud{'y' if entry['units'] == 1 else 'ies'}, "
                     f"{entry['trials']} trial folder(s))" for method, entry in families.items())


def left_behind(count):
    """``, 2 folder(s) of stopped trials left as they are``, or nothing."""
    return f', {count} folder(s) of stopped trials left as they are' if count else ''


def transfers_of(unit):
    """``, transfers 3/15 trained, 12 to train``, for a unit that trains any; else nothing."""
    transfer = unit.get('transfer')
    if not transfer or not (transfer['sources'] or transfer['later']):
        return ''
    return (f", transfers {transfer['done']}/{transfer['sources']} trained"
            + (f", {transfer['pending']} to train" if transfer['pending'] else '')
            + (f", up to {transfer['later']} more after the global study" if transfer['later'] else ''))


def transfers_to_train(totals):
    """``, 84 transfer trial(s) to train``, with what the global studies will still add; or nothing."""
    now, later = totals.get('transfer', 0), totals.get('transfer_later', 0)
    return ((f', {now} transfer trial(s) to train' if now else '')
            + (f", up to {later} {'more ' if now else ''}transfer trial(s) once the global studies have finished"
               if later else ''))


def describe(report):
    """The plan as lines a person reads before deciding."""
    lines = [f"action: {report['action']}   inputs {report['dependency_id'][:12]}   "
             f"comparable-with {report['comparability_id'][:12]}"]
    for unit in report['units']:
        budget = unit.get('budget')
        if unit.get('retrain'):
            lines.append(
                f"  {unit['cluster']}/{unit['method']}: trained from the beginning -- "
                f"{unit['other_configuration']} finished trial(s) are of another configuration; "
                f"{unit['discard']} trial folder(s) and the study are removed, {budget['wanted']} trial(s) wanted"
                + transfers_of(unit))
            continue
        lines.append(
            f"  {unit['cluster']}/{unit['method']}: "
            + (f"{budget['done']}/{budget['wanted']} trials done" if budget else f"study unreadable "
               f"({unit.get('study_error')})")
            + f", reuse {unit['reuse']}, redo {unit['redo']}"
            + transfers_of(unit)
            + left_behind(unit['left'])
            + (f", {unit['other_revision']} from another revision" if unit['other_revision'] else '')
            + (f", {unit['caches_to_recompute']} cached prediction(s) to recompute"
               if unit['caches_to_recompute'] else ''))
        if unit.get('would_delete'):
            lines.append(f"      would delete {len(unit['would_delete'])} item(s)")
    totals = report['totals']
    lines.append(f"  total: {totals['units']} unit(s), reuse {totals['reuse']}, redo {totals['redo']}"
                 f"{left_behind(totals['left'])}, "
                 f"{totals['other_revision']} from another revision, "
                 f"{totals['caches_to_recompute']} cache(s) to recompute"
                 + transfers_to_train(totals)
                 + (f", {totals['retrain']} trained from the beginning ({totals['discard']} trial folder(s) removed)"
                    if totals.get('retrain') else ''))
    if report['action'] == 'rebuild':
        lines.append('  nothing was deleted: this is a preview. Run the training with --refit to rebuild.')
    return '\n'.join(lines)


def write_report(static_data, report):
    path = os.path.join(static_data['path_model'], 'Results', 'revision_plan.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f'{path}.{os.getpid()}.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1, default=str)
    os.replace(temporary, path)
    return path
