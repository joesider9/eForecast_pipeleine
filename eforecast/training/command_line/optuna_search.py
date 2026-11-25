# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Optuna studies for the GPU trials: one study per (cluster, method).

* storage: ``<cluster_dir>/<method>/study.journal`` (Optuna JournalStorage, safe for
  several worker processes on a local disk) or any Optuna storage URL such as
  ``postgresql://user:pass@host/db`` when workers run on several machines
* search space: the ``space`` / ``space_structure`` dictionaries that
  :class:`ObjectiveAbstractClass` derives from the method parameters (sets are
  categorical, lists are ranges) and from the layer sizes of every experiment tag;
  the structure parameters are conditional on ``experiment_tag``
* history: trials of the former hyperopt loop (``trials/trial<k>.pickle``) are
  imported once, so a search continues from what was already computed
* records: a completed trial is also written as ``trials/trial<k>.pickle`` with the
  same fields as before, so ``get_results`` and everything downstream is unchanged
"""
import glob
import hashlib
import logging
import math
import json
import os
import warnings

import joblib
import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState, create_trial

from eforecast.common_utils import completion, lineage, trial_layout
from eforecast.training.command_line import leases

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RECORD_COLUMNS = ['trial_number', 'duration', 'value', 'mae_test', 'mae_val', 'sse_val', 'sse_test']


def journal_path(cluster_dir, method, static_data=None):
    return str(trial_layout.method_directory(static_data or {}, cluster_dir, method) / 'study.journal')


def storage_for(cluster_dir, method, storage_url=None, static_data=None):
    if storage_url:
        return storage_url
    path = journal_path(cluster_dir, method, static_data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


IDENTITY_KEYS = ('project_owner', 'projects_group', 'version_group', 'type', 'project_name',
                 'horizon_type', 'version_model')


def study_identity(static_data, cluster_name, method):
    """``(digest, provenance)``: which model tree, cluster and method this study belongs to.

    The name used to be ``<project name>__<cluster>__<method>``, and the project name is the only
    part of the model tree in it.  Six trees differing in owner, group, group version, model
    version or project type produced **one** study name, so a shared storage mixed their histories,
    their trial numbers and the paths their trials point at.

    The digest covers the logical identity and never the filesystem path: moving a model tree, or
    mounting it somewhere else inside a container, must not orphan its studies.  The path is kept
    beside it as provenance, which is what a mismatch report needs to be readable.
    """
    fields = {key: static_data.get(key) for key in IDENTITY_KEYS}
    fields.update(cluster=cluster_name, method=method)
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return digest, dict(fields, identity=digest, path_model=static_data.get('path_model'),
                        partition=current_partition(static_data))


def current_partition(static_data):
    """The digest of the split the trials of this study are scored against, or ``None``.

    Not part of the identity: a redrawn partition does not make the study belong to someone else,
    it makes its earlier values incomparable with the new ones, which is worth saying out loud and
    is not worth discarding a history for.
    """
    try:
        from eforecast.datasets.data_preprocessing.data_split import partition_fingerprint
        from eforecast.datasets.files_manager import FilesManager
        cv_mask = FilesManager(static_data, is_online=False, train=True).check_if_exists_cv_data()
        return partition_fingerprint(static_data, cv_mask)['masks_sha256'] if cv_mask else None
    except Exception as error:  # noqa: BLE001 -- provenance must never be the thing that stops a run
        log.debug('cannot read the partition for the study provenance (%r)', error)
        return None


def study_name(static_data, cluster_name, method):
    digest, _ = study_identity(static_data, cluster_name, method)
    revision = trial_layout.generation(static_data, trial_layout.scope_directory(static_data, cluster_name), method)
    return (f"{static_data['_id']}__{cluster_name}__{method}__{digest}"
            + (f'__r{revision}__g1' if revision else ''))


def legacy_study_name(static_data, cluster_name, method):
    """What the study was called before the identity was in the name."""
    return f"{static_data['_id']}__{cluster_name}__{method}"


def study_candidates(static_data, cluster_name, method):
    name = study_name(static_data, cluster_name, method)
    if trial_layout.generation(static_data, trial_layout.scope_directory(static_data, cluster_name), method):
        return (name,)
    return (name, legacy_study_name(static_data, cluster_name, method))


def existing_study_names(storage):
    try:
        return set(optuna.study.get_all_study_names(storage=storage))
    except Exception as error:  # noqa: BLE001 -- a storage that cannot be listed is simply not adopted from
        log.debug('cannot list the studies of this storage (%r)', error)
        return set()


def resolve_study_name(static_data, cluster_name, method, storage):
    """The name this study actually has in ``storage``: the qualified one, or a legacy one to adopt.

    Every reader needs this, not only the trainer: the web UI opens a study by name, and an adopted
    study still lives under the name it was created with.
    """
    name = study_name(static_data, cluster_name, method)
    legacy = legacy_study_name(static_data, cluster_name, method)
    if name == legacy:
        return name
    if trial_layout.generation(static_data, trial_layout.scope_directory(static_data, cluster_name), method):
        return name  # a revision never adopts the unqualified legacy history
    known = existing_study_names(storage)
    return legacy if name not in known and legacy in known else name


def check_study_identity(study, provenance):
    """Stamp a study with its provenance, or refuse one that belongs to another model tree.

    A study carrying no provenance predates this check: it is adopted and stamped, which is how an
    existing model keeps its history.  One that carries a different identity is a collision -- two
    model trees meeting in one shared storage -- and continuing it would mix their trials.
    """
    stored = study.user_attrs.get('eforecast_provenance')
    if not stored:
        study.set_user_attr('eforecast_provenance', provenance)
        return provenance
    keys = IDENTITY_KEYS + ('cluster', 'method')
    differences = {key: (stored.get(key), provenance[key]) for key in keys if stored.get(key) != provenance[key]}
    if differences:
        raise ValueError(
            f'study {study.study_name} belongs to another model tree: '
            + ', '.join(f'{key} was {was!r}, now {now!r}' for key, (was, now) in differences.items())
            + f' (it was created for {stored.get("path_model")}, this run is {provenance["path_model"]}). '
              'Continuing it would mix two histories, their trial numbers and the folders the trials point '
              'at. Give one of them a different owner, group, model version or project name, or train it '
              'against its own storage.')
    if stored.get('path_model') != provenance['path_model']:
        log.info('study %s was created for %s and runs now from %s; the identity is the same, so it continues',
                 study.study_name, stored.get('path_model'), provenance['path_model'])
    if provenance['partition'] and stored.get('partition') and stored['partition'] != provenance['partition']:
        log.warning('study %s was built against partition %s and this run uses %s: the values of its earlier '
                    'trials were measured on another train/validation/test split, so they are not comparable '
                    'with the new ones and the sampler is steered by them. Train with --refit for a fresh '
                    'study, or restore the partition it was built against.',
                    study.study_name, stored['partition'], provenance['partition'])
        study.set_user_attr('eforecast_partition_changed', provenance['partition'])
    return stored


def smallest(value, default):
    """The least a parameter can be: itself, its smallest choice, or the bottom of its range.

    A family's parameter is a number, or a search over numbers: a set of choices or a
    ``[low, high]`` range. Anything read from the family before a trial has drawn its own value
    has to cope with all three.
    """
    if isinstance(value, (set, frozenset, list, tuple)):
        numbers = [item for item in value if isinstance(item, (int, float)) and not isinstance(item, bool)]
        return min(numbers) if numbers else default
    return default if value is None else value


def pruner_for(static_data, method):
    """Median pruner; ``static_data[method]['pruner']`` may set n_startup_trials, n_warmup_steps or ``enabled``.

    The warm-up is a fifth of the epochs. When the epochs are themselves searched, it is a fifth
    of the shortest choice, so that the shortest trial can still be pruned; ``int()`` of the
    search space used to stop the run before its first trial.
    """
    params = static_data[method]
    cfg = dict(params.get('pruner') or {})
    if not cfg.get('enabled', True):
        return NopPruner()
    n_trials = int(smallest(params.get('n_trials', 1), 1))
    epochs = int(smallest(params.get('max_iterations', 100), 100))
    return MedianPruner(n_startup_trials=int(cfg.get('n_startup_trials', max(3, n_trials // 4))),
                        n_warmup_steps=int(cfg.get('n_warmup_steps', max(5, epochs // 5))),
                        interval_steps=int(cfg.get('interval_steps', 1)))


COMPARABILITY_KEY = 'comparability_id'


def completed_compatible(study, comparability=None, *, unit_identity=None):
    """``(compatible, finished, unverified)`` trial numbers of a study, toward the current budget.

    A trial carries the identity of the inputs it was trained on. One that predates the identity
    carries none: it is counted, because a history is not discarded for having been written before
    this bookkeeping existed, and it is reported as unverified so the count can be read honestly.

    ``comparability`` is the id of the unit's inputs. A family's is a ``lineage.Inputs``, which
    also knows the names the same inputs were recorded under by an earlier formula.
    """
    finished = [t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]
    unverified = [t.number for t in finished if t.user_attrs.get(COMPARABILITY_KEY) is None]

    def same_inputs(trial):
        return lineage.names(comparability, trial.user_attrs.get(COMPARABILITY_KEY))

    def bound(trial):
        """Whether the trial was fitted on what its unit has now (``cpu_scope.Binding``)."""
        recorded = trial.user_attrs.get('cpu_unit_identity')
        if unit_identity is None:
            return True
        if isinstance(unit_identity, str):
            return recorded in (None, unit_identity)
        return unit_identity.admits(trial.number, recorded,
                                    same_family=comparability is not None and same_inputs(trial))
    compatible = [t.number for t in finished
                  if (comparability is None or t.user_attrs.get(COMPARABILITY_KEY) is None or same_inputs(t))
                  and bound(t)]
    if unit_identity is not None:
        unverified = sorted(set(unverified) | {t.number for t in finished if t.user_attrs.get('cpu_unit_identity') is None})
    return compatible, [t.number for t in finished], unverified


def budget_state(static_data, study, method, comparability=None, *, unit_identity=None):
    """How much of this method's trial budget is already done with compatible work."""
    wanted = int(static_data[method]['n_trials'])
    compatible, finished, unverified = completed_compatible(study, comparability, unit_identity=unit_identity)
    return {'wanted': wanted, 'done': len(compatible), 'finished': len(finished),
            'incompatible': len(finished) - len(compatible), 'unverified': len(unverified),
            'missing': max(0, wanted - len(compatible)),
            'comparability_id': None if comparability is None else str(comparability)}


def begins_again(state):
    """Whether a study with this budget is trained from the beginning rather than continued.

    A family's trials all belong to one configuration of the family. A study that holds
    finished trials of another one -- its parameters or its search space were changed, or the
    data under it was built again -- is not continued beside them: they are removed and the
    search starts over. That is what "this family is retrained" means, for the queue, for the
    plan a person approves and for what the combination step may read.
    """
    return bool(state) and state.get('incompatible', 0) > 0


def may_begin_again(answer, method):
    """Whether a run was told it may start the studies of ``method`` over.

    The answer is given to a list: the families the plan said would be trained from the
    beginning. So that is what it covers -- the names in ``answer``. A study that turns out to
    hold trials of another configuration *without* its family having been on that list is not
    removed on the strength of an answer that was given about others; it stops the run, as it
    does when nobody was asked. ``True`` answers for every family: a caller that has no list.
    """
    if answer is True:
        return True
    if isinstance(answer, str):
        return answer == method
    return bool(answer) and method in answer


class OtherConfiguration(ValueError):
    """A study holds trials of another configuration and nobody was asked whether they may go."""


def refuse_other_configuration(cluster_name, method, state):
    return OtherConfiguration(
        f"{cluster_name} {method}: {state['incompatible']} finished trial(s) of its study were trained under "
        f"another configuration of {method} or on other data. They are neither continued nor removed by a "
        f"run that did not name {method} before it started. Start the training with run_training.py, which "
        f"says what it removes before it does, or train this method again from nothing with --refit.")


def describe_budget(label, state):
    return (f"{label}: {state['done']} of {state['wanted']} trials done"
            + (f", {state['incompatible']} from another revision" if state['incompatible'] else '')
            + (f", {state['unverified']} from before this bookkeeping" if state['unverified'] else ''))


def load_study(static_data, cluster_name, cluster_dir, method, storage_url=None, seed=None):
    seed = static_data.get('split_seed', 1) if seed is None else seed
    # multivariate + group TPE: parameters that exist together (e.g. the layer sizes of one experiment
    # tag) are sampled jointly; Optuna still labels the options experimental, the interface is stable
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    sampler = TPESampler(multivariate=True, group=True, seed=seed,
                         n_startup_trials=max(3, int(static_data[method].get('n_trials', 1)) // 4))
    storage = storage_for(cluster_dir, method, storage_url, static_data)
    digest, provenance = study_identity(static_data, cluster_name, method)
    name = resolve_study_name(static_data, cluster_name, method, storage)
    if name != study_name(static_data, cluster_name, method):
        # Optuna cannot rename a study, so the history is adopted where it already lives: a naming
        # fix must not quietly abandon a model that has been training for weeks.  From now on the
        # identity is recorded on it, so another tree meeting it in a shared storage is refused.
        log.warning('study %s predates the qualified names and is adopted as it stands; its identity %s is '
                    'recorded on it now', name, digest)
    study = optuna.create_study(study_name=name, storage=storage, direction='minimize',
                                sampler=sampler, pruner=pruner_for(static_data, method), load_if_exists=True)
    study._eforecast_storage_reference = ({'kind': 'url', 'url': storage_url} if storage_url else
                                         {'kind': 'journal', 'path': os.path.abspath(journal_path(cluster_dir, method, static_data))})
    check_study_identity(study, provenance)
    return study


def read_study(static_data, cluster_name, cluster_dir, method, storage_url=None):
    """An existing study, opened read-only: nothing is created, adopted or stamped.

    ``load_study`` is the execution path: it creates what is missing and records identity and
    provenance on what it finds. A planner or a preview must do neither, so it reads instead.
    """
    storage = storage_for(cluster_dir, method, storage_url, static_data)
    for name in study_candidates(static_data, cluster_name, method):
        try:
            return optuna.load_study(study_name=name, storage=storage)
        except KeyError:
            continue
    return None


def delete_study(static_data, cluster_name, cluster_dir, method, storage_url=None):
    """A fresh search: the study, the legacy trial records, the trial folders and the results table go."""
    import shutil
    if storage_url:
        for name in (study_name(static_data, cluster_name, method),
                     legacy_study_name(static_data, cluster_name, method)):
            try:
                optuna.delete_study(study_name=name, storage=storage_url)
            except KeyError:
                pass
    else:
        path = journal_path(cluster_dir, method, static_data)
        for file in (path, path + '.lock'):
            if os.path.exists(file):
                os.remove(file)
    method_dir = str(trial_layout.method_directory(static_data, cluster_dir, method))
    if os.path.isdir(method_dir):
        for name in os.listdir(method_dir):
            full = os.path.join(method_dir, name)
            if name == 'trials' or name.startswith('test_'):
                shutil.rmtree(full, ignore_errors=True)
    results = str(trial_layout.results_file(static_data, cluster_dir, cluster_name, method))
    if os.path.exists(results):
        os.remove(results)
    log.info('%s %s: study, trials and trial folders removed for a fresh search', cluster_name, method)


# ------------------------------------------------------------------ space <-> optuna

def _sorted_choices(values):
    """Categorical choices in a process-independent order (sets iterate in hash order, which differs per process)."""
    return sorted((_plain(v) for v in values), key=lambda v: (v is None, type(v).__name__, str(v)))


def distribution(spec):
    kind = spec['type']
    if kind == 'bool':
        return CategoricalDistribution([False, True])
    if kind in ('cat', 'ordinal'):
        return CategoricalDistribution(_sorted_choices(spec['values']))
    low, high = spec['range'][0], spec['range'][-1]
    if kind == 'int':
        return IntDistribution(int(low), int(high))
    return FloatDistribution(float(min(low, high)), float(max(low, high)))


def _plain(value):
    """Optuna categorical choices must be None, bool, int, float or str."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def suggest_space(trial, space):
    return {name: trial._suggest(name, distribution(spec)) for name, spec in space.items()}


def suggest_params(trial, objective):
    """``(params, structure, experiment_tag)`` suggested for ``trial`` from the objective's spaces."""
    params = suggest_space(trial, objective.space)
    tag = params.get('experiment_tag', objective.fix_params.get('experiment_tag'))
    structure = suggest_space(trial, objective.space_structure.get(tag, {}))
    return params, structure, tag


def distributions_for(objective, tag):
    dists = {name: distribution(spec) for name, spec in objective.space.items()}
    dists.update({name: distribution(spec) for name, spec in objective.space_structure.get(tag, {}).items()})
    return dists


# ------------------------------------------------------------------ history and records

def import_legacy_trials(study, objective, path_trials):
    """Add the trials of the former hyperopt loop to an empty study; returns how many were added."""
    if len(study.trials) > 0 or not os.path.isdir(path_trials):
        return 0
    added = 0
    for file in sorted(glob.glob(os.path.join(path_trials, 'trial*.pickle'))):
        try:
            record = joblib.load(file)
            tag = record.get('experiment_tag', objective.fix_params.get('experiment_tag'))
            dists = distributions_for(objective, tag)
            params = {name: _plain(record[name]) for name in dists if name in record}
            dists = {name: dist for name, dist in dists.items() if name in params}
            for name, dist in dists.items():
                if not dist._contains(dist.to_internal_repr(params[name])):
                    raise ValueError(f'{name}={params[name]!r} outside the current space')
            value = float(record.get('value', np.inf))
            if math.isfinite(value):
                trial = create_trial(params=params, distributions=dists, value=value,
                                     user_attrs={'legacy_file': os.path.basename(file)})
            else:
                trial = create_trial(params=params, distributions=dists, state=TrialState.FAIL,
                                     user_attrs={'legacy_file': os.path.basename(file)})
            study.add_trial(trial)
            added += 1
        except Exception as error:
            log.warning('legacy trial %s not imported: %s', file, error)
    if added:
        log.info('study %s: imported %d legacy trials from %s', study.study_name, added, path_trials)
    return added


def trial_counts(study):
    states = [t.state for t in study.get_trials(deepcopy=False)]
    return {'complete': states.count(TrialState.COMPLETE), 'pruned': states.count(TrialState.PRUNED),
            'running': states.count(TrialState.RUNNING), 'fail': states.count(TrialState.FAIL),
            'waiting': states.count(TrialState.WAITING), 'total': len(states)}


# ------------------------------------------------------------------ the attempts of a study

#: A study is asked for at most this many times the trials wanted of it. That is what ends a
#: study whose every trial fails: without it the scheduler would ask it for ever.
ASK_BUDGET_FACTOR = 2
#: Why a failed trial is not one of its study's attempts: it says nothing about the
#: hyper-parameters that were drawn. ``cancelled`` and ``abandoned`` -- the run was stopped, or
#: ended, under the trial; ``interrupted`` -- the run died and a later one found the trial still
#: RUNNING (such a trial carries no reason at all); ``card_lost`` -- the driver lost the card the
#: trial was on. Each is bounded by something other than the study, which is what allows it not
#: to count: a run stops once, and a lost card is given no further trial (``Orchestrator.lose_card``).
#: Every other reason counts -- out of memory, an error of the trial's own, a preparation or a
#: launch that failed -- because a study that fails like that every time has to end.
NOT_AN_ATTEMPT = ('cancelled', 'abandoned', 'interrupted', 'card_lost')
#: A study's own record of failed trials that are not its attempts either, ``{trial number: why}``
#: (``excuse``). It is for what is found out after a trial was told: a finished trial cannot be
#: written to, a study can.
EXCUSED_KEY = 'eforecast_attempts_excused'


def excused_trials(study):
    """``{trial number: why}`` of the failed trials a study was told not to count (``excuse``)."""
    recorded = study.user_attrs.get(EXCUSED_KEY)
    out = {}
    for number, why in (recorded.items() if isinstance(recorded, dict) else ()):
        try:
            out[int(number)] = str(why)
        except (TypeError, ValueError):
            continue
    return out


def not_an_attempt(trial, excused=None):
    """Why a trial is not one of its study's attempts, or ``None`` when it is one."""
    if trial.state != TrialState.FAIL:
        return None
    if excused and trial.number in excused:
        return excused[trial.number]
    reason = trial.user_attrs.get('fail_reason')
    if reason is None:
        return 'interrupted'
    return reason if reason in NOT_AN_ATTEMPT else None


def attempts(study, wanted, limited=True):
    """How a study stands against the trials wanted of it and the attempts it is allowed.

    One definition for the scheduler, which asks by it (``StudyUnit.remaining``), and for the
    pages that say what is left (``eforecast.config.study_progress``): they used to count
    separately, and the pages promised trials the scheduler would never ask for.

    ``counted`` are the attempts: every trial but the failed ones that say nothing about their
    hyper-parameters (``not_an_attempt``). ``ask`` is what the scheduler may ask for now.
    ``left`` is what the study will still be given -- the trials running, and the ones still to
    be asked as far as the attempts last -- and ``given_up`` what it owes beyond that.
    ``limited`` is false for a study nothing bounds: the methods trained on the processors are
    asked for what is missing at every run. ``wanted`` may be unknown (``None``).
    """
    excused = excused_trials(study)
    counts = {'complete': 0, 'pruned': 0, 'running': 0, 'waiting': 0, 'failed': 0, 'excused': 0}
    for trial in study.get_trials(deepcopy=False):
        if trial.state == TrialState.FAIL:
            counts['excused' if not_an_attempt(trial, excused) else 'failed'] += 1
        else:
            counts[{TrialState.COMPLETE: 'complete', TrialState.PRUNED: 'pruned',
                    TrialState.RUNNING: 'running'}.get(trial.state, 'waiting')] += 1
    done = counts['complete'] + counts['pruned']
    counted = sum(counts.values()) - counts['excused']
    out = dict(counts, done=done, counted=counted, wanted=wanted, allowed=None, owed=None, ask=None, left=None,
               given_up=None)
    if wanted is None:
        return out
    allowed = ASK_BUDGET_FACTOR * wanted if limited else None
    owed = max(0, wanted - done)
    spent = allowed is not None and counted >= allowed
    running = min(owed, counts['running'])
    further = owed - running if allowed is None else min(owed - running, max(0, allowed - counted))
    out.update(allowed=allowed, owed=owed, ask=0 if spent else max(0, wanted - done - counts['running']),
               left=running + further, given_up=owed - running - further)
    return out


def excuse(study, numbers, why):
    """Record that failed trials of a study are not its attempts; returns the ones newly recorded.

    For a failure that turned out to be the machine's after the study had been told -- a card
    that was lost with the scheduler still starting trials on it, as on 19 and 23 November 2026.
    Only a failed trial can be excused, and the record is the study's own, kept with it.
    """
    known = excused_trials(study)
    failed = {trial.number for trial in study.get_trials(deepcopy=False) if trial.state == TrialState.FAIL}
    fresh = sorted({int(number) for number in numbers} & failed - set(known))
    if fresh:
        merged = {str(number): reason for number, reason in sorted(known.items())}
        merged.update({str(number): str(why) for number in fresh})
        study.set_user_attr(EXCUSED_KEY, merged)
    return fresh


def write_trial_record(path_trials, trial_number, params, structure, result):
    """The legacy per-trial record read by ``get_results``."""
    os.makedirs(path_trials, exist_ok=True)
    record = {'trial_number': trial_number, 'duration': result.get('duration'), 'value': result.get('value'),
              'mae_test': result.get('mae_test'), 'mae_val': result.get('mae_val'),
              'sse_val': result.get('sse_val'), 'sse_test': result.get('sse_test')}
    record.update(params)
    record.update(structure)
    joblib.dump(record, os.path.join(path_trials, f'trial{trial_number}.pickle'))
    return record


def tell_once(study, trial, value=None, state=None):
    """Tell the study how a trial ended, unless something already has.

    Two things can settle the same trial: this run, and a restart that found its result on disk and
    adopted it.  Telling twice raises ``Cannot tell a COMPLETE trial`` and used to take the whole
    run down with it, so completion is idempotent here: the second caller is told it was too late.
    """
    settled = tuple(f'Cannot tell a {state_.name} trial' for state_ in TrialState if state_.is_finished())
    try:
        if state is None:
            study.tell(trial, value)
        else:
            study.tell(trial, state=state)
        return True
    except ValueError as error:
        # only "already finished"; a trial that was never created, or any other refusal, is a bug
        if not str(error).startswith(settled):
            raise
        number = trial if isinstance(trial, int) else getattr(trial, 'number', trial)
        log.info('study %s: trial %s was already settled by someone else (%s)', study.study_name, number, error)
        return False


def read_trial_result(trial):
    """The ``trial_result.json`` beside a trial's weights, or ``None``."""
    path_weights = trial.user_attrs.get('path_weights')
    result_file = os.path.join(path_weights, 'trial_result.json') if path_weights else None
    if not result_file or not os.path.exists(result_file):
        return None
    try:
        with open(result_file, encoding='utf-8') as stream:
            result = json.load(stream)
        return result if isinstance(result, dict) else None
    except (OSError, ValueError):
        return None


def adoptable(trial, study):
    """Whether a trial's folder is finished work, so that adopting its result is sound.

    A result file saying ``complete`` beside a folder whose manifest rejects it -- a later attempt
    that failed, a payload that no longer matches -- is not work to adopt; it is work to redo.
    """
    path_weights = trial.user_attrs.get('path_weights')
    usable, reason = completion.is_finished(path_weights) if path_weights else (False, 'no trial folder recorded')
    if not usable:
        log.warning('study %s: trial %d reports a complete result but its folder is not finished work (%s)',
                    study.study_name, trial.number, reason)
    return usable


def settle_running(study, objective, path_trials, run_id=None, lease_seconds=None):
    """Trials left RUNNING by a previous run: adopt, fail, or leave to the worker still holding them.

    A finished result on disk is adopted whoever produced it -- a trial debugged by hand with
    ``trial_runner.py``, or one that finished while the orchestrator was down -- because the work
    exists and adopting it is idempotent.  What is left is failed **only when this run may have
    it**: its own lease, an expired one, or none at all.  A trial whose owner is still beating its
    heartbeat belongs to a live worker, and failing it is what made two machines unsafe to restart.
    """
    kwargs = {} if lease_seconds is None else {'lease_seconds': lease_seconds}
    adopted, failed, held = [], [], []
    for trial in study.get_trials(deepcopy=False):
        if trial.state != TrialState.RUNNING:
            continue
        result = read_trial_result(trial)
        if result and result.get('state') == 'complete' and result.get('value') is not None \
                and adoptable(trial, study):
            if tell_once(study, trial.number, float(result['value'])):
                params = {k: v for k, v in trial.params.items() if k in objective.space}
                structure = {k: v for k, v in trial.params.items() if k not in objective.space}
                write_trial_record(path_trials, trial.number, params, structure, result)
                adopted.append(trial.number)
            continue
        if result and result.get('state') == 'pruned':
            if tell_once(study, trial.number, state=TrialState.PRUNED):
                adopted.append(trial.number)
            continue
        mine, lease = leases.recoverable(trial.user_attrs, run_id, **kwargs)
        if not mine:
            held.append(leases.describe(trial.number, lease))
            continue
        if tell_once(study, trial.number, state=TrialState.FAIL):
            failed.append(trial.number)
    if adopted:
        log.info('study %s: adopted %d trials finished outside the orchestrator: %s', study.study_name, len(adopted),
                 adopted)
    if failed:
        log.warning('study %s: %d stale running trials marked as failed: %s', study.study_name, len(failed), failed)
    if held:
        log.warning('study %s: %d running trials left alone, another worker still holds them: %s',
                    study.study_name, len(held), '; '.join(held))
    return adopted, failed


def fail_stale_running(study, run_id=None, lease_seconds=None):
    """Trials left RUNNING by a worker that died: mark them FAIL so they stop counting as in progress.

    Trials a live worker still holds are left alone, so this is safe to call while another machine
    is working on the same study.
    """
    kwargs = {} if lease_seconds is None else {'lease_seconds': lease_seconds}
    stale, held = [], []
    for trial in study.get_trials(deepcopy=False):
        if trial.state != TrialState.RUNNING:
            continue
        mine, lease = leases.recoverable(trial.user_attrs, run_id, **kwargs)
        if not mine:
            held.append(leases.describe(trial.number, lease))
        elif tell_once(study, trial.number, state=TrialState.FAIL):
            stale.append(trial.number)
    if stale:
        log.warning('study %s: %d stale running trials marked as failed: %s', study.study_name, len(stale), stale)
    if held:
        log.warning('study %s: %d running trials left alone, another worker still holds them: %s',
                    study.study_name, len(held), '; '.join(held))
    return stale
