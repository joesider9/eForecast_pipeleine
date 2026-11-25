# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Training of the shallow (CPU) models of every cluster: RF, CatBoost, lightgbm, lasso, RBFols, GA_RBFols.

One Optuna study per (cluster, method), persisted like the GPU studies (journal file in the
method folder, or the storage URL in EFORECAST_OPTUNA_STORAGE), so an interrupted study
continues where it stopped. Each study runs in a process of its own, with the threads the
dispatcher gave it (``cpu_dispatch``), and several run side by side. Every trial keeps its folder
(``test_<k>`` with ``net_weights.pickle`` and ``trial_result.json``); the evaluation step later
picks the best models on the unseen last period.

Search space: the data options (merge, compress, scaling, feature selection) and the method's
parameters as in the configuration; sets are categorical choices, two-element lists are ranges.
The sklearn/CatBoost/lightgbm default of a parameter is added to its choices or used to widen its
range once, when the space is built (the configuration is never modified).

    python run_training.py --config <case> --cpu [--cpu-method RF] [--refit]
"""
import copy
import contextlib
import json
import logging
import os
import shutil
import socket
import time
import traceback
import uuid
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from eforecast.shallow_models.shallow_model import ShallowModel
from eforecast.shallow_models.foundation_model import FOUNDATION_METHODS, FoundationModel
from eforecast.common_utils import artifact, completion, lineage, trial_layout, cpu_scope
from eforecast.config import run_reporting
from eforecast.config import model_compute  # installs the hosted foundation adapter
from eforecast.training.command_line import leases
from eforecast.datasets.data_preprocessing import partition_report
from eforecast.training.command_line.optuna_search import (COMPARABILITY_KEY, begins_again, budget_state,
                                                           may_begin_again,
                                                           check_study_identity, describe_budget, journal_path,
                                                           legacy_study_name, refuse_other_configuration,
                                                           resolve_study_name, storage_for, study_identity,
                                                           study_name, tell_once)

log = logging.getLogger(__name__)

DATA_OPTIONS = ('merge', 'compress', 'scale_nwp_method', 'scale_row_method', 'feature_selection_method')
RBF_METHODS = ('RBFols', 'GA_RBFols')
SKIP_FOUNDATION = {'n_trials', 'n_jobs'}   # everything else of a foundation family is a fixed parameter of the model


def make_model(static_data, path_weights, params, n_jobs, refit, train, online=False):
    if train and params['method'] in FOUNDATION_METHODS and static_data.get('cpu_study_device'):
        # Runtime placement belongs to this model instance; keep the case/search and
        # unit binding intact so a device override does not invalidate finished work.
        params = {**params, 'device': static_data['cpu_study_device']}
    cls = FoundationModel if params['method'] in FOUNDATION_METHODS else ShallowModel
    return cls(static_data, path_weights, params=params, online=online, train=train, n_jobs=n_jobs, refit=refit)
SKIP_PARAMS = {'n_trials', 'warming_iterations', 'n_jobs'}


def cpu_methods(static_data):
    return [m for m, on in (static_data.get('methods_cpu') or {}).items() if on]


def n_threads(static_data):
    """The configured ``n_cpus``, or every core but one: what a study fitted outside the dispatcher uses."""
    return int(static_data.get('n_cpus') or max(1, (os.cpu_count() or 2) - 1))


def library_defaults(static_data, method):
    """Default parameters of the library estimator, used to seed the search space."""
    if method == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor().get_params()
    if method == 'CatBoost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor().get_params()
    if method == 'lightgbm':
        import lightgbm as lgb
        return lgb.LGBMRegressor().get_params()
    if method == 'lasso':
        from sklearn.linear_model import LassoCV, MultiTaskLassoCV
        model = MultiTaskLassoCV() if static_data['horizon_type'] == 'short_term' else LassoCV()
        return model.get_params()
    return {}


class ShallowSpace:
    """The search space of (cluster, method): fixed values and Optuna suggestions, built once."""

    def __init__(self, static_data, method):
        self.static_data = static_data
        self.method = method
        self.fixed = {'what_data': 'row_all'}
        self.categorical = {}     # name -> choices (list)
        self.ranges = {}          # name -> (low, high, is_int)
        self.n_features = None    # GA_RBFols: one width per feature
        defaults = library_defaults(static_data, method)
        options = {'merge': static_data['nwp_data_merge'], 'compress': static_data['compress_data'],
                   'scale_nwp_method': static_data['scale_nwp_method'],
                   'scale_row_method': static_data['scale_row_method'],
                   'feature_selection_method': static_data['feature_selection_methods']}
        if method in FOUNDATION_METHODS:
            # zero-shot on the lstm data: the data options do not change the model, so one fixed value each
            self.fixed['what_data'] = 'lstm'
            for name, values in options.items():
                first = list(values)[0] if isinstance(values, (list, tuple, set)) and len(values) else values
                self.fixed[name] = None if isinstance(first, float) and np.isnan(first) else first
            for name, value in copy.deepcopy(static_data[method]).items():
                if name not in SKIP_FOUNDATION:
                    self.fixed[name] = sorted(value) if isinstance(value, set) else value
            return
        for name, values in options.items():
            self._add(name, list(values) if not isinstance(values, (str, type(None))) else [values])
        for name, value in copy.deepcopy(static_data[method]).items():
            if name in SKIP_PARAMS or (method in RBF_METHODS and name == 'width'):
                continue
            default = defaults.get(name)
            if isinstance(value, set):
                choices = sorted(value, key=lambda v: (type(v).__name__, v) if isinstance(v, (int, float)) else ('~', str(v)))
                if default is not None and default not in choices and type(default) in {type(c) for c in choices}:
                    choices.append(default)
                self._add(name, choices)
            elif isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value) \
                    and not isinstance(value[0], bool):
                low, high = min(value), max(value)
                if isinstance(default, (int, float)) and not isinstance(default, bool):
                    low, high = min(low, default), max(high, default)
                self.ranges[name] = (low, high, isinstance(value[0], int) and isinstance(value[1], int))
            elif isinstance(value, list):
                self._add(name, value)
            else:
                self.fixed[name] = value
        if method in RBF_METHODS:
            width = static_data[method]['width']
            self.width_range = (float(min(width)), float(max(width)))

    def _add(self, name, choices):
        choices = [None if (isinstance(c, float) and np.isnan(c)) else c for c in choices]
        if len(choices) > 1:
            self.categorical[name] = choices
        else:
            self.fixed[name] = choices[0] if choices else None

    @property
    def names(self):
        return list(self.categorical) + list(self.ranges)

    def suggest(self, trial):
        params = dict(self.fixed)
        for name, choices in self.categorical.items():
            params[name] = trial.suggest_categorical(name, choices)
        for name, (low, high, is_int) in self.ranges.items():
            if name == 'depth' and params.get('boosting_type') == 'Ordered' \
                    and self.static_data['horizon_type'] == 'short_term':
                high = min(high, 6)      # CatBoost limit for ordered boosting with MultiRMSE
            params[name] = trial.suggest_int(name, int(low), int(high)) if is_int \
                else trial.suggest_float(name, float(low), float(high))
        if params.get('bootstrap_type') == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.5, 6.0)
        elif 'bootstrap_type' in params:
            params['subsample'] = 0.9
        if self.method in RBF_METHODS:
            low, high = self.width_range
            if self.method == 'GA_RBFols' and self.n_features:
                params['width'] = np.array([trial.suggest_float(f'width_{i}', low, high, log=True)
                                            for i in range(self.n_features)])
            else:
                params['width'] = np.array([trial.suggest_float('width', low, high, log=True)])
        return params


def describe_space(space):
    parts = [f'{k}: {v}' for k, v in space.categorical.items()]
    parts += [f'{k}: [{lo}, {hi}]' for k, (lo, hi, _) in space.ranges.items()]
    if space.method in RBF_METHODS:
        parts.append(f'width: {space.width_range}{" per feature" if space.method == "GA_RBFols" else ""}')
    return ', '.join(parts) or 'no free parameters'


# ---------------------------------------------------------------------------------- one study

class Objective:
    def __init__(self, static_data, cluster_name, cluster_dir, method, space, threads, run_id=None):
        static_data = trial_layout.pin(static_data, cluster_dir, method, cluster_name)
        self.static_data = static_data
        self.cluster_name = cluster_name
        self.cluster_dir = cluster_dir
        self.method = method
        self.directory = trial_layout.method_directory(static_data, cluster_dir, method)
        self.space = space
        self.threads = threads
        self.run_id = run_id
        self.comparability = lineage.comparability_id(static_data, method=method)
        self.cv_masks = joblib.load(os.path.join(cluster_dir, 'cv_mask.pickle'))
        self.input_observations = lineage.input_observations(static_data, cluster_dir)
        # What a new trial records, and which of the trials already there were fitted on the same.
        self.binding = cpu_scope.Binding(static_data, cluster_dir, method)
        self.unit_identity = self.binding.identity

    def path_weights(self, number):
        return str(self.directory / f'test_{number}')

    def __call__(self, trial):
        run_reporting.check_cancelled()
        if run_reporting.context() is None:
            return self._call(trial)
        parent = run_reporting.node_id('study', f'{self.run_id}/{self.cluster_dir}/{self.method}')
        key = f'{self.run_id}/{self.cluster_dir}/{self.method}/{trial.number}'
        details = {'trial': trial.number, 'path_weights': self.path_weights(trial.number)}
        run_reporting.child('trial', key, parent=parent, identity=details, execution=run_reporting.process())
        try:
            value = self._call(trial)
        except BaseException as error:
            run_reporting.child('trial', key, parent=parent, identity=details,
                outcome={'kind': 'nothing_to_do' if isinstance(error, optuna.TrialPruned) else
                         'cancelled' if isinstance(error, KeyboardInterrupt) else 'failure',
                         'detail': f'{type(error).__name__}: {error}'})
            raise
        # The study emits terminal success after Optuna confirms its own write.
        return value

    def _call(self, trial):
        self._private = None
        try:
            with leases.Beating(trial, self.run_id):
                return self.run_trial(trial)
        except BaseException as error:
            if self._private is not None and os.path.isdir(self._private):
                from eforecast.datasets.data_preprocessing.partition_report import atomic_json
                with contextlib.suppress(OSError):
                    atomic_json(artifact.manifest_file(self._private),
                                {'version': artifact.VERSION, 'kind': 'cpu_trial', 'state': 'error', 'files': {}})
                    atomic_json(os.path.join(self._private, 'trial_result.json'),
                                {'state': 'error', 'detail': f'{type(error).__name__}: {error}'})
            raise

    def run_trial(self, trial):
        if lineage.inputs_changed(self.input_observations, lineage.input_observations(self.static_data, self.cluster_dir)):
            raise ValueError('CPU study inputs changed after mask attachment; start a new study invocation')
        params = self.space.suggest(trial)
        if self.method in FOUNDATION_METHODS:
            requested_batch = int(params.get('batch_size') or 256)
            params['batch_size'] = max(1, requested_batch) if params.get('cross_learning') else max(1, min(256, requested_batch))
        params.update({'method': self.method, 'cluster_name': self.cluster_name, 'cluster_dir': self.cluster_dir,
                       'unit': {'scope': 'global' if self.cluster_name == 'global' else 'cluster',
                                'cluster': self.cluster_name, 'method': self.method, 'kind': 'study',
                                'generation': self.directory.name if self.directory.parent.name == '.generations' else None}})
        path_weights = self.path_weights(trial.number)
        os.makedirs(path_weights, exist_ok=True)
        attempt = uuid.uuid4().hex
        trial.set_user_attr('eforecast_cpu_attempt', attempt)
        work = os.path.join(path_weights, '.attempt-cpu-' + attempt)
        os.makedirs(work)
        self._private = work
        trial.set_user_attr('path_weights', path_weights)
        trial.set_user_attr(COMPARABILITY_KEY, str(self.comparability))   # what this trial may be compared with
        trial.set_user_attr('cpu_unit_identity', self.unit_identity)
        start = time.time()
        inputs = lineage.dependencies(self.static_data, scope=self.cluster_dir, method=self.method,
                                      selectors=params.get('feature_selection_method') is not None)
        model = make_model(self.static_data, work, params, self.threads, refit=True, train=True)
        partition_report.write_training_scope(
            self.static_data, work, f'cpu:{self.method}', self.cluster_name,
            self.cv_masks, self.cv_masks, partition_report.load_global_masks(self.static_data),
            reason='CPU trainer passes supplied masks unchanged')
        try:
            sse = model.fit(self.cv_masks)
        except Exception as error:
            with open(os.path.join(work, 'error.txt'), 'w', encoding='utf-8') as stream:
                stream.write(''.join(traceback.format_exception(error)))
            raise
        if model.model is None or model.best_mae_test is None:
            raise optuna.TrialPruned('too few training samples or the estimator failed')
        if lineage.inputs_changed(self.input_observations, lineage.input_observations(self.static_data, self.cluster_dir)):
            raise ValueError('CPU trial inputs changed during fitting; private result was not published')
        result = {'state': 'complete', 'attempt': attempt, 'value': float(model.best_mae_test), 'mae_test': float(model.best_mae_test),
                  'mae_val': float(model.best_mae_val) if model.best_mae_val is not None else None,
                  'sse_test': float(model.sse_test) if getattr(model, 'sse_test', None) is not None else None,
                  'sse_val': float(model.sse_val) if getattr(model, 'sse_val', None) is not None else None,
                  'duration': time.time() - start}
        with open(os.path.join(work, 'trial_result.json'), 'w', encoding='utf-8') as stream:
            json.dump(result, stream, indent=1)
        # A metadata-write failure stays private and cannot bless an Optuna FAIL.
        for key in ('mae_test', 'mae_val', 'sse_test', 'sse_val', 'duration'):
            trial.set_user_attr(key, result[key])
        identity = artifact.worker_identity(path_weights, run_id=self.run_id,
                                            task=f'{self.cluster_name}/{self.method}', attempt=attempt)
        manifest = artifact.publish(work, kind='cpu_trial', state='complete',
                         value=result['value'], duration=result['duration'],
                         depends_on={**inputs, 'unit': params['unit'], 'cpu_unit_identity': self.unit_identity},
                         **identity)
        if manifest is None:
            raise OSError('CPU trial manifest publication failed; its private folder is kept')
        valid, reason = artifact.verify_digests(work)
        if not valid:
            raise ValueError(f'CPU trial is incomplete: {reason}')
        for name in os.listdir(work):
            with open(os.path.join(work, name), 'rb') as stream:
                os.fsync(stream.fileno())
        run_reporting.check_cancelled()
        def promote():
            from eforecast.datasets.data_preprocessing.partition_report import atomic_json
            marker = self.directory.parent.parent / trial_layout.MARKER if self.directory.parent.name == trial_layout.GENERATIONS else None
            previous = marker.read_bytes() if marker is not None and marker.exists() else None
            if lineage.inputs_changed(self.input_observations,
                                      lineage.input_observations(self.static_data, self.cluster_dir)):
                raise ValueError('CPU inputs changed before publication')
            atomic_json(artifact.manifest_file(path_weights),
                        {'version': artifact.VERSION, 'kind': 'cpu_trial', 'state': 'publishing', 'files': {}})
            try:
                names = [name for name in os.listdir(work) if name != artifact.MANIFEST_FILE]
                for name in [*sorted(names), artifact.MANIFEST_FILE]:
                    os.replace(os.path.join(work, name), os.path.join(path_weights, name))
                descriptor = os.open(path_weights, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                trial_layout.activate(path_weights)
            except BaseException:
                atomic_json(artifact.manifest_file(path_weights),
                            {'version': artifact.VERSION, 'kind': 'cpu_trial', 'state': 'publishing', 'files': {}})
                if marker is not None:
                    if previous is None:
                        marker.unlink(missing_ok=True)
                    else:
                        from eforecast.compute_worker import atomic_bytes
                        atomic_bytes(marker, previous)
                raise
        # Nothing is ever fitted in test_k itself: the trial is moved in whole, under the
        # folder's lock, once it is complete and verified.
        from eforecast.folder_lock import folder_lock
        with folder_lock(path_weights, timeout=2):
            promote()
        try:
            os.rmdir(work)
        except OSError as error:    # the published model is valid; an emptied folder stayed behind
            log.warning('the private folder of a published CPU trial was not removed: %s', error)
        return result['value']


def load_cpu_study(static_data, cluster_name, cluster_dir, method, storage_url=None, refit=False):
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    name = study_name(static_data, cluster_name, method)
    if refit:
        from eforecast.training.command_line.optuna_search import delete_study
        delete_study(static_data, cluster_name, cluster_dir, method, storage_url)
    directory = trial_layout.method_directory(static_data, cluster_dir, method)
    directory.mkdir(parents=True, exist_ok=True)
    storage = storage_url or storage_for(cluster_dir, method, static_data=static_data)
    # the same resolution as the GPU side: a study created before the identity was in the name
    # keeps that name, and one belonging to another model tree is refused rather than continued
    name = resolve_study_name(static_data, cluster_name, method, storage)
    sampler = TPESampler(seed=int(static_data.get('split_seed', 42)) + 1, multivariate=True, n_startup_trials=4)
    study = optuna.create_study(study_name=name, storage=storage, sampler=sampler, direction='minimize',
                                load_if_exists=True)
    study._eforecast_storage_reference = ({'kind': 'url', 'url': storage_url} if storage_url else
                                         {'kind': 'journal', 'path': os.path.abspath(journal_path(cluster_dir, method, static_data))})
    check_study_identity(study, study_identity(static_data, cluster_name, method)[1])
    return study


def settle_running(study, run_id=None, *, exclusive=False):
    """Trials left RUNNING by an interrupted run: adopt, fail, or leave to the worker holding them.

    The same rule as the GPU side: a finished result on disk is adopted whoever produced it, and
    what is left is failed only when this run may have it -- its own lease, an expired one, or
    none. A trial another live worker is still beating for is left exactly as it is.
    """
    for trial in study.get_trials(deepcopy=False):
        if trial.state != TrialState.RUNNING:
            continue
        result_file = os.path.join(trial.user_attrs.get('path_weights', ''), 'trial_result.json')
        result = None
        if os.path.exists(result_file):
            try:
                with open(result_file, encoding='utf-8') as stream:
                    result = json.load(stream)
                if not isinstance(result, dict):
                    result = None
            except (OSError, ValueError):
                pass
        usable, reason = completion.is_finished(trial.user_attrs.get('path_weights') or '')
        expected_attempt = trial.user_attrs.get('eforecast_cpu_attempt')
        if usable and expected_attempt is not None:
            manifest = artifact.read(os.path.dirname(result_file))
            produced_attempt = manifest.get('attempt') if manifest else (result or {}).get('attempt')
            if produced_attempt != expected_attempt or not os.path.isfile(os.path.join(os.path.dirname(result_file), 'net_weights.pickle')):
                usable, reason = False, 'published CPU artifact belongs to another attempt or is missing its weights'
        if result and result.get('state') == 'complete' and usable:
            if tell_once(study, trial.number, float(result['value'])):
                log.info('study %s: adopted trial %d from its result file', study.study_name, trial.number)
            continue
        if result and result.get('state') == 'complete':
            log.warning('study %s: trial %d reports a complete result but its folder is not finished work (%s)',
                        study.study_name, trial.number, reason)
        mine, lease = leases.recoverable(trial.user_attrs, run_id)
        # A study started by the dispatcher is the only one on its journal: a case is trained
        # by one run at a time, and a run's studies stop with it. It need not wait out the
        # beat of a trial nobody is fitting.
        mine = mine or exclusive
        if not mine:
            log.warning('study %s: %s, leaving it alone', study.study_name, leases.describe(trial.number, lease))
        elif tell_once(study, trial.number, state=TrialState.FAIL):
            log.warning('study %s: interrupted trial %d marked as failed', study.study_name, trial.number)


def results_table(study):
    """The per-(cluster, method) results table: one row per finished trial, best first."""
    df = study.trials_dataframe(attrs=('number', 'value', 'state', 'params', 'user_attrs'))
    if df.empty:
        return pd.DataFrame(columns=['trial_number', 'number', 'value', 'state'])
    df = df[df['state'] == 'COMPLETE'].copy()
    df.insert(0, 'trial_number', df['number'])
    rename = {c: c.replace('user_attrs_', '') for c in df.columns if c.startswith('user_attrs_')}
    df = df.rename(columns=rename)
    df = df.drop(columns=[c for c in ('path_weights',) if c in df.columns])
    return df.sort_values('value')


def _train_study(static_data, cluster_name, cluster_dir, method, storage_url=None, refit=False, run_id=None,
                 threads=None, exclusive=False, begin_again=False):
    """One study, continued from its journal.

    ``begin_again`` is the answer to a question the run asked before it started: a study that
    holds finished trials of another configuration of its family is trained from the beginning,
    and those trials are removed first. Without that answer such a study is refused rather than
    continued beside them, or emptied unasked. The answer covers the families the accepted plan
    named (``optuna_search.may_begin_again``), which is decided where the studies are started.
    """
    results_file = str(trial_layout.results_file(static_data, cluster_dir, cluster_name, method))
    study = load_cpu_study(static_data, cluster_name, cluster_dir, method, storage_url, refit)
    run_reporting.child('study', f'{run_id}/{cluster_dir}/{method}', identity={'study_name': study.study_name})
    settle_running(study, run_id=run_id, exclusive=exclusive)
    before = {trial.number for trial in study.trials}
    space = ShallowSpace(static_data, method)
    threads = threads or n_threads(static_data)
    objective = Objective(static_data, cluster_name, cluster_dir, method, space, threads, run_id=run_id)
    if method == 'GA_RBFols':
        probe = ShallowModel(static_data, os.path.join(cluster_dir, method), params=dict(
            space.fixed, method=method, cluster_name=cluster_name, cluster_dir=cluster_dir,
            **{k: v[0] for k, v in space.categorical.items()}), online=False, train=True, n_jobs=1, refit=True)
        X, _ = probe.load_data()
        space.n_features = X.shape[1]
    binding = getattr(objective, 'binding', None) or objective.unit_identity
    state = budget_state(static_data, study, method, objective.comparability, unit_identity=binding)
    if begins_again(state) and not refit:
        if not may_begin_again(begin_again, method):
            raise refuse_other_configuration(cluster_name, method, state)
        log.warning('%s %s: %d finished trial(s) are of another configuration of %s; they are removed and the '
                    'study starts from the beginning', cluster_name, method, state['incompatible'], method)
        study = load_cpu_study(static_data, cluster_name, cluster_dir, method, storage_url, True)
        before = {trial.number for trial in study.trials}
        state = budget_state(static_data, study, method, objective.comparability, unit_identity=binding)
    log.info('%s %s: %d threads, %s, space %s', cluster_name, method, threads,
             describe_budget(f'{cluster_name} {method}', state), describe_space(space))
    if state['missing']:
        def reported(study, trial):
            if trial.state != TrialState.COMPLETE:
                return  # the objective already reported its failure/pruning detail
            parent = run_reporting.node_id('study', f'{run_id}/{cluster_dir}/{method}')
            run_reporting.child('trial', f'{run_id}/{cluster_dir}/{method}/{trial.number}', parent=parent,
                outcome={'kind': 'success'})
        study.optimize(objective, n_trials=state['missing'], gc_after_trial=True,
                       catch=(Exception,), callbacks=[reported])
    table = results_table(study)
    if len(table):
        directory = trial_layout.method_directory(static_data, cluster_dir, method)
        if directory.parent.name == trial_layout.GENERATIONS:
            usable = [directory / f'test_{t.number}' for t in study.trials if t.state == TrialState.COMPLETE
                      and completion.is_finished(str(directory / f'test_{t.number}'))[0]
                      and artifact.verify_digests(str(directory / f'test_{t.number}'))[0]]
            if usable:
                trial_layout.activate(usable[0])
    def publish_table():
        temporary = results_file + '.tmp-' + uuid.uuid4().hex
        try:
            table.to_csv(temporary)
            with open(temporary, 'rb') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, results_file)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    publish_table()
    counts = {s.name.lower(): sum(1 for t in study.trials if t.state == s)
              for s in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)}
    log.info('%s %s: done, best mae_test %.4f (trial %s); %s', cluster_name, method,
             table['value'].iloc[0] if len(table) else float('nan'),
             int(table['trial_number'].iloc[0]) if len(table) else '-', counts)
    fresh = [trial for trial in study.trials if trial.number not in before]
    current = {state.name.lower(): sum(trial.state == state for trial in fresh)
               for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)}
    done = current['complete'] + current['pruned']
    kind = ('partial' if current['fail'] else 'success') if done else 'failure' if current['fail'] else 'nothing_to_do'
    table.attrs['run_outcome'] = {'kind': kind, 'counts': current}
    return table


# Keep public imports compatible; the normal caller remains a metadata-only dispatcher.
from eforecast.training.cpu_dispatch import train_study, train_on_cpus
