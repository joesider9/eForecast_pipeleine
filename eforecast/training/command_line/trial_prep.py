# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Preparation and bookkeeping of one trial, shared by the orchestrator and ``run_training.py --prepare-only``.

* ``prepare_study``: the study of a (cluster, method) ready to be asked (legacy trials imported,
  grid points enqueued, trials finished outside the orchestrator adopted)
* ``prepare_trial``: sample a trial, write ``test_<k>/parameters.pickle``
* ``forward_progress``: report the newest epoch of a running trial to the pruner, raise ``prune.flag``
* ``read_result``: the ``trial_result.json`` of a finished trial
"""
import json
import logging
import os
import shutil

import joblib

from eforecast.common_utils import artifact, completion, lineage, trial_layout
from eforecast.training.command_line.optuna_search import (COMPARABILITY_KEY, delete_study, import_legacy_trials, load_study,
                                                           settle_running, suggest_params, write_trial_record)
from eforecast.training.command_line.search_space import SearchSpace

log = logging.getLogger(__name__)
record_trial = write_trial_record


def prepare_study(static_data, cluster_name, cluster_dir, method, storage_url=None, refit=False, run_id=None):
    """The study of (cluster, method) ready to be asked: legacy trials imported, grid points enqueued.

    ``run_id`` is this run's identity: settlement recovers its own and expired work, and leaves a
    trial another live worker still holds exactly where it is.
    """
    static_data = trial_layout.pin(static_data, cluster_dir, method, cluster_name)
    directory = trial_layout.method_directory(static_data, cluster_dir, method)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(static_data, directory / 'static_data.pickle')
    if directory.parent.name != trial_layout.GENERATIONS:
        joblib.dump(static_data, os.path.join(cluster_dir, 'static_data.pickle'))
    if refit:
        delete_study(static_data, cluster_name, cluster_dir, method, storage_url)
    objective = SearchSpace(static_data, cluster_name, cluster_dir, method)
    study = load_study(static_data, cluster_name, cluster_dir, method, storage_url)
    path_trials = str(directory / 'trials')
    if not refit:
        settle_running(study, objective, path_trials, run_id=run_id)
    import_legacy_trials(study, objective, path_trials)
    n_trials = int(static_data[method]['n_trials'])
    if static_data[method].get('fix_grid_search', False) and not study.user_attrs.get('grid_enqueued'):
        for point in objective.grid_()[:n_trials]:
            study.enqueue_trial({k: v for k, v in point.items() if k in objective.space}, skip_if_exists=True)
        study.set_user_attr('grid_enqueued', True)
    return study, objective, path_trials


def prepare_trial(static_data, study, trial, objective, cluster_dir, method, refit, gpu, *, persist=True):
    """Sample the trial, write ``test_<k>/parameters.pickle``; returns path, params, structure, tag, experiment."""
    static_data = getattr(objective, 'static_data', static_data)
    params, structure, tag = suggest_params(trial, objective)
    experiment_params = objective.init_experiment_params(trial.number, params, tag, structure)
    experiment_params['experiment'] = objective.select_structure(structure, tag, static_data['experiments'][tag])
    experiment_params['inputs'] = [key for key in experiment_params['experiment'].keys() if 'output' not in key]
    path_weights = str(trial_layout.trial_directory(static_data, cluster_dir, method, trial.number))
    if persist and os.path.exists(path_weights) and refit:
        shutil.rmtree(path_weights)
    os.makedirs(path_weights, exist_ok=True)
    # Pixel loading/batches belong to the attached trial, never the scheduler.
    if persist:
        joblib.dump(experiment_params, os.path.join(path_weights, 'parameters.pickle'))
    # This family's own id: a sibling's parameter change must not make this trial incomparable.
    trial.set_user_attr(COMPARABILITY_KEY, str(lineage.comparability_id(static_data, method=method)))
    trial.set_user_attr('path_weights', path_weights)
    trial.set_user_attr('experiment_tag', tag)
    return path_weights, params, structure, tag, experiment_params


#: A process the kernel killed, seen directly or through the docker client (128 + SIGKILL).
KILLED = (-9, 137)


def exit_meaning(returncode):
    """What the exit status of a child that left no result says, for whoever reads the failure."""
    if returncode in KILLED:
        return ('it was killed (SIGKILL); when nobody stopped it, that is the kernel ending the largest '
                'process because the machine ran out of memory')
    if isinstance(returncode, int) and returncode < 0:
        return f'it was ended by signal {-returncode}'
    return None


def read_result(path_weights, returncode, *, allow_staged=False):
    """``trial_result.json`` of a finished child, or an error record when it left none."""
    result_file = os.path.join(path_weights, 'trial_result.json')
    if os.path.exists(result_file):
        try:
            with open(result_file, encoding='utf-8') as stream:
                result = json.load(stream)
            if not isinstance(result, dict):
                raise ValueError('result must be a mapping')
        except (OSError, ValueError) as error:
            result = {'state': 'error', 'value': None, 'error': f'trial_result.json cannot be read ({error})'}
    else:
        meaning = exit_meaning(returncode)
        result = {'state': 'error', 'value': None,
                  'error': f'no trial_result.json (exit code {returncode})' + (f': {meaning}' if meaning else '')}
    if result.get('state') in artifact.USABLE_STATES:
        # What the child says it did, checked against what it left behind.
        usable, reason = completion.is_finished(path_weights, allow_staged=allow_staged)
        if not usable:
            result = {**result, 'state': 'error', 'value': None,
                      'error': f"the trial reported {result['state']!r} but its folder is not finished work ({reason})"}
    result['returncode'] = returncode
    return result


def forward_progress(trial, progress_file, last_epoch, path_weights):
    """Report the newest epoch to the study; raise the prune flag when the pruner says so."""
    if not os.path.exists(progress_file):
        return last_epoch
    try:
        with open(progress_file, encoding='utf-8') as stream:
            progress = json.load(stream)
    except (OSError, ValueError):
        return last_epoch
    epoch = int(progress.get('epoch', -1))
    if epoch <= last_epoch:
        return last_epoch
    trial.report(float(progress['best_mae_test']), step=epoch)
    if trial.should_prune() and not os.path.exists(os.path.join(path_weights, 'prune.flag')):
        open(os.path.join(path_weights, 'prune.flag'), 'w').close()
        log.info('trial %d: pruned at epoch %d (best test mae %.4f)', trial.number, epoch, progress['best_mae_test'])
    return epoch


