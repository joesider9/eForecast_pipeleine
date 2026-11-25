# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Deep-model training on the GPUs: one orchestrator run per call, all studies in one queue.

``train_on_gpus`` builds the study units (global methods first in the queue, then the cluster
methods, plus the transfer-learning tasks that wait for their global study) and hands them to
``Orchestrator``, which packs trials of every method on the GPUs by memory.  ``train_on_gpus_projects``
does the same for a list of projects (the ADMIE multi-farm run).  Results tables come from the
unchanged ``get_results``.

Environment: EFORECAST_OPTUNA_STORAGE (storage URL instead of the journal files),
EFORECAST_TRIAL_CMD (template to run a trial from the Docker image, see docker/run_trial.sh).
"""
import logging
import os

import joblib

from eforecast.common_utils import trial_layout
from eforecast.training.command_line.orchestrator import Orchestrator

log = logging.getLogger(__name__)


def gpu_list(static_data, gpu_id):
    if gpu_id is None:
        return list(range(int(static_data['n_gpus'])))
    return list(gpu_id) if isinstance(gpu_id, (list, tuple)) else [gpu_id]


def method_lists(static_data, global_method, cluster_method):
    if cluster_method is None:
        cluster_methods = [method for method, values in static_data['cluster_methods'].items() if values]
    else:
        cluster_methods = [cluster_method]
    if global_method is None:
        global_methods = [method for method, values in static_data['global_methods'].items() if values]
        global_methods += [cl_m for cl_m in cluster_methods if static_data[cl_m]['transfer_learning_from_global'] and
                           cl_m not in global_methods]
    else:
        global_methods = [global_method]
    from eforecast.config.capabilities import CPU_METHODS
    # GRAPH is the cross-site model: `run_site_graph.py` trains it over several cases and exports
    # it into each. It has no family of parameters here and no study, and a case that had it
    # switched on used to stop on KeyError 'GRAPH' before any of its own networks were queued.
    return [m for m in global_methods if m not in CPU_METHODS and m != 'GRAPH'], cluster_methods


def max_trials_per_gpu(static_data, methods, n_jobs):
    """Concurrent trials per GPU: the given n_jobs, else the largest n_jobs of the methods in the run."""
    if n_jobs is not None:
        return int(n_jobs)
    values = [int(static_data[m].get('n_jobs', 1)) for m in methods if m in static_data]
    return max(values + [1])


def add_project(orchestrator, static_data, cluster, global_method, cluster_method):
    """Queue the studies of one project: global methods, then cluster methods and their transfer tasks."""
    global_methods, cluster_methods = method_lists(static_data, global_method, cluster_method)
    global_dir = os.path.join(static_data['path_model'], 'global')
    os.makedirs(global_dir, exist_ok=True)
    for method in global_methods:
        orchestrator.add_studies(static_data, method, {'global': global_dir})
    if static_data['is_Fuzzy']:
        clusters = cluster if cluster is not None else trial_layout.cluster_folders(static_data)
        for method in cluster_methods:
            orchestrator.add_studies(static_data, method, clusters)
            orchestrator.add_transfer(static_data, method, clusters)
    return global_methods + cluster_methods


def train_on_gpus(static_data, cluster=None, global_method=None, cluster_method=None,
                  refit=False, gpu_id=None, n_jobs=None, storage_url=None, halve_batch_on_oom=False,
                  begin_again=False):
    """Train the deep models of one project: an Optuna study per (cluster, method), one orchestrator run.

    ``begin_again`` lets a study whose finished trials are of another configuration of its family
    be emptied and trained from the beginning. ``run_training.py`` passes it once the person
    starting the run has been shown what that removes: the names of the families the plan listed,
    which is as far as the answer reaches. Without it, or for a family that was not listed, such
    a study stops the run.
    """
    return train_on_gpus_projects([static_data], cluster=cluster, global_method=global_method,
                                  cluster_method=cluster_method, refit=refit, gpu_id=gpu_id, n_jobs=n_jobs,
                                  storage_url=storage_url, halve_batch_on_oom=halve_batch_on_oom,
                                  begin_again=begin_again)


def train_on_gpus_projects(static_data_list, cluster=None, global_method=None, cluster_method=None,
                           refit=False, gpu_id=None, n_jobs=None, storage_url=None, halve_batch_on_oom=False,
                           begin_again=False):
    """Train the deep models of several projects in one queue (every study of every project competes for the GPUs)."""
    first = static_data_list[0]
    gpus = gpu_list(first, gpu_id)
    log.info('Train deep learning models of %d project(s) on gpus %s', len(static_data_list), gpus)
    global_methods, cluster_methods = method_lists(first, global_method, cluster_method)
    # ``n_jobs`` given on the command line is the operator's own bound and is kept as one. The
    # ``n_jobs`` of the methods is only what applies when the cards cannot be measured: with
    # measurements, how many trials share a card is decided by how busy the card is.
    orchestrator = Orchestrator(gpus, first, refit=refit, storage_url=storage_url,
                                max_per_gpu=int(n_jobs) if n_jobs is not None else None,
                                declared_per_gpu=max_trials_per_gpu(first, global_methods + cluster_methods, None),
                                halve_batch=halve_batch_on_oom, begin_again=begin_again)
    for static_data in static_data_list:
        add_project(orchestrator, static_data, cluster, global_method, cluster_method)
    if not orchestrator.units:
        log.info('nothing to train: every study has its results table (refit to redo)')
        return orchestrator
    orchestrator.run()
    log.info('Training of deep models ends')
    return orchestrator
