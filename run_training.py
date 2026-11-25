#!/usr/bin/env python
# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Train the deep models of a case on the GPUs (the step after ``run_datasets.py --stage all``).

    python run_training.py --config configuration/cases/<case>.yaml
    python run_training.py --config <case> --refit                  # fresh studies, old trials removed
    python run_training.py --config <case> --docker                 # trials run from the eforecast-worker image
    python run_training.py --config <case> --storage postgresql://optuna:optuna@localhost:5432/optuna
    python run_training.py --config <case> --gpus 1 --n-jobs 2      # one GPU, two trials at a time
    python run_training.py --config <case> --global-method MLP      # one method only
    python run_training.py --config <case> --prepare-only           # write one trial per study, print the debug commands
    python run_training.py --config <case> --cpu [--cpu-method RF]  # the shallow models of every cluster

What happens: one Optuna study per (cluster, method) and one orchestrator that packs the trials
of every study on the GPUs (see docs/gpu_training_optuna.md).  Every trial is a separate process,
``eforecast/training/command_line/trial_runner.py --path-weights <test folder> --gpu 0``, which is
also the command to run a trial under the debugger.

Exit codes of a GPU run: ``0`` when trials completed (``succeeded``, or ``succeeded_with_failures``
when some were lost or some unit could not run here) and when there was nothing to train; ``1``
when the run completed nothing although it had work -- every trial failed, or every unit was
skipped as too large for this machine; ``2`` when the run was stopped by SIGINT or SIGTERM.  The
outcome is also the first field of ``orchestrator_summary.json``.
"""
import json
import logging
import os
import sys

path_pycharm = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_pycharm)

import joblib

from eforecast.common_utils.logger import quieted
from eforecast.init.initialize import initializer
from eforecast.config.entrypoint_parsers import parse_training_args
from eforecast.training.command_line import docker_task
from eforecast.config.capabilities import CPU_METHODS, CLUSTER_METHODS, GLOBAL_METHODS
from eforecast.common_utils import trial_layout

log = logging.getLogger('run_training')
default_config_file = os.environ.get('EFORECAST_CASE', 'configuration/cases/default.yaml')  # or a symlink named default.yaml

def parse_args(argv=None):
    return parse_training_args(argv, description=__doc__, default_config=default_config_file)

def validate_training_request(static_data, args, gpu_count=None, *, cards=True):
    """Validate runtime choices before creating folders, studies or worker processes.

    ``cards=False`` leaves out the one question that is about this machine and not about the
    request: whether the cards it names are here (``require_cards``). A run whose training
    stage is whole fits nothing, so it asks for none.
    """
    for method in (args.cpu_method, args.global_method, args.cluster_method):
        if method and method not in static_data:
            raise ValueError(f'{method} has no configured method_params family')
    for key in ('cpu_workers', 'cpu_study_threads', 'cpu_study_device'):
        if getattr(args, key, None) is not None and not args.cpu:
            raise ValueError('--' + key.replace('_', '-') + ' requires --cpu')
    if getattr(args, 'cpu_scope', None) and not args.cpu:
        raise ValueError('--cpu-scope requires --cpu')
    if args.cpu_method and not args.cpu:
        raise ValueError('--cpu-method requires --cpu')
    if args.cpu:
        from eforecast.common_utils.cpu_scope import methods
        glob, clusters = methods(static_data, getattr(args, 'cpu_scope', None), args.cpu_method)
        if not static_data['is_Fuzzy'] and (getattr(args, 'cpu_scope', None) or static_data.get('cpu_scope', 'clusters')) == 'clusters':
            raise ValueError('CPU methods currently train on fuzzy clusters; enable is_fuzzy and build clusters first')
        if args.prepare_only:
            raise ValueError('--prepare-only prepares deep trials; use it without --cpu')
        return
    if static_data.get('backend', 'TORCH') != 'TORCH':
        raise ValueError('the current deep-training CLI uses TORCH; legacy TF configurations remain readable')
    if args.gpus and len(set(args.gpus)) != len(args.gpus):
        raise ValueError('--gpus must not contain duplicate GPU ids')
    if args.prepare_only:
        return  # Preparing/debugging a trial does not require CUDA on this machine.
    if cards:
        require_cards(static_data, args, gpu_count)


def require_cards(static_data, args, gpu_count=None):
    """The cards a deep-training run would fit on have to be on this machine."""
    gpus = args.gpus if args.gpus is not None else list(range(static_data['n_gpus']))
    if not gpus:
        raise ValueError('no GPUs selected (n_gpus=0); use --cpu or supply --gpus for deep training')
    if gpu_count is None:
        import torch
        gpu_count = torch.cuda.device_count()
    invalid = [gpu for gpu in gpus if gpu >= gpu_count]
    if invalid:
        raise ValueError(f'GPU ids {invalid} are unavailable; CUDA sees {gpu_count} device(s)')


def mounts_for(static_data):
    """The ``-v`` arguments of a Docker trial, as a list.

    It used to be one shell string of quoted pairs, expanded unquoted by ``docker/run_trial.sh``:
    the shell split it on whitespace without re-reading the quotes, so a host folder containing a
    space became two arguments with apostrophes in them. The orchestrator builds the argv itself
    now and this is here for the script and for anything that still wants the list.
    """
    pairs = []
    for folder in docker_task.host_folders(static_data):
        pairs += ['-v', f'{folder}:{folder}']
    return pairs


def configure_docker(static_data, image):
    """Tell the orchestrator to launch trials in containers, and with which mounts."""
    folders = docker_task.host_folders(static_data)
    os.environ['EFORECAST_WORKER_IMAGE'] = image
    os.environ['EFORECAST_WORKER_FOLDERS'] = json.dumps(folders)
    os.environ.pop('EFORECAST_TRIAL_CMD', None)      # the argv is built in Python, not by a template
    os.environ['EFORECAST_WORKER_MOUNTS'] = '\n'.join(folders)   # for docker/run_trial.sh, one per line
    if not docker_task.docker_available():
        raise SystemExit('--docker was asked for but docker is not usable here')
    log.info('docker trials from %s (%s) mounting %s', image, docker_task.image_identity(image) or '?',
             ', '.join(folders))


def prepare_only(static_data, args, *, refit=None):
    """One asked trial per study with its parameters.pickle on disk, and the command to run it."""
    from eforecast.training.command_line.trial_prep import prepare_study, prepare_trial
    from eforecast.training.command_line.train_manager_for_gpu import method_lists
    global_methods, cluster_methods = method_lists(static_data, args.global_method, args.cluster_method)
    targets = [('global', os.path.join(static_data['path_model'], 'global'), m) for m in global_methods]
    if static_data['is_Fuzzy']:
        from eforecast.common_utils import trial_layout
        clusters = trial_layout.cluster_folders(static_data)
        targets += [(name, path, m) for m in cluster_methods for name, path in clusters.items()]
    runner = os.path.join(path_pycharm, 'eforecast', 'training', 'command_line', 'trial_runner.py')
    commands = []
    refit = args.refit if refit is None else refit
    from eforecast.config import run_reporting
    for cluster_name, cluster_dir, method in targets:
        if os.path.exists(str(trial_layout.results_file(static_data, cluster_dir, cluster_name, method))) and not refit:
            continue
        study, objective, _ = prepare_study(static_data, cluster_name, cluster_dir, method, args.storage, refit)
        trial = study.ask()
        parent = run_reporting.child('study', study.study_name, identity={'study_name': study.study_name,
            'cluster': cluster_name, 'method': method, 'cluster_dir': cluster_dir})
        path_weights, params, structure, tag, _ = prepare_trial(static_data, study, trial, objective, cluster_dir,
                                                                method, refit, 0)
        run_reporting.child('trial', f'{study.study_name}/{trial.number}', parent=parent,
            identity={'trial': trial.number, 'path_weights': path_weights},
            outcome={'kind': 'nothing_to_do', 'detail': 'parameters prepared; no executor dispatched'})
        run_reporting.child('study', study.study_name,
            outcome={'kind': 'success', 'detail': 'manual trial preparation completed'})
        log.info('%s trial %d prepared (%s): %s', study.study_name, trial.number, tag, params)
        commands.append(f'{sys.executable} {runner} --path-weights {path_weights} --gpu 0')
    print('\nRun (or debug in PyCharm with the same arguments; CUDA_VISIBLE_DEVICES selects the card):')
    for command in commands:
        print('  ' + command)
    print('The next `run_training.py` (without --refit) adopts the trials you finished this way.')
    return len(commands)


def _main(argv=None, report=None):
    args = parse_args(argv)
    # Quiet for this command only: this function is also called in-process.
    with quieted(args.log_level):
        return _run(args, report)


def _run(args, report=None):
    config = args.config if os.path.isabs(args.config) else os.path.join(path_pycharm, args.config)
    if not os.path.exists(config):
        raise SystemExit(f'configuration not found: {config}')
    from eforecast.config.run_inputs import prepare
    static_data = prepare(config, 'training', args)
    from eforecast.config.run_reporting import context as managed_context
    if managed_context() is not None and os.environ.get('EFORECAST_STAGE_PLAN_ID'):
        # The accepted UI action overrides the case's default for this execution.
        # Keep ordinary CLI/legacy configuration precedence outside managed runs.
        action = os.environ.get('EFORECAST_STUDY_ACTION')
        if action in {'continue','new','rebuild'}:
            static_data['study_action'] = action
            if action == 'new' and os.environ.get('EFORECAST_STUDY_REVISION'):
                static_data['study_revision'] = os.environ['EFORECAST_STUDY_REVISION']
    from eforecast.config.stage_catalogue import resolve, describe, require_ready
    from eforecast.config.run_inputs import request_options
    static_data = initializer(static_data, dry_run=True)
    plan = resolve(static_data, 'training', request_options('training', args))
    log.info('%s', describe(plan))
    if plan.get('completion_version') == 1:
        from eforecast.config import stage_resume
        approved = stage_resume.accepted()
        resume = stage_resume.decide(static_data, plan, verify=not args.dry_run,
                                     action=stage_resume.requested_action(static_data, plan['options']))
        stage_resume.reconcile(resume, approved)     # the accepted permissions are not replaced here
        log.info('%s', stage_resume.describe(resume))
        stage_resume.pin(plan, resume)   # what may be skipped, and what an explicit rebuild covers
    if args.dry_run:
        report.update(kind='nothing_to_do', detail='training stage preview')
        return 0
    require_ready(plan)
    # A family whose trials are of another configuration is trained from the beginning, and
    # those trials go first. The run says which families before it starts -- a run the service
    # started was asked about them in the browser, and stops here if it was not -- and the
    # families it named are the ones it may start over: a study that turns out to hold trials
    # of another configuration without its family having been named stops the run instead of
    # being emptied unannounced.
    listed = True                       # a plan that lists nothing of the kind cannot narrow it
    if plan.get('completion_version') == 1:
        stage_resume.announce_changes(resume, static_data)
        listed = stage_resume.started_over(resume)
    args.begin_again = listed
    training = next(node for node in plan['nodes'] if node['key'] == 'training')
    if not training['enabled']:
        report.update(kind='nothing_to_do', detail=training['skip_reason'])
        return 0
    # A stage the plan found whole fits nothing, so it asks for no card. A finished case then
    # passes its training step on a machine that has none -- a cluster node, a laptop --
    # instead of stopping with "GPU ids [0] are unavailable" before it had looked at what was
    # left to do. Everything else about the request is checked as before.
    whole = plan.get('completion_version') == 1 and stage_resume.whole(resume, 'training')
    try:
        validate_training_request(static_data, args, cards=not whole)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    static_data = initializer(static_data)
    return train_stage(plan, static_data, args, report, planned_whole=whole)


def train_stage(plan, static_data, args, report, *, planned_whole=False):
    """The training stage of a run that may start: reused where it is whole, fitted where it is not."""
    from eforecast.config.run_reporting import stage
    with stage(plan, 'training', static_data, set()) as observation:
        if not observation:
            # The budget is satisfied by compatible trials and the payloads verify: the stage
            # reports its reuse by itself, and nothing here may dispatch a fit.
            report.update(kind='success', detail='reused the published training artifacts')
            return 0
        if planned_whole and not args.cpu and not args.prepare_only:
            # The plan found the stage whole and the boundary, hashing it a moment later, did
            # not: it is fitted after all, and now it is asked whether its cards are here.
            try:
                require_cards(static_data, args)
            except ValueError as error:
                raise SystemExit(str(error)) from error
        # Every trial maps the scaled weather inputs instead of building them; a case whose
        # datasets were made before these existed gets them here, once, before its first trial.
        from eforecast.datasets import scaled_inputs
        scaled_inputs.ensure(static_data)
        code = _train(static_data, args, report)
        observation.update(kind=report['kind'], detail=report['detail'])
        return code


def _train(static_data, args, report):
    from eforecast.training.command_line.revisions import chosen_action
    effective_refit = args.refit or chosen_action(static_data) == 'rebuild'
    if args.storage:
        os.environ['EFORECAST_OPTUNA_STORAGE'] = args.storage
    if args.metrics_port is not None:
        os.environ['EFORECAST_METRICS_PORT'] = str(args.metrics_port)
    if static_data.get('transfer_learning') and 'static_data_base' not in static_data:
        static_data['static_data_base'] = joblib.load(static_data['transfer_learning_from']['configuration'])
    from eforecast.common_utils.cpu_scope import methods as cpu_methods
    needs_clusters = not args.cpu or bool(cpu_methods(static_data, getattr(args, 'cpu_scope', None), args.cpu_method)[1])
    if static_data['is_Fuzzy'] and needs_clusters:
        from eforecast.clustering.clustering_manager import ClusterOrganizer
        ClusterOrganizer(static_data, is_online=False, train=True, refit=False).update_cluster_folders()
    if args.cpu:
        from eforecast.training.cpu_dispatch import train_on_cpus
        log.info('project %s: CPU training%s', static_data['_id'], ' [refit]' if args.refit else '')
        from eforecast.config.run_inputs import apply_overrides
        static_data = apply_overrides(static_data, 'training', vars(args))
        outcome = train_on_cpus(static_data, method=args.cpu_method, refit=effective_refit, storage_url=args.storage,
                                scope=getattr(args, 'cpu_scope', None), begin_again=getattr(args, 'begin_again', False))
        if isinstance(outcome, dict):
            report.update(outcome)
        return 1 if isinstance(outcome, dict) and outcome['kind'] == 'failure' else 0
    if args.docker:
        configure_docker(static_data, args.image)
    if args.prepare_only:
        prepared = prepare_only(static_data, args, refit=effective_refit)
        report.update(kind='success' if prepared else 'nothing_to_do', detail='trial parameter preparation only',
                      counts={'prepared_trials': prepared})
        return 0
    from eforecast.training.command_line.train_manager_for_gpu import train_on_gpus
    log.info('project %s: training on gpus %s%s%s', static_data['_id'], args.gpus or f"0..{static_data['n_gpus'] - 1}",
             ' [refit]' if args.refit else '', ' [docker]' if args.docker else '')
    orchestrator = train_on_gpus(static_data, global_method=args.global_method,
                                 cluster_method=args.cluster_method, refit=effective_refit, gpu_id=args.gpus,
                                 n_jobs=args.n_jobs, storage_url=args.storage,
                                 halve_batch_on_oom=args.halve_batch, begin_again=getattr(args, 'begin_again', False))
    # a run that completed nothing, or was stopped, must not report success to whoever started it
    log.info('run outcome: %s (exit %d)', orchestrator.outcome, orchestrator.exit_code)
    report.update(kind={'succeeded': 'success', 'succeeded_with_failures': 'partial', 'failed': 'failure',
                        'cancelled': 'cancelled', 'nothing_to_do': 'nothing_to_do'}[orchestrator.outcome],
                  detail=orchestrator.outcome,
                  counts={'completed': getattr(orchestrator, 'completed', 0),
                          'failures': len(getattr(orchestrator, 'failures', [])),
                          'unschedulable_units': len(getattr(orchestrator, 'unschedulable_units', []))})
    return orchestrator.exit_code


def main(argv=None):
    from eforecast.config.run_reporting import operation
    with operation('training') as report:
        return _main(argv, report)


if __name__ == '__main__':
    sys.exit(main())
