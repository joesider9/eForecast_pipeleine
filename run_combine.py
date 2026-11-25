#!/usr/bin/env python
# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Predict, select, combine and evaluate the trained models of a case (the step after run_training.py).

    python run_combine.py --config <case>                     # all stages
    python run_combine.py --config <case> --stage predict     # cache the evaluation-period predictions of every trial
    python run_combine.py --config <case> --stage select      # keep the best trials per (cluster, method)
    python run_combine.py --config <case> --stage combine     # fit the combiners on the selection slice, evaluate
    python run_combine.py --config <case> --stage evaluate    # report only (fitted combiners)
    python run_combine.py --config <case> --refit             # recompute predictions and refit everything

Protocol: the evaluation period is split by ``selection_fraction`` (methods section, default 0.5)
into a selection slice, which chooses the trials and fits the combiners, and a holdout slice,
which only the final report sees.  Outputs: ``<trial>/predictions_eval.pickle``,
``<cluster_dir>/<method>/selected_trials.json``, ``<cluster_dir>/combine/...``, ``Results/calibration.json``,
``<path_model>/combine_models/...``, ``<path_model>/Results/{selection,evaluation,summary}.csv``, and with
every evaluation what the trials' hyper-parameters say: ``Results/hyper_parameters.csv`` (every finished
trial, its hyper-parameters and its errors) and ``Results/hyper_parameter_report.txt``. An evaluation made
before there was such a report gets it the next time the evaluate stage is asked for.
"""
import logging
import os
from pathlib import Path
import sys

path_pycharm = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_pycharm)

from eforecast.common_utils.logger import quieted
from eforecast.init.initialize import initializer
from eforecast.config.entrypoint_parsers import parse_combine_args
from eforecast.config.pipeline_stages import COMBINE_STAGES, COMBINE_DEFAULT_STAGES
from eforecast.config import model_compute, site_graph_compute  # hosted model adapters

log = logging.getLogger('run_combine')
STAGES = COMBINE_STAGES


def parse_args(argv=None):
    return parse_combine_args(argv, description=__doc__)

def _main(argv=None, report=None):
    args = parse_args(argv)
    # Quiet for this command only: this function is also called in-process.
    with quieted(args.log_level):
        return _run(args, report)


def _run(args, report=None):
    config = args.config if os.path.isabs(args.config) else os.path.join(path_pycharm, args.config)
    from eforecast.config.run_inputs import prepare
    static_data = initializer(prepare(config, 'combine', args), dry_run=True)
    from eforecast.config.stage_catalogue import resolve, describe, require_ready
    from eforecast.config.run_inputs import request_options
    from eforecast.config.run_reporting import stage
    plan = resolve(static_data, 'combine', request_options('combine', args))
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
        report.update(kind='nothing_to_do', detail='combination stage preview')
        return 0
    require_ready(plan)
    if not any(node['selected'] and node['enabled'] for node in plan['nodes']):
        report.update(kind='nothing_to_do', detail='no applicable combination stages')
        return 0
    static_data = initializer(static_data)
    # Predicting every trained model reads the weather once per model: from the scaled inputs,
    # written here if the case does not have them yet.
    from eforecast.datasets import scaled_inputs
    scaled_inputs.ensure(static_data)
    from eforecast.config.inference_compute import run_stage
    completed, executed = set(), set()
    for key in plan['execution_order']:
        with stage(plan, key, static_data, completed) as observation:
            if not observation:
                # A reused stage still owes its product to the stages after it; the worker loads
                # the published prediction caches and combined_eval.pickle in its own process,
                # which keeps those frames out of this process (R09-08-09).
                continue
            private = observation.static_data
            run_stage(key, private, args, observation.workspace.rebase(plan))
            executed.add(key)
    if 'evaluate' not in executed:
        report_of_an_earlier_evaluation(static_data, plan)
    log.info('done; results in %s', os.path.join(static_data['path_model'], 'Results'))
    return 0


def report_of_an_earlier_evaluation(static_data, plan):
    """The hyper-parameter report of an evaluation that was made before there was one.

    The evaluate stage writes ``Results/hyper_parameters.csv`` and its report itself (since
    23 November 2026). A stage that is reused writes nothing, so a case evaluated earlier would
    have none until something made its evaluation stale. When the evaluation is asked for and
    its report is missing, the evaluation is made again by itself -- nothing is predicted,
    selected or combined again, and its own tables come out as they were -- which adds the two
    files. Returns whether it added them; it cannot fail the run.
    """
    from eforecast.combine_predictions.hyper_parameters import TABLE, REPORT
    node = next((node for node in plan['nodes'] if node['key'] == 'evaluate'), None)
    results = Path(static_data['path_model']) / 'Results'
    if (node is None or not node['selected'] or not node['enabled'] or (results / TABLE).is_file()
            or not (results / 'evaluation.csv').is_file()):
        return False
    try:
        from eforecast.config.inference_compute import dispatch
        dispatch('evaluate', static=static_data)
    except Exception as error:  # noqa: BLE001 -- a report; the evaluation it describes is complete
        log.warning('the hyper-parameter report of this evaluation could not be added (%s: %s)',
                    type(error).__name__, error)
        return False
    if not (results / TABLE).is_file():
        return False
    log.info('this evaluation had no hyper-parameter report; it was added: %s', results / REPORT)
    return True


def main(argv=None):
    from eforecast.config.run_reporting import operation
    with operation('combine') as report:
        return _main(argv, report)


if __name__ == '__main__':
    sys.exit(main())
