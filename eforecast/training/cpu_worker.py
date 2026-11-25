# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A whole CPU study in a process of its own: loading, the search, fitting and publication."""
import json
import os
from pathlib import Path
import pickle

from eforecast.compute_worker import current_worker
from eforecast.config import run_reporting
from eforecast.training.command_line import resource_reporting as resources

#: In the worker's own folder: the threads the study has at present (``cpu_dispatch``). The
#: estimator of its next trial is built with them; a foundation model computes with them.
SHARE_FILE = 'threads'


def execute(payload, context):
    if current_worker() != context:
        raise RuntimeError('a CPU study runs in its own worker process')
    with (Path(context.result_directory) / 'inputs/study.pickle').open('rb') as stream:
        static = pickle.load(stream)
    # The dispatcher keeps this study's present share of the CPUs in this file and raises it as
    # the studies beside it finish; each trial's estimator is built with what it says then, and
    # a foundation model tells torch the number before each forecast.
    from eforecast import execution_policy
    os.environ[execution_policy.SHARE] = str(Path(context.result_directory) / SHARE_FILE)
    key = f"{context.run_id}/{payload['directory']}/{payload['method']}"
    process = run_reporting.process()
    if process is not None:
        process.update(launch_id=context.launch_id)
    run_reporting.child('study', key, execution=process)
    from eforecast.training.train_cpu import _train_study

    def report(phase='running', error=None):
        status = {'run_id': context.run_id, 'phase': phase, 'error': error,
                  'studies': [{'id': key, 'method': payload['method'], 'cluster': payload['cluster'],
                               'state': phase, 'estimate': payload['estimate']}],
                  'cpu': resources.cpu_inventory(),
                  'running': {'studies': 1, 'cpu_threads': max(execution_policy.estimator_threads(),
                                                               execution_policy.network_threads() or 1)},
                  'observed': resources.process_tree([os.getpid()]), 'gpus': {},
                  'policy': json.loads(os.environ['EFORECAST_EXECUTION_POLICY'])}
        resources.write_status(status, str(Path(context.result_directory) / 'resource_status.json'))
        if not payload.get('aggregate_status'):
            resources.write_status(status, str(Path(static['path_model']) / 'cpu_resource_status.json'))
            resources.write_status(status)

    report()
    # One run trains a case at a time and a run's studies stop with it, so whatever an
    # earlier study left RUNNING in this journal has nobody behind it any more.
    table = _train_study(static, payload['cluster'], payload['directory'], payload['method'],
        payload.get('storage_url'), payload['refit'], run_id=context.run_id,
        threads=context.threads, exclusive=True, begin_again=bool(payload.get('begin_again')))
    report('study_finished')
    run_reporting.child('study', key, outcome=table.attrs['run_outcome'])
    return {'table': json.loads(table.to_json(orient='split', double_precision=15)),
            'outcome': table.attrs['run_outcome'], 'pid': os.getpid(), 'estimate': payload['estimate']}
