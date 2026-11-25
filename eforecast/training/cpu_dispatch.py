# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""CPU studies started side by side. No estimators, datasets or Optuna load in this module.

A study is one method on one cluster (or on the whole case). Each runs in a process of its own,
so its memory goes back to the machine when it ends; the trials inside a study run one after
the other. This module decides how many studies run at once and how many threads each one's
estimator is told to use, starts them, and collects what they return.

Nothing is reserved for a study and none waits to be admitted. The CPUs the case uses are
divided among the studies that run at once, and the operating system schedules them like any
other processes.
"""
import copy
import io
import json
import logging
import os
from pathlib import Path
import pickle
import signal
import socket
import uuid
import time

from eforecast.common_utils import cpu_scope, trial_layout
from eforecast.config import run_reporting
from eforecast.config.capabilities import FOUNDATION_METHODS, foundation_device
from eforecast.config.compute_workers import ComputeWorkers, failure_reason
from eforecast.config.model_compute import device_for
from eforecast.training.command_line import resource_reporting as resources

log = logging.getLogger(__name__)


def limits(static, count):
    """How many of ``count`` studies run at once, and with how many threads.

    ``n_cpus`` is how many CPUs the case uses: every CPU this process may use when it names
    none. ``cpu_workers`` is how many studies run at once. A case that names none runs as many
    as it has studies, up to one a CPU -- or up to as many as get the threads it asks for a
    study (``cpu_study_threads``).

    The CPUs are then divided among the studies that run at once. ``threads_per_study`` is the
    even share; what an even division leaves over is given out when the studies start, and
    what a finished study leaves goes to the ones still running (``_dispatch``, ``_share``),
    so the CPUs the case uses stay in use until the last study ends.
    """
    capacity = resources.cpu_slots()
    total = min(capacity, max(1, int(static.get('n_cpus') or capacity)))
    asked = static.get('cpu_study_threads')
    asked = max(1, min(total, int(asked))) if asked else None
    requested = int(static['cpu_workers']) if static.get('cpu_workers') else None
    if requested is not None:
        effective = max(1, min(requested, count, total))
    else:
        effective = max(1, min(count, total // (asked or 1)))
    share = max(1, total // effective)
    threads = min(share, asked or share)
    return {'requested_workers': requested, 'effective_workers': effective,
            'requested_threads': static.get('cpu_study_threads'), 'threads_per_study': threads,
            'requested_total_threads': static.get('n_cpus'), 'total_threads': total,
            'effective_total_threads': total if asked is None else min(total, threads * effective),
            'trials_per_study': 1, 'studies': count}


def describe(policy):
    """One line saying how many studies run at once and why that many.

    ``or more``: a study is given the threads of the ones that finish. Not when the case asks
    for a number of threads a study: then that is all a study ever gets.
    """
    asked = policy['requested_workers']
    more = '' if policy['requested_threads'] else ' or more'
    line = (f"CPU studies: {policy['effective_workers']} at a time, {policy['threads_per_study']} thread(s) each"
            f"{more} ({policy['studies']} to train, {policy['total_threads']} CPUs")
    if asked is not None:
        return line + f'; the case asks for {asked} at a time)'
    return line + ')'


def fair_units(selected):
    """Rotate methods and scopes rather than draining all clusters of one method."""
    groups = {}
    for unit in selected:
        groups.setdefault(unit['method'], []).append(unit)
    result = []
    while groups:
        for method in list(groups):
            result.append(groups[method].pop(0))
            if not groups[method]:
                del groups[method]
    return result


def _wanted_device(static, method):
    """The device a study asks for: a foundation family's preference, or the CPU."""
    if method not in FOUNDATION_METHODS:
        return 'cpu'
    return foundation_device(static, method)


def _device_of(static, unit):
    """The device a study is started for: its family's preference, or the CPUs once a card has failed it."""
    return 'cpu' if unit.get('off_card') else _wanted_device(static, unit['method'])


def _share(workers, job, threads):
    """Tell a study how many threads it has now.

    The estimator of its next trial is built with them; a foundation model computes its next
    forecast with them.
    """
    from eforecast.compute_worker import atomic_bytes
    from .cpu_worker import SHARE_FILE
    job['threads'] = threads
    atomic_bytes(workers.root(job['handle']) / SHARE_FILE, str(threads).encode('ascii'))


def _spread(workers, jobs, policy):
    """Give the CPUs nobody is using to the studies that are running, fewest threads first.

    A study keeps the threads it started with for the trial it is fitting, and builds the
    estimator of its next trial with what it has been given by then. Never more than the case
    asks for a study, when it asks; a study on a card keeps what it has.
    """
    most = policy['threads_per_study'] if policy['requested_threads'] else policy['total_threads']
    free = policy['total_threads'] - sum(job['threads'] for job in jobs)
    shares = {id(job): job['threads'] for job in jobs if not job['device']}
    while free > 0 and any(share < most for share in shares.values()):
        least = min((key for key, share in shares.items() if share < most), key=shares.get)
        shares[least] += 1
        free -= 1
    for job in jobs:
        if shares.get(id(job), job['threads']) != job['threads']:
            _share(workers, job, shares[id(job)])


def _start(static_data, unit, run_id, workers, threads, most, device, aggregate, storage_url, refit,
           begin_again=False):
    """Prepare and start one study with ``threads`` now, at most ``most`` later, and a card or none."""
    directory = str(Path(unit['directory']).absolute())
    static = trial_layout.pin(static_data, directory, unit['method'], unit['cluster'])
    bounded = copy.copy(static)
    bounded.update(n_cpus=most, n_jobs=1)
    ident = uuid.uuid4().hex
    wanted = _device_of(static, unit)
    info = {'cpu_threads': threads, 'requested_device': wanted, 'device': 'cuda' if device else 'cpu'}
    if unit['method'] not in FOUNDATION_METHODS:
        info['normalization'] = 'Shallow models use CPU; a foundation device preference does not apply.'
    elif unit.get('off_card'):
        info['normalization'] = 'It did not finish on a card; the study runs on the CPU.'
    elif device is None and str(wanted).lower() not in {'cpu', 'none'}:
        info['normalization'] = 'No GPU matches the preference; the study runs on the CPU.'
    payload = {'cluster': unit['cluster'], 'directory': directory, 'method': unit['method'],
               'storage_url': storage_url or os.environ.get('EFORECAST_OPTUNA_STORAGE') or None, 'refit': bool(refit),
               # the answer the run was given before it started: a study of another configuration
               # of its family may be emptied and trained from the beginning (``train_cpu``)
               'begin_again': bool(begin_again),
               'estimate': info, 'aggregate_status': aggregate}
    # The worker is started with the most threads it may ever be given; what it uses at a
    # time is its share, which is written beside it before it starts and raised as others end.
    request = workers.request(run_id, os.path.realpath(os.path.join(directory, unit['method'])), ident,
                              most, device=device)
    key = f"{run_id}/{directory}/{unit['method']}"
    run_reporting.child('study', key, identity={**unit, 'cluster_dir': directory})
    execution = ({'device': wanted if device else 'cpu', 'threads': most} if unit['method'] in FOUNDATION_METHODS else
                 {'device': 'cpu', 'threads': 1, 'n_jobs': most, 'parallel_layer': 'estimator_or_cv'})
    handle = workers.prepare(request, 'eforecast.training.cpu_worker:execute', payload,
        input_files={'study.pickle': pickle.dumps(bounded, protocol=5)}, execution=execution)
    job = {'unit': unit, 'key': key, 'handle': handle, 'threads': threads, 'device': device,
           'info': info, 'started': time.monotonic()}
    _share(workers, job, threads)
    workers.submit(handle)
    return job


def _status(static, run_id, workers, policy, jobs, phase, *, waiting=(), completed=None, started=None):
    """What is running and what is done, for the run's resource file and the model folder."""
    def study(job):
        return {**job['unit'], 'state': 'cleanup' if phase == 'stopped' else 'active',
                'estimate': job['info']}
    counts = {key: sum(value['outcome'].get('counts', {}).get(key, 0) for value in (completed or {}).values())
              for key in ('complete', 'pruned', 'fail')}
    elapsed = max(0., time.monotonic() - started) if started is not None else 0.
    pids = [pid for pid in (workers.pid(job['handle']) for job in jobs) if pid]
    status = {'run_id': run_id, 'phase': phase, 'dispatch': policy,
              'progress': {'verified_complete': counts['complete'], 'pruned': counts['pruned'], 'failed': counts['fail'],
                           'elapsed_seconds': elapsed, 'verified_models_per_minute': counts['complete'] * 60 / elapsed if elapsed else 0.},
              'cpu': resources.cpu_inventory(), 'gpus': {},
              'running': {'studies': len(jobs), 'cpu_threads': sum(job['threads'] for job in jobs)},
              'observed': resources.process_tree(pids),
              'studies': [study(job) for job in jobs] + [
                  {**unit, 'state': 'waiting',
                   'reason': 'its turn has not come: the studies that run at once are running'} for unit in waiting]}
    resources.write_status(status, str(Path(static['path_model']) / 'cpu_resource_status.json'))
    resources.write_status(status)


def _dispatch(static, selected, run_id, workers, storage_url, refit, *, aggregate=True, begin_again=False):
    """Run the studies; ``{unit id: what its worker returned}``.

    A study whose process fails does not stop the others: it is recorded as failed, with the
    reason, and the rest are trained. Two exceptions, because they are the failures this way
    of running invites. A study the kernel killed -- which it does to the largest process when
    memory runs out -- is started once more when every other study has finished, with the
    machine to itself. And a foundation study that failed on a card -- nothing holds a card for
    it, so its memory may be taken -- is started once more on the CPUs. A study continues from
    its journal, so the trials it had finished are not fitted again.
    """
    policy = limits(static, len(selected))
    if aggregate:
        log.info('%s', describe(policy))
    def answer(unit):
        """What this study is told. Passed on only when it was given: a study is started as it
        always was otherwise -- and given for the families the accepted plan named, no others."""
        from eforecast.training.command_line.optuna_search import may_begin_again
        return {'begin_again': True} if may_begin_again(begin_again, unit['method']) else {}
    pending = fair_units(selected)
    jobs, completed, alone = [], {}, []
    last_status = 0.
    started = time.monotonic()
    try:
        while jobs or pending or alone:
            run_reporting.check_cancelled()
            if not jobs and not pending:
                # Everything else is done: a study that was killed gets the machine.
                unit = alone.pop(0)
                jobs.append(_start(static, unit, run_id, workers, policy['total_threads'], policy['total_threads'],
                                   device_for(_device_of(static, unit)), aggregate, storage_url, refit,
                                   **answer(unit)))
            for_a_card = False
            while pending and len(jobs) < policy['effective_workers']:
                # A card runs one foundation study of this run at a time: nothing but the
                # card itself would tell two of them that they do not fit together.
                busy = {job['device'] for job in jobs if job['device']}
                chosen = None
                for index, unit in enumerate(pending):
                    wanted = _device_of(static, unit)
                    device = device_for(wanted, exclude=busy)
                    if device is None and device_for(wanted) is not None:
                        continue            # its card is busy with a sibling; another study goes first
                    chosen = index, device
                    break
                if chosen is None:
                    for_a_card = True       # every study still to start is waiting for a card
                    break
                index, device = chosen
                unit = pending.pop(index)
                # What is free is divided among the studies still to start now, so that an
                # uneven division is given out rather than left idle, and a study that starts
                # when few are left gets the threads the finished ones gave back.
                free = policy['total_threads'] - sum(job['threads'] for job in jobs)
                starting = min(policy['effective_workers'] - len(jobs), len(pending) + 1)
                most = policy['threads_per_study'] if policy['requested_threads'] else policy['total_threads']
                threads = max(1, min(most, free // starting))
                jobs.append(_start(static, unit, run_id, workers, threads, most, device, aggregate, storage_url, refit,
                                   **answer(unit)))
            # The CPUs nobody is using go to the studies that are running: when every study has
            # started, and also while the ones still to start can only wait for a card. A case
            # with a foundation model for each of seven clusters has twenty-one of them queued
            # for two cards, and its tree studies ran with one thread each beside seven idle
            # CPUs until the last of them had started.
            if (not pending or for_a_card) and not alone:
                _spread(workers, jobs, policy)
            for job in list(jobs):
                run_reporting.check_cancelled()
                result = workers.collect(job['handle'])
                if result['state'] == 'cancelled':
                    raise KeyboardInterrupt(f"CPU study cancelled; execution {job['handle'].id}")
                if result['state'] in {'failure', 'unknown'}:
                    # The reason, not only where to look for it: a worker's traceback is in
                    # its own log, two directories named after hashes away.
                    reason = (f"CPU study {job['unit']['id']} did not finish: {failure_reason(result)}"
                              f"; execution {job['handle'].id}")
                    if job['device'] and not job['unit'].get('off_card'):
                        # A card can be out of memory, or lost to its driver, while the CPUs are
                        # fine. What a study did not finish on a card it is given the CPUs for,
                        # once; it continues from its journal like any study started again.
                        log.warning('%s; it ran on a card and is started again on the CPUs', reason)
                        jobs.remove(job)
                        pending.append({**job['unit'], 'off_card': True})
                        continue
                    if not aggregate:
                        raise resources.ResourceUnavailable(reason)
                    jobs.remove(job)
                    unit = job['unit']
                    if result.get('returncode') == -signal.SIGKILL and not unit.get('again') and (jobs or pending):
                        log.warning('%s; it is started again, alone, when the other studies have finished', reason)
                        alone.append({**unit, 'again': True})
                        continue
                    log.error('%s', reason)
                    outcome = {'kind': 'failure', 'detail': reason, 'study_failed': True,
                               'counts': {'complete': 0, 'pruned': 0, 'fail': 0}}
                    completed[unit['id']] = {'outcome': outcome}
                    run_reporting.child('study', job['key'], outcome={'kind': 'failure', 'detail': reason})
                    continue
                if result['state'] != 'success':
                    continue
                result = workers.wait(job['handle'], timeout=None)
                value = result['value']
                completed[job['unit']['id']] = {'outcome': value['outcome']} if aggregate else value
                run_reporting.child('study', job['key'], outcome=value['outcome'])
                if value['outcome']['kind'] in {'success', 'nothing_to_do'}:
                    try:
                        workers.discard_result(job['handle'])
                    except OSError as error:
                        log.warning('the folder of a finished CPU study was not removed: %s', error)
                jobs.remove(job)
            if aggregate and time.monotonic() - last_status >= 1.:
                _status(static, run_id, workers, policy, jobs, 'running', waiting=[*pending, *alone],
                        completed=completed, started=started)
                last_status = time.monotonic()
            if jobs:
                time.sleep(.1)
        return completed
    except BaseException as error:
        for job in jobs:
            run_reporting.child('study', job['key'], outcome={
                'kind': 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failure', 'detail': str(error)})
            workers.cancel(job['handle'], force=True)
        raise
    finally:
        if aggregate:
            try:
                _status(static, run_id, workers, policy, jobs, 'stopped' if jobs else 'finished',
                        completed=completed, started=started)
            except Exception:
                log.exception('the CPU studies\' status file could not be written')


def train_study(static_data, cluster_name, cluster_dir, method, storage_url=None, refit=False, run_id=None,
                *, workers=None, begin_again=False):
    run_id = run_id or f'cpu-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    workers = workers or ComputeWorkers()
    unit = {'kind': 'study', 'scope': 'global' if cluster_name == 'global' else 'cluster',
            'cluster': cluster_name, 'directory': str(Path(cluster_dir).absolute()), 'method': method,
            'id': f'{cluster_name}:{method}:study'}
    value = _dispatch(static_data, [unit], run_id, workers, storage_url, refit, aggregate=False,
                      begin_again=begin_again)[unit['id']]
    import pandas as pd
    table = pd.read_json(io.StringIO(json.dumps(value['table'])), orient='split')
    table.attrs['run_outcome'] = value['outcome']
    return table


def train_on_cpus(static_data, cluster=None, method=None, refit=False, storage_url=None, run_id=None, scope=None,
                  *, workers=None, begin_again=False):
    run_id = run_id or f'{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}'
    selected = cpu_scope.units(static_data, scope, method, cluster)
    if not selected:
        return {'kind': 'nothing_to_do', 'detail': 'no enabled CPU units in the requested scope', 'counts': {}, 'units': []}
    workers = workers or ComputeWorkers()
    values = _dispatch(static_data, selected, run_id, workers, storage_url, refit, begin_again=begin_again)
    totals = {key: sum(values[u['id']]['outcome'].get('counts', {}).get(key, 0) for u in selected)
              for key in ('complete', 'pruned', 'fail')}
    # A study whose process failed has no trials to count, and is a failure all the same.
    totals['studies_failed'] = sum(bool(values[u['id']]['outcome'].get('study_failed')) for u in selected)
    done = totals['complete'] + totals['pruned']
    failed = totals['fail'] + totals['studies_failed']
    kind = ('partial' if failed else 'success') if done else 'failure' if failed else 'nothing_to_do'
    return {'kind': kind, 'counts': totals,
            'units': [{**u, 'outcome': values[u['id']]['outcome'], 'global_reuse': False} for u in selected]}
