# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""One parallel layer at a time, inside the owning worker's CPU budget.

Joblib uses threads (no independent loky process tree). RF trees or LassoCV
folds may fan out, with one native thread each; LightGBM/CatBoost instead use
their explicit native budget. Candidate, permutation and method loops stay
serial. Data/helper pools must not run beside an estimator.

The layer is the estimator when one selector is fitted at a time, and the
selectors when several are (:func:`side_by_side`): each then fits with its
part of the budget, :func:`share`, which is one thread whenever there are as
many selectors as CPUs. A case has a selector for every cluster, method and
group of inputs, most of them over a few thousand rows, and an estimator given
sixteen threads for that spends them waiting for each other: seventy-eight
selectors took 71.5 s one after another with sixteen threads each (1,106 s of
CPU) and 15.4 s sixteen at a time with one thread each (174 s of CPU), and kept
the same features.
"""
from contextlib import contextmanager
import threading

from eforecast.bounded_pools import capacity
from eforecast.execution_policy import current

_worker = threading.local()                 # set in a side-by-side worker: its threads, its pool's stop


def jobs(requested):
    policy = current()
    if policy is None:
        # Keep direct/debug callers usable without granting all of the host.
        return 1
    available = min(capacity(), policy['effective']['n_jobs'])
    wanted = int(requested or available)
    return max(1, min(available, wanted if wanted > 0 else available))


def share(granted):
    """The threads one estimator may use here: the threads of this work, or this selector's part of them."""
    return getattr(_worker, 'threads', None) or granted


def shares(tasks, granted):
    """``(selectors fitted at once, threads each)`` for ``tasks`` selectors within ``granted`` CPUs.

    As many at once as there are CPUs or selectors, and the CPUs divided among them: one CPU or
    one selector means one at a time, each estimator with every CPU.

    Memory is not looked at. Selectors fitted at once each hold their own working copy -- about
    nine times their inputs for a CatBoost-SHAP selector, and 20 KB for every training row for a
    forest of 200 trees -- and the operating system carries that or ends the stage. The case's
    ``n_jobs`` is how many are fitted at once.
    """
    at_once = max(1, min(int(granted), int(tasks)))
    return at_once, max(1, int(granted) // at_once)


def check_cancelled():
    stop = getattr(_worker, 'stop', None)
    if stop is not None and stop.is_set():
        # Nobody reads this: the batch this worker belonged to has already ended with its own error.
        raise KeyboardInterrupt('the selector batch ended')
    from eforecast.bounded_pools import cancel_check
    check = cancel_check.get()
    if check is not None:
        check()


@contextmanager
def limits(requested):
    from joblib import parallel_config
    from threadpoolctl import threadpool_limits
    check_cancelled()
    # Re-enter for each call: native libraries may change OpenMP defaults in a
    # previous fit. BLAS cannot multiply the joblib or estimator-native layer.
    with threadpool_limits(limits=1), parallel_config(backend='threading', n_jobs=jobs(requested)):
        yield
    check_cancelled()


def side_by_side(function, tasks, at_once, each):
    """Yield ``function(task)`` for every task, in their order, ``at_once`` of them running.

    The workers are threads: the estimators release the interpreter while they fit, and every
    selector reads the same frames, which processes would each have to be sent. An estimator in a
    worker gets ``each`` threads (:func:`share`). This generator holds the native libraries to
    one thread for as long as its workers live, so a worker entering :func:`limits` finds the
    limit it sets and restores the limit it found, whatever the others are doing.

    A result is yielded when every one before it has been, so the caller can save as it goes
    exactly as it did one selector at a time. A failure is raised where its result was due; a
    cancellation is raised within a fifth of a second. Either way the selectors still waiting
    are dropped and the ones running stop at their next estimator call.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from contextvars import copy_context
    from threadpoolctl import threadpool_limits
    stop = threading.Event()

    def run(task):
        _worker.threads, _worker.stop = each, stop
        try:
            check_cancelled()
            return function(task)
        finally:
            del _worker.threads, _worker.stop

    with threadpool_limits(limits=1):
        executor = ThreadPoolExecutor(at_once, thread_name_prefix='selector')
        futures = []
        try:
            # Each call carries the caller's context (the run's cancellation check) into its thread.
            futures.extend(executor.submit(copy_context().run, run, task) for task in tasks)
            for future in futures:
                while True:
                    check_cancelled()
                    try:
                        value = future.result(timeout=.2)
                        break
                    except TimeoutError:
                        if future.done():
                            value = future.result()         # the selector's own TimeoutError
                            break
                yield value
        finally:
            stop.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
