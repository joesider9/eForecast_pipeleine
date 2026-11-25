# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The CPUs a dataset stage may use, now that it runs in the process that was started.

Reader pools (``bounded_pools.map_bounded``) and the selectors' estimators size themselves from
an *execution policy*: a document that says how many CPU threads the work was granted. Without
one they run inline on a single CPU, which is the right answer for an unhosted caller and for
a debugger.

A dataset stage used to get its policy from the worker process it was sent to. Since stages
write in place there is no such worker, so every stage ran without a policy: extraction, the
weather dataset and feature selection each used one CPU of the machine, and a three-year case
that had taken minutes took the better part of two hours. ``budget`` is that policy again, for
the stages that have something to do with more than one CPU.
"""
from contextlib import contextmanager
import logging
import os

from eforecast import execution_policy
from eforecast.bounded_pools import capacity

log = logging.getLogger(__name__)


def has_readers(static, stage):
    transformations = [item.get('transformer') for item in
                       [*static.get('variables', []), static.get('target_variable', {})]]
    return (stage in {'extract', 'nwp', 'images'} or bool(static.get('remove_night_hours'))
            or any('clear_sky' in str(value) for value in transformations))


def threads(static, stage, *, serial=False):
    """The CPU threads of one dataset stage on this machine.

    One unless the stage has readers (extract, nwp, images, or a night-hour or clear-sky
    transformation) or is a feature selection with a method configured. Otherwise the case's
    ``n_jobs`` within the CPUs this process may use.

    Memory is not looked at. A reader holds an interpreter and the file it is decoding -- about
    0.8 GB and thirty-two times a raw weather file or a day of extracted weather -- and the
    operating system carries as many readers as the case asks for or ends the stage; the case's
    ``n_jobs`` is the setting to lower then.
    """
    available = capacity()
    wanted = int(static.get('n_jobs') or available)
    count = 1 if serial else max(1, min(available, wanted if wanted > 0 else available))
    if stage != 'features' and not has_readers(static, stage):
        count = 1
    if stage == 'features' and not any(m is not None for m in static.get('feature_selection_methods', [])):
        count = 1
    return count


@contextmanager
def budget(static, stage, *, serial=False):
    """Run a stage of this process with the CPUs ``threads`` gives it; yields the thread count.

    Only what the pools and estimators read is installed: the policy document and the thread
    limits its children start with. ``CUDA_VISIBLE_DEVICES`` is left alone -- a clustering or
    image stage resolves its card from this environment before it launches a worker, and each
    worker installs a policy of its own.
    """
    count = threads(static, stage, serial=serial)
    if count == 1:
        yield 1
        return
    requested = {'device': 'cpu', 'threads': 1, 'n_jobs': 1}
    if stage == 'features':
        requested.update(parallel_layer='estimator_or_cv', n_jobs=count)
    policy = execution_policy.resolve(count, None, requested)
    changes = {key: value for key, value in execution_policy.environment(policy).items()
               if key != 'CUDA_VISIBLE_DEVICES'}
    before = {key: os.environ.get(key) for key in changes}
    os.environ.update(changes)
    log.info('stage %s: %d CPUs (%s)', stage, count,
             'estimators' if stage == 'features' else f'{count - 1} readers beside this process')
    try:
        yield count
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
