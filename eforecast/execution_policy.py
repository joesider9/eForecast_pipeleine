# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""How a piece of work runs: how many threads its libraries use and which card it sees.

Whoever starts the work resolves this document before the process exists, and the process
installs it before it imports anything. It says what the work was started with -- some threads,
perhaps one card -- and nothing is reserved to back it: the operating system schedules the
threads and the driver hands out the card's memory. Importing this module imports no ML,
training or GPU packages. A device failure ends the attempt.
"""
from contextlib import contextmanager
import importlib.util
import json
import logging
import os
import sys

ENV = 'EFORECAST_EXECUTION_POLICY'
#: A file holding the threads this work should use at present; see ``estimator_threads`` and
#: ``network_threads``.
SHARE = 'EFORECAST_THREAD_SHARE'
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'LOKY_MAX_CPU_COUNT', 'POLARS_MAX_THREADS',
              'RAYON_NUM_THREADS')
log = logging.getLogger(__name__)


def resolve(threads, device_uuid=None, requested=None):
    """Normalize stored preferences to ``threads`` and the card named, if one is.

    ``threads`` is how many the work was started with; ``device_uuid`` the one card it may
    see -- its UUID, or, for a card the driver's library cannot name, the entry of the
    visible-device mask that selects it. The case that holds the preferences is never edited.
    """
    original = dict(requested or {})
    # Settings of the throughput profiles this pipeline once had; a stored preference that
    # still carries them is read without them.
    for dropped in ('profile_plan', 'profile_selection', 'resource_plan'):
        original.pop(dropped, None)
    requested = original
    granted = int(threads)
    if granted < 1:
        raise ValueError('execution needs at least one CPU thread')
    notes = []
    choice = requested.get('device', 'auto')
    spelling = str(choice).lower() if choice is not None else 'cpu'
    cpu = spelling in ('cpu', 'none')
    device = 'cuda:0' if device_uuid and not cpu else 'cpu'
    if not cpu and spelling != 'auto':
        notes.append(f'device {choice!r} resolves to {device}')
    if device == 'cpu' and not cpu:
        notes.append('no GPU was given to this work; auto/CUDA preference resolves to CPU before loading')
    count = requested.get('threads', granted)
    if count in (None, -1, 0):
        count = granted
    threads = max(1, min(granted, int(count)))
    if requested.get('threads', granted) != threads:
        notes.append(f'threads normalized to {threads} of {granted}')
    selector_layer = requested.get('parallel_layer') == 'estimator_or_cv'
    estimator_jobs = 1
    if selector_layer:
        count = requested.get('n_jobs', granted)
        estimator_jobs = max(1, min(granted, int(count if count not in (None, -1, 0) else granted)))
        if threads != 1:
            notes.append('estimator/CV parallel layer requires one ambient native thread')
        threads = 1
    loaders = 0
    # Selectors explicitly choose one active estimator/CV call with serial outer
    # loops. All other adapters retain the original native-only policy. Omitting
    # the opt-in retains the exact document shape of existing sealed plans.
    for name, effective in (('n_jobs', estimator_jobs), ('loader_workers', loaders), ('fallback', 'none')):
        if name in requested and requested[name] != effective:
            notes.append(f'{name} normalized to {effective!r}')
    return {'version': 1, 'requested': original,
            'given': {'cpu_threads': granted, 'device_uuid': device_uuid},
            'effective': {'device': device, 'device_uuid': device_uuid if device != 'cpu' else None,
                          'threads': threads, 'torch_interop_threads': 1, 'n_jobs': estimator_jobs,
                          'loader_workers': loaders, 'fallback': 'none',
                          **({'parallel_layer': 'estimator_or_cv'} if selector_layer else {})}, 'normalization': notes}


def validate(policy, threads, device_uuid):
    if policy != resolve(threads, device_uuid, policy.get('requested')):
        raise ValueError('execution policy disagrees with what the work was started with')
    return policy


def current():
    value = os.environ.get(ENV)
    return json.loads(value) if value else None


def environment(policy):
    effective = policy['effective']
    return {**dict.fromkeys(THREAD_ENV, str(effective['threads'])),
            'CUDA_VISIBLE_DEVICES': effective['device_uuid'] or '',
            'EFORECAST_TRIAL_THREADS': str(effective['threads']),
            'EFORECAST_LOADER_WORKERS': str(effective['loader_workers']), 'EFORECAST_ESTIMATOR_THREADS': str(effective['n_jobs']),
            'OMP_MAX_ACTIVE_LEVELS': '1', 'OMP_NESTED': 'FALSE', 'MKL_DYNAMIC': 'FALSE',
            'OPENCV_FOR_THREADS_NUM': '1',
            'TOKENIZERS_PARALLELISM': 'false', 'JAX_PLATFORMS': 'cpu',
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', ENV: json.dumps(policy, sort_keys=True)}


def apply(policy):
    """Install before imports. An already initialized CUDA runtime cannot be remasked."""
    torch = sys.modules.get('torch')
    changes = environment(policy)
    if (torch is not None and torch.cuda.is_initialized()
            and (policy['effective']['device'] == 'cpu'
                 or changes['CUDA_VISIBLE_DEVICES'] != os.environ.get('CUDA_VISIBLE_DEVICES'))):
        raise RuntimeError('device isolation requires a fresh worker process')
    os.environ.update(changes)


def configure_torch(policy):
    """Set runtime-only limits before the target's imports/constructors, if installed."""
    if importlib.util.find_spec('torch') is None:
        return
    import torch
    torch.set_num_threads(policy['effective']['threads'])
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


@contextmanager
def activated(policy):
    """Compatibility for in-process CLI/debug callers; production workers are isolated."""
    old = {key: os.environ.get(key) for key in environment(policy)}
    apply(policy)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def torch_device(requested='auto'):
    """Honor CPU spelling; map any ordinal to the one card this work was started with.

    A loader running without a policy normalizes to CPU, with a visible warning: a card being
    visible does not make it this work's card.
    """
    import torch
    policy = current()
    explicit_cpu = requested is None or str(requested).lower() in ('cpu', 'none')
    if explicit_cpu or not policy or policy['effective']['device'] == 'cpu':
        if not policy and not explicit_cpu:
            log.warning('device %r normalized to CPU: this process has no execution policy', requested)
        return torch.device('cpu')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('the CUDA device of this work is unavailable; no automatic fallback')
    expected = str(policy['effective']['device_uuid'])
    if expected.startswith(('GPU-', 'MIG-')):       # a card given by its place in the mask cannot be checked
        actual = str(torch.cuda.get_device_properties(0).uuid)
        def canonical(value):
            return value.removeprefix('GPU-').removeprefix('MIG-').lower()
        if canonical(actual) != canonical(expected):
            raise RuntimeError('visible CUDA UUID differs from the device of this work; refusing model load')
    return torch.device('cuda:0')


def _present(most):
    """``most``, or the share of it that the file named by ``EFORECAST_THREAD_SHARE`` holds now.

    A file that is absent or unreadable leaves ``most``.
    """
    path = os.environ.get(SHARE)
    if path:
        try:
            with open(path, encoding='ascii') as stream:
                return max(1, min(most, int(stream.read().strip())))
        except (OSError, ValueError):
            pass
    return most


def estimator_threads(requested=1):
    """The threads an estimator is built with now.

    The policy's ``n_jobs`` is the most this work may use. Whoever started the work can also
    keep a file, named by ``EFORECAST_THREAD_SHARE``, with the share it should use at present:
    a study that runs beside others is given their threads as they finish, and reads the
    number each time it builds an estimator -- once a trial. A file that is absent or
    unreadable leaves the policy's number.
    """
    policy = current()
    if not policy:
        return max(1, int(requested or 1))
    return _present(policy['effective']['n_jobs'])


def network_threads():
    """The threads a network of this work computes with on the CPU now; None outside a policy.

    What ``estimator_threads`` is to an estimator, for work whose threads are torch's own: the
    policy's ``threads`` is the most, and the share file says how many of them to use at
    present. Torch is told the most once, when the worker starts (``configure_torch``). Work
    that runs beside other work has to tell it its share before it computes, or every one of
    them computes with all the CPUs of the case.
    """
    policy = current()
    return _present(policy['effective']['threads']) if policy else None


def bound_estimator(model, parameters):
    """Bound stored estimator overrides as well as constructor defaults, in memory only."""
    if current() is None or model is None:
        return
    for estimator in model.values() if isinstance(model, dict) else (model,):
        supported = estimator.get_params()
        count = estimator_threads()
        changes = {key: count for key in ('n_jobs', 'thread_count', 'nthread', 'num_threads') if key in supported}
        for key in changes:
            if key in parameters:
                parameters[key] = count
        if changes:
            estimator.set_params(**changes)
