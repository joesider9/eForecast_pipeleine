# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Run one model request inside the image that can serve it -- A16-02.

The host never imports a model package.  It stages a task directory, runs the pinned image
over it and reads back `.npy` buffers, so pipe_env keeps pandas 3.0.3 while the foundation
image keeps 2.3.3 and the Lag-Llama image keeps 2.1.4.

What a model container is given, and nothing more:

* its own task directory, read-write -- the only place it may write;
* one framework cache, keyed by the image's immutable id, so two runtimes never share a
  compiled kernel or a half-downloaded checkpoint;
* the credentials that method actually needs, as files on a read-only mount.

Credentials are deliberately not passed with `-e`: `docker run -e KEY=value` puts the value
in `docker inspect` output for anyone who can reach the daemon.  The adapter reads them from
the mount into its own environment instead, so the value exists in the container's memory
and nowhere else.  No source tree, no Docker socket and no host model tree is mounted.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from eforecast import image_runtime, runtime_contract as contract
from eforecast.runtime_adapters import adapter_name, transport

_serving_cache = ContextVar('eforecast_serving_runtime_cache', default=None)

@contextmanager
def serving_cache(path):
    token = _serving_cache.set(path)
    try:
        yield
    finally:
        _serving_cache.reset(token)


def runtime_cache_path():
    return Path(_serving_cache.get() or os.environ.get(
        'EFORECAST_RUNTIME_CACHE', Path.home() / '.cache' / 'eforecast' / 'runtimes'))

#: A case names its zero-shot members in its own vocabulary; the model images use the
#: package's. Chronos2 is the case's name for what foundationforecast calls Chronos.
#: Read by readiness and doctor, so a case's method resolves to the image that serves it.
CASE_METHODS = {'Chronos2': 'Chronos', 'TimesFM': 'TimesFM', 'TiRex': 'TiRex',
                'Timestral': 'Timestral', 'T0': 'T0', 'TimeGPT': 'TimeGPT',
                'Lag-Llama': 'Lag-Llama'}

# the pinned image per adapter; a method resolves through eforecast.runtime_adapters
IMAGES = {
    'foundation': 'eforecast-foundation:a16-02',
    'lag_llama': 'eforecast-lagllama:a16-02',
}

ENTRYPOINT = 'eforecast.runtime_adapters:run'
#: what image_runtime.record() asks the image to describe
ENTRYPOINT_WITNESS = 'eforecast.runtime_adapters.__main__:run'
TASK_MOUNT = '/opt/task'
#: downloaded checkpoints: shared, because Hugging Face's cache is content-addressed and the
#: same checkpoint is the same bytes whatever image reads it
CACHE_MOUNT = '/opt/cache/huggingface'
#: compiled kernels and framework scratch: per image id, because these are NOT portable
#: between two different builds and a stale one is a wrong answer, not a slow one
RUNTIME_MOUNT = '/opt/cache/runtime'
SECRET_MOUNT = '/opt/secrets'
TIMEOUT = 1800


class LaunchError(RuntimeError):
    """The image could not serve the request, with what a colleague should do about it."""


def image_for(method):
    """The pinned image reference that serves a method."""
    name = adapter_name(method)
    reference = IMAGES.get(name)
    if reference is None:
        raise LaunchError(f'{method}: no image is pinned for the {name} adapter')
    return reference


def cache_root(reference, *, root=None):
    """(checkpoint cache, per-image runtime cache) for this image.

    The two are deliberately separate.  Compiled kernels are not portable between builds, so
    they are keyed by the image's immutable id -- a retagged or rebuilt image must never
    inherit another runtime's compiled artifacts, because a stale kernel is a wrong answer
    rather than a slow one.  Downloaded checkpoints are the opposite: Hugging Face's cache is
    content-addressed, so the same checkpoint is the same bytes in every image, and keying it
    by image id too would re-download hundreds of megabytes on every rebuild.

    A11's reader pins and retention rules apply to dataset caches; neither of these is one.
    """
    identity = image_runtime.image_id(reference)
    if identity is None:
        raise LaunchError(f'{reference} is not present on this host; build it first '
                          f'(docker build -f docker/Dockerfile.foundation ...)')
    base = Path(root) if root else runtime_cache_path()
    checkpoints = base / 'checkpoints'
    runtime = base / identity.replace(':', '_')
    for directory in (checkpoints, runtime):
        directory.mkdir(parents=True, exist_ok=True)
    return checkpoints, runtime


def required_credentials(method):
    """(environment variable, secret name) for every credential this method needs."""
    from eforecast.runtime_adapters.foundation import FAMILIES
    family = FAMILIES.get(method) or {}
    if family.get('credential'):
        return [(family['environment'], family['credential'])]
    return []


def resolve_checkpoint(method, model, revision=None):
    """Normalize legacy model names to a real commit without rewriting the case.

    Reuse the selected ref in the mounted cache first. Otherwise ask only for
    repository metadata (with a timeout); model weights are loaded by the worker.
    Hosted TimeGPT has no inspectable weights and makes no checkpoint claim.
    """
    import re
    if method == 'TimeGPT':
        return None
    if not isinstance(model, str) or not re.fullmatch(r'[\w.-]+/[\w.-]+', model):
        raise LaunchError('isolated checkpoints require a Hub repository id')
    if isinstance(revision, str) and re.fullmatch('[0-9a-f]{40}', revision) and revision != '0' * 40:
        return {'id': model, 'revision': revision}
    requested = revision if revision and revision != '0' * 40 else 'main'
    if not isinstance(requested, str) or '..' in Path(requested).parts or Path(requested).is_absolute():
        raise LaunchError('checkpoint revision is not a valid reference')
    base = runtime_cache_path()
    cached = base / 'checkpoints' / 'hub' / ('models--' + model.replace('/', '--'))
    try:
        commit = (cached / 'refs' / requested).read_text().strip()
        if re.fullmatch('[0-9a-f]{40}', commit) and (cached / 'snapshots' / commit).is_dir():
            return {'id': model, 'revision': commit}
    except OSError:
        pass
    if os.environ.get('HF_HUB_OFFLINE', '').lower() in ('1', 'true', 'yes'):
        raise LaunchError(f'{model}: no cached commit for {requested}; pre-stage the checkpoint')
    try:
        from huggingface_hub import HfApi
        from eforecast.config.secrets_source import resolve, SecretMissing
        try:
            token = resolve('hf_token', environment='HF_TOKEN')
        except SecretMissing:
            token = None
        commit = HfApi(token=token.reveal() if token else None).model_info(
            model, revision=requested, timeout=10).sha
    except Exception as error:
        raise LaunchError(f'{model}: cannot resolve checkpoint {requested} ({type(error).__name__}); '
                          'pre-stage the checkpoint or check Hub access') from error
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit) or commit == '0' * 40:
        raise LaunchError(f'{model}: Hub metadata did not return an immutable commit')
    return {'id': model, 'revision': commit}


def licence_note(method):
    """The weight licence a colleague is accepting by running this method, when restricted."""
    from eforecast.runtime_adapters.foundation import FAMILIES
    return (FAMILIES.get(method) or {}).get('licence')


def prepare(task_dir, *, method, history, history_index, forecast_index, targets,
            target_units, timezone, frequency_ns, identity, runtime, component,
            device='cpu', quantiles=(), future=None, future_covariates=(),
            past_covariates=(), target_scale='physical', output_scale=None, missing='forbid',
            dataset_id='dataset', partition_id='partition', view_id='view', scaler_id=None,
            budget_bytes=None, reuse=True, manifest=True):
    """Stage the arrays and return the validated request they belong to.

    ``missing='mask'`` is for the real pipeline's inputs: its item builder fills the history
    of a future-only covariate with NaN, and nothing non-finite may cross the boundary. The
    mask is derived here rather than asked of the caller, because a mask that disagrees with
    its payload is a silent wrong answer, and deriving it is the only way the two cannot
    disagree.
    """
    task_dir = Path(task_dir)
    (task_dir / transport.ARRAYS).mkdir(parents=True, exist_ok=True)
    samples, context = history.shape[0], history.shape[1]
    horizon = forecast_index.shape[1]
    if missing not in ('forbid', 'mask'):
        raise LaunchError(f'missing {missing!r}: expected forbid or mask')

    # Refused before anything is written, not part-way through: an over-budget staging
    # operation must not start model work, and the cheapest way to guarantee that is to
    # never begin it. The estimate is the payloads themselves; the npy headers and the masks
    # add a little, which is why the budget is a bound and not an exact accounting.
    if budget_bytes is not None:
        # Transport promotes floating inputs to float64 and writes an npy header for
        # every array. The caller's float32 nbytes is not the amount written to disk.
        planned = (int(np.asarray(history).size) * 8 + int(np.asarray(history_index).size) * 8
                   + int(np.asarray(forecast_index).size) * 8
                   + (int(np.asarray(future).size) * 8 if future is not None else 0)
                   + (int(np.asarray(history).size) + (int(np.asarray(future).size) if future is not None else 0)
                      if missing == 'mask' else 0)
                   + 256 * (3 + int(future is not None) + (1 + int(future is not None) if missing == 'mask' else 0)))
        if planned > int(budget_bytes):
            raise LaunchError(
                f'staging {planned} bytes would exceed the {int(budget_bytes)}-byte budget for '
                f'this attempt; nothing was written and no model work was started')

    from eforecast import runtime_staging
    keep = ((lambda name, descriptor: runtime_staging.reusable(task_dir, name, descriptor))
            if reuse else None)

    def split(name, values):
        """(finite payload, mask) for one array; the mask is True where a value is real."""
        values = np.asarray(values, dtype='float64')
        if missing != 'mask':
            return values, None
        present = np.isfinite(values)
        if present.all():
            return values, present
        filled = np.where(present, values, 0.0)       # a finite filler, never read through the mask
        return filled, present

    history_values, history_mask = split('history', history)
    arrays = {'history': transport.store(task_dir, 'history', history_values, dtype='<f8', reuse=keep),
              'history_index': transport.store(task_dir, 'history_index', history_index, dtype='<i8', reuse=keep),
              'forecast_index': transport.store(task_dir, 'forecast_index', forecast_index, dtype='<i8', reuse=keep)}
    if len(future_covariates):
        if future is None:
            raise LaunchError('future_covariates were named but no future array was staged')
        future_values, future_mask = split('future', future)
        arrays['future'] = transport.store(task_dir, 'future', future_values, dtype='<f8', reuse=keep)
    if missing == 'mask':
        arrays['history_mask'] = transport.store(task_dir, 'history_mask', history_mask, dtype='|b1', reuse=keep)
        if len(future_covariates):
            arrays['future_mask'] = transport.store(task_dir, 'future_mask', future_mask, dtype='|b1', reuse=keep)

    output_scale = output_scale or target_scale
    inversion = 'global_once' if target_scale == 'global_scaled' and output_scale == 'physical' else 'none'
    units = list(target_units) if output_scale == 'physical' else ['global_scaled'] * len(targets)
    request = {
        'version': contract.VERSION, 'operation': 'predict', 'device': device,
        'identity': dict(identity), 'runtime': dict(runtime), 'method': method,
        'component': dict(component),
        'input': {'dataset_id': dataset_id, 'partition_id': partition_id, 'view_id': view_id,
                  'scaler_id': scaler_id, 'samples': int(samples), 'context': int(context),
                  'horizon': int(horizon), 'frequency_ns': int(frequency_ns),
                  'timezone': timezone, 'targets': list(targets),
                  'past_covariates': list(past_covariates),
                  'future_covariates': list(future_covariates),
                  'target_units': list(target_units), 'target_scale': target_scale,
                  'missing': missing, 'arrays': arrays},
        'output': {'target_scale': output_scale, 'units': units, 'inversion': inversion,
                   'quantiles': [float(q) for q in quantiles]},
    }
    request = contract.validate_request(request)
    # The manifest is published last and only after every artifact has been re-read and
    # re-digested, so it never describes a set that is not on disk. The request follows it,
    # because the request is what dispatch reads.
    if manifest:
        runtime_staging.write_manifest(task_dir, arrays)
    transport.write_document(task_dir, transport.REQUEST, request)
    return request


def _secret_mount(method, *, stack):
    """A read-only directory holding one file per credential, or None when none is needed."""
    needed = required_credentials(method)
    if not needed:
        return None
    from eforecast.config import secrets_source

    directory = Path(tempfile.mkdtemp(prefix='eforecast-secret-'))
    stack.append(directory)
    os.chmod(directory, 0o700)
    for variable, name in needed:
        try:
            secret = secrets_source.resolve(name, environment=variable)
        except Exception as error:
            raise LaunchError(f'{method} needs the {name} credential. {error}') from None
        path = directory / variable
        with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w',
                  encoding='utf-8') as handle:
            handle.write(secret.reveal())
    return directory


#: Docker's own rule for a container name.
_NAME = re.compile(r'[^A-Za-z0-9_.-]')


def container_name(request):
    """A deterministic, attempt-specific name for the container serving one request.

    Deterministic because a cancelled or timed-out attempt has to be stoppable by something
    that never held a handle to it, and attempt-specific because two attempts of the same
    task must not be mistaken for one another. The fingerprint covers the identity and the
    runtime, so another start of the same attempt gets its own name.
    """
    identity = request['identity']
    token = contract.fingerprint({'identity': identity, 'runtime': request['runtime']})[:16]
    stem = _NAME.sub('-', f'{identity["run_id"]}-{identity["attempt_id"]}')[:40]
    return f'eforecast-{stem}-{token}'.strip('-')


def remove_container(name, *, timeout=60):
    """Stop and remove a container by name. Absent is success: the point is that it is gone."""
    try:
        result = subprocess.run(['docker', 'rm', '--force', name], capture_output=True, text=True,
                                timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 or 'No such container' in (result.stderr or '')


def cancel(request, *, timeout=60):
    """Stop the container serving this request, if one is running.

    For a coordinator cancelling an attempt it does not hold the process for. Safe to call
    when nothing is running.
    """
    return remove_container(container_name(request), timeout=timeout)


def command(request, *, reference, task_dir, cache, runtime=None, secrets=None, gpus=None,
            offline=True, name=None):
    """The exact docker invocation, built so a test can read it without running it."""
    runtime = runtime if runtime is not None else cache
    from eforecast.compute_worker import current_worker
    context = current_worker()
    if context is not None and (
            any(str(request['identity'][key]) != str(getattr(context, key))
                for key in ('run_id', 'task_id', 'attempt_id'))
            or str(request['identity']['allocation_id']) != str(context.launch_id)):
        raise LaunchError('runtime request does not belong to the worker that is launching it')
    # the container runs as the invoking user so the task directory does not fill with
    # root-owned files -- which leaves it with no passwd entry, so HOME is named explicitly
    # rather than left for getpwuid() to fail on
    argv = ['docker', 'run', '--rm', '--entrypoint', 'python',
            # named so that giving up on the process can still stop the work
            '--name', name or container_name(request),
            '--user', f'{os.getuid()}:{os.getgid()}',
            '-e', f'HOME={RUNTIME_MOUNT}', '-e', f'XDG_CACHE_HOME={RUNTIME_MOUNT}',
            # USER because torch._inductor asks getpass.getuser() for a cache path name,
            # and the compiled-kernel caches are pinned under this image's own mount so two
            # builds never read each other's kernels (A16-02)
            '-e', 'USER=eforecast',
            '-e', f'TORCHINDUCTOR_CACHE_DIR={RUNTIME_MOUNT}/inductor',
            '-e', f'TRITON_CACHE_DIR={RUNTIME_MOUNT}/triton',
            '-e', f'TORCH_HOME={RUNTIME_MOUNT}/torch',
            '-e', f'HF_HOME={CACHE_MOUNT}',
            '-v', f'{Path(task_dir).resolve()}:{TASK_MOUNT}',
            '-v', f'{Path(cache).resolve()}:{CACHE_MOUNT}',
            '-v', f'{Path(runtime).resolve()}:{RUNTIME_MOUNT}']
    if secrets is not None:
        argv += ['-v', f'{Path(secrets).resolve()}:{SECRET_MOUNT}:ro',
                 '-e', f'EFORECAST_SECRETS={SECRET_MOUNT}']
    if offline:
        # an offline run proves the cache really holds the checkpoint, instead of quietly
        # re-downloading it and reporting success
        argv += ['--network', 'none', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1']
    if request['device'] == 'cuda':
        granted = f'device={context.device_uuid}' if context is not None and context.device_uuid else gpus
        if not granted or granted == 'all':
            raise LaunchError('CUDA model work needs an explicitly assigned GPU UUID')
        argv += ['--gpus', granted]
    if context is not None:
        # The libraries in the container are told how many threads to use. The container
        # itself is not capped: it is a process of this run like the worker that starts it.
        threads = max(1, int(context.threads))
        for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            argv += ['-e', f'{variable}={threads}']
    # The neutral contract, not `run_reporting`: that module owns reporting and names A09's
    # allocator to do it, which would put the whole training package on the inference
    # image's import graph for the sake of two container labels.
    from eforecast.managed_identity import context as reporting_context
    reporting = reporting_context()
    if reporting is not None:
        # A09 discovers these even if the docker client/parent dies before returning a PID.
        argv += ['--label', f'eforecast.managed_run={reporting["run_id"]}',
                 '--label', f'eforecast.executor={reporting["executor_id"]}']
    argv += [reference, '-m', 'eforecast.runtime_adapters', 'run', TASK_MOUNT]
    return argv


def launch(task_dir, request, *, reference=None, gpus=None, offline=True, timeout=TIMEOUT,
           cache_root_path=None, runner=None):
    """Serve a staged request in its image and return the validated result document."""
    method = request['method']
    reference = reference or image_for(method)
    cache, runtime = cache_root(reference, root=cache_root_path)
    name = container_name(request)
    stack = []
    try:
        secrets = _secret_mount(method, stack=stack)
        argv = runner or command(request, reference=reference, task_dir=task_dir, cache=cache,
                                 runtime=runtime, secrets=secrets, gpus=gpus, offline=offline,
                                 name=name)
        try:
            outcome = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except BaseException:
            # Giving up on the process is not giving up on the work. `subprocess.run` kills
            # the docker *client* on timeout and the container keeps running: it holds its
            # CPU or GPU, and it can still publish a result into a task directory the host
            # has abandoned -- which a retry would then read as its own. Measured, not
            # assumed rather than measured.
            remove_container(name)
            raise
    finally:
        for directory in stack:
            shutil.rmtree(directory, ignore_errors=True)

    if outcome.returncode != 0:
        raise LaunchError(_failure(task_dir, reference, outcome))
    result = contract.validate_result(transport.read_document(task_dir, transport.RESULT), request)
    return result


def _failure(task_dir, reference, outcome):
    """The adapter's own published reason when it has one, the process output when it does not."""
    from eforecast.config import secrets_source

    try:
        published = transport.read_document(task_dir, transport.FAILURE)
    except (OSError, ValueError):
        published = None
    if published:
        text = f'{reference}: {published.get("detail", "failed")}\n{published.get("remedy", "")}'
    else:
        tail = (outcome.stderr or outcome.stdout or '').strip().splitlines()[-6:]
        text = (f'{reference} exited {outcome.returncode} without publishing a reason:\n'
                + '\n'.join(tail))
    # a model's own message can quote a key it was handed; never widen the exposure by
    # putting it into a run record or a support report
    return secrets_source.redact(text)


def collect(task_dir, result):
    """The prediction, and the quantile block when one was requested."""
    arrays = result['arrays']
    prediction = transport.load(task_dir, arrays['prediction'], 'result.arrays.prediction')
    spread = (transport.load(task_dir, arrays['quantiles'], 'result.arrays.quantiles')
              if 'quantiles' in arrays else None)
    return prediction, spread


def capability(method, *, reference=None, timeout=120, runner=None):
    """Ask the image what it supports, without running a model or touching a checkpoint."""
    reference = reference or image_for(method)
    argv = runner or ['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python',
                      reference, '-m', 'eforecast.runtime_adapters', 'capability', method]
    outcome = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if outcome.returncode != 0:
        raise LaunchError(f'{reference} cannot declare {method}: '
                          f'{(outcome.stderr or outcome.stdout).strip()[-300:]}')
    import json
    return json.loads(outcome.stdout)


def forecast(method, *, identity, runtime, component, task_dir=None, **staged):
    """Stage, run and read back one forecast; the convenience the pipeline calls."""
    owned = task_dir is None
    directory = Path(task_dir or tempfile.mkdtemp(prefix='eforecast-task-'))
    try:
        request = prepare(directory, method=method, identity=identity, runtime=runtime,
                          component=component, **staged)
        result = launch(directory, request)
        prediction, spread = collect(directory, result)
        return np.asarray(prediction), (None if spread is None else np.asarray(spread))
    finally:
        if owned:
            shutil.rmtree(directory, ignore_errors=True)


def registry_entries(*, references=None):
    """A16-01 runtime definitions for the images present on this host.

    A16-02 builds and qualifies the images; A16-03 owns dispatch. This is the handoff
    between them: the registry insists on immutable locators, so the definitions are keyed
    by each image's id rather than by its tag -- a tag that moves would otherwise silently
    re-point an approved route at a different build.
    """
    from eforecast.runtime_adapters import CATALOGUE
    from eforecast import runtime_registry

    entries = []
    for adapter, reference in sorted((references or IMAGES).items()):
        identity = image_runtime.image_id(reference)
        if identity is None:
            continue
        name = reference.split(':')[0]
        entries.append(runtime_registry.validate_definition({
            'id': adapter,
            'kind': 'container',
            'locator': identity,
            'methods': sorted(case for case, served in CASE_METHODS.items() if served in CATALOGUE[adapter]),
            'devices': ['cpu', 'cuda'],
            'contract': 1}))
    return {'version': 1, 'runtimes': entries}
