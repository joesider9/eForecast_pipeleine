# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Launching one trial in a container: argv, ownership and cleanup.

The command used to be a **string**. ``run_training.mounts_for`` built
``-v '/a path':'/a path'`` with shell quoting, put it in an environment variable, and
``docker/run_trial.sh`` expanded it unquoted; the shell splits such a value on whitespace but does
not re-read the quotes, so a host folder with a space in it became two arguments with literal
apostrophes in them. The same applies to ``EFORECAST_TRIAL_CMD``, which the orchestrator passes
through ``shlex.split`` after formatting the path into it. Nothing here goes through a shell any
more: the command is a list of arguments, and a path is one element of it whatever it contains.

Ownership is a container **name** and a set of labels -- the run id, the case, the trial and the
host pid of the orchestrator -- so a container can be found, stopped and told apart from anything
else on the machine. Labels, not the name prefix: this host also runs ``eforecast-grafana-1`` and
friends, and a cleanup that matched on the name would stop the monitoring stack.
"""
import hashlib
import json
import logging
import os
import re
import subprocess

log = logging.getLogger(__name__)

LABEL_RUN, LABEL_CASE, LABEL_TRIAL = 'eforecast.run', 'eforecast.case', 'eforecast.trial'
LABEL_PID, LABEL_ATTEMPT, LABEL_TASK = 'eforecast.pid', 'eforecast.attempt', 'eforecast.task'
LABEL_MANAGED_RUN, LABEL_EXECUTOR = 'eforecast.managed_run', 'eforecast.executor'
NAME_LIMIT = 118          # the readable part; the digest and its separator take the rest of 128
DEFAULT_IMAGE = 'eforecast-worker:latest'
# The runner inside the image, relative to its WORKDIR -- never the host path of this file: the
# repository is not mounted, only the model tree is.
RUNNER = os.environ.get('EFORECAST_WORKER_RUNNER') or 'eforecast/training/command_line/trial_runner.py'
SHM_SIZE = os.environ.get('EFORECAST_DOCKER_SHM_SIZE', '8g')


def docker_available():
    try:
        subprocess.run(['docker', 'version', '--format', '{{.Server.Version}}'], check=True,
                       capture_output=True, timeout=20)
    except Exception as error:  # noqa: BLE001
        log.warning('docker is not usable here (%r)', error)
        return False
    return True


def image_identity(image):
    """The digest of the image a run is using, so a run record names the code it ran."""
    try:
        out = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                             check=True, capture_output=True, text=True, timeout=30)
    except Exception as error:  # noqa: BLE001
        log.warning('cannot read the identity of image %s (%r)', image, error)
        return None
    return out.stdout.strip() or None


def host_folders(static_data):
    """The host folders a trial refers to, de-duplicated and without a nested folder of another."""
    folders = [static_data['path_group']]
    for key in ('sat_folder',):
        if static_data.get(key) and os.path.isdir(static_data[key]):
            folders.append(static_data[key])
    for nwp in static_data.get('NWP') or []:
        if nwp.get('path_nwp_source') and os.path.isdir(nwp['path_nwp_source']):
            folders.append(nwp['path_nwp_source'])
    filename = static_data.get('filename')
    if filename and (os.path.isdir(filename) or os.path.isfile(filename)):
        folders.append(os.path.dirname(filename))
    unique = []
    for folder in folders:
        folder = os.path.abspath(folder)
        if not any(folder == kept or folder.startswith(kept + os.sep) for kept in unique):
            unique = [kept for kept in unique if not kept.startswith(folder + os.sep)] + [folder]
    return unique


def container_name(run_id, trial, attempt=0, task=''):
    """A name Docker accepts, unique to one attempt of one task of one run.

    The trial number alone is not an identity: every study of a run starts at trial 0, so the
    name was ``eforecast-<run>-0-0`` for the global study and for every cluster study at once, and
    two of them launching together would compete for it. The name now carries the task -- the
    study or transfer unit -- and ends in a digest of the whole identity, so two names cannot
    collide even when the readable part is sanitised to what Docker accepts or truncated to fit.
    """
    identity = '|'.join((str(run_id), str(task), str(trial), str(attempt)))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:8]
    raw = '-'.join(part for part in ('eforecast', str(run_id), str(task), str(trial), str(attempt)) if part)
    readable = re.sub(r'[^a-zA-Z0-9_.-]', '-', raw).strip('-_.') or 'eforecast-trial'
    return f'{readable[:NAME_LIMIT]}-{digest}'


def trial_argv(path_weights, gpu, refit=False, *, image=None, folders=(), run_id='', case='', trial='',
               attempt=0, task='', name=None, runner=None, image_id=None):
    """The whole ``docker run`` command as a list of arguments. No shell, ever.

    ``name`` is passed by the caller that will later have to stop this container, so launch and
    cleanup use the one name rather than each deriving it again. The container gets a card and
    the thread counts of its trial, and no cap on CPU or memory: it is a process of the run.
    """
    image = image or os.environ.get('EFORECAST_WORKER_IMAGE') or DEFAULT_IMAGE
    runner = runner or RUNNER
    argv = ['docker', 'run', '--rm', '--gpus', f'device={gpu}', f'--shm-size={SHM_SIZE}',
            '--name', name or container_name(run_id, trial, attempt, task),
            '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'HOME=/tmp',
            '-e', 'EFORECAST_MEM_FRACTION', '-e', 'PYTORCH_ALLOC_CONF',
            '-e', f'EFORECAST_RUN_ID={run_id}', '-e', f'EFORECAST_TASK_ID={task}', '-e', f'EFORECAST_ATTEMPT={attempt}',
            *(['-e', f'EFORECAST_IMAGE_ID={image_id}'] if image_id else []),
            '--label', f'{LABEL_RUN}={run_id}', '--label', f'{LABEL_CASE}={case}',
            '--label', f'{LABEL_TASK}={task}',
            '--label', f'{LABEL_TRIAL}={trial}', '--label', f'{LABEL_ATTEMPT}={attempt}',
            '--label', f'{LABEL_PID}={os.getpid()}']
    from eforecast.training.command_line.resource_reporting import THREAD_ENV
    for key in (*THREAD_ENV, 'EFORECAST_TRIAL_THREADS', 'EFORECAST_LOADER_WORKERS'):
        argv += ['-e', key]
    from eforecast.execution_policy import ENV
    for key in (ENV, 'CUDA_VISIBLE_DEVICES', 'EFORECAST_ESTIMATOR_THREADS', 'OMP_MAX_ACTIVE_LEVELS',
                'OMP_NESTED', 'MKL_DYNAMIC', 'TOKENIZERS_PARALLELISM', 'JAX_PLATFORMS',
                'XLA_PYTHON_CLIENT_PREALLOCATE', 'OPENCV_FOR_THREADS_NUM'):
        argv += ['-e', key]
    from eforecast.model_library import docker_arguments
    argv += docker_arguments()
    for folder in folders:
        argv += ['-v', f'{folder}:{folder}']        # one argument, whatever the path contains
    argv += ['--entrypoint', 'python', image, runner, '--path-weights', str(path_weights), '--gpu', '0']
    if refit:
        argv.append('--refit')
    from eforecast.config.run_reporting import CONTEXT_ENV, context
    if context() is not None:
        # The managed journal is a private host directory shared with these owned workers.
        # Insert Docker options before the image, preserving the existing trial argv.
        extra = []
        ctx = context()
        extra += ['--label', f'{LABEL_MANAGED_RUN}={ctx["run_id"]}',
                  '--label', f'{LABEL_EXECUTOR}={ctx["executor_id"]}']
        for key in (*CONTEXT_ENV.values(), 'EFORECAST_RUN_REPORT_DIR',
                    'EFORECAST_MANAGED_STUDY_NODE', 'EFORECAST_MANAGED_TRIAL_KEY'):
            extra += ['-e', key]
        report_dir = os.environ['EFORECAST_RUN_REPORT_DIR']
        extra += ['-v', f'{report_dir}:{report_dir}']
        at = argv.index('--entrypoint')
        argv[at:at] = extra
    return argv


# ------------------------------------------------------------------ lifecycle

def create_trial(argv, env):
    """Create without starting, so the exact container ID can be attached before execution."""
    if argv[:2] != ['docker', 'run']:
        raise ValueError('expected docker run argv')
    result = subprocess.run(['docker', 'create', *argv[2:]], env=env,
                            check=True, capture_output=True, text=True, timeout=30)
    identity = result.stdout.strip()
    if not re.fullmatch('[0-9a-f]{64}', identity):
        raise ValueError('docker create did not return a full container ID')
    return identity


def host_pid(name):
    """Container init PID on the host, or None until Docker can attribute the worker."""
    try:
        result = subprocess.run(['docker', 'inspect', '--format', '{{.State.Pid}}', name],
                                check=True, capture_output=True, text=True, timeout=2)
        pid = int(result.stdout.strip())
        return pid if pid > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def containers(run_id=None):
    """``[{name, id, run, trial, pid}]`` of the eforecast trial containers now running.

    Selected by label, so nothing else on the machine is ever in this list.
    """
    command = ['docker', 'ps', '--no-trunc', '--filter', f'label={LABEL_RUN}']
    if run_id:
        command += ['--filter', f'label={LABEL_RUN}={run_id}']
    command += ['--format', '{{json .}}']
    try:
        out = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except Exception as error:  # noqa: BLE001
        log.debug('cannot list the trial containers (%r)', error)
        return []
    found = []
    for line in out.stdout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        labels = dict(part.split('=', 1) for part in (row.get('Labels') or '').split(',') if '=' in part)
        record = {'name': row.get('Names'), 'id': row.get('ID'), 'run': labels.get(LABEL_RUN),
                  'task': labels.get(LABEL_TASK), 'trial': labels.get(LABEL_TRIAL), 'pid': labels.get(LABEL_PID)}
        found.append(record)
    return found


def stop(names, timeout=10):
    """Stop containers by name; a container that is already gone is not an error."""
    names = [name for name in names if name]
    if not names:
        return []
    try:
        # --timeout, not --time: the old spelling is deprecated and docker prints a notice about it
        # on stdout, which is the same stream the stopped names come back on
        out = subprocess.run(['docker', 'stop', '--timeout', str(timeout), *names],
                             capture_output=True, text=True, timeout=timeout + 30)
    except Exception as error:  # noqa: BLE001
        log.warning('cannot stop %s (%r)', ', '.join(names), error)
        return []
    asked = set(names)
    stopped = [line.strip() for line in out.stdout.splitlines() if line.strip() in asked]
    if stopped:
        log.warning('stopped %d trial container(s): %s', len(stopped), ', '.join(stopped))
    return stopped


def _alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def reconcile(run_id):
    """Stop trial containers whose orchestrator is gone.

    ``docker run --rm`` removes a container when it exits, and the client forwards a SIGTERM to it
    -- but a client that is SIGKILLed leaves the container running with nothing attached to it,
    holding a GPU that the next run will not see as free. An orphan is a labelled container whose
    recorded host pid no longer exists; a container of another *live* run is left alone, as its
    trial is.
    """
    orphans = [c for c in containers() if c['run'] != run_id and not _alive(c['pid'])]
    if not orphans:
        return []
    log.warning('%d trial container(s) left by a run that is gone: %s', len(orphans),
                ', '.join(f"{c['name']} (run {c['run']}, pid {c['pid']})" for c in orphans))
    return stop([c['name'] for c in orphans])
