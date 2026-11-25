# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A13-05: one command that says whether this installation is working, and what to do if not.

    python -m eforecast.config.doctor [--case NAME] [--json] [--artifacts PATH]

Exit codes, so a script can act on it:

    0  ready           every check passed
    1  needs attention something is degraded but work can proceed
    2  cannot start    something must be fixed first
    3  the doctor itself could not run

Read-only by construction. It installs nothing, retags nothing, records no witness, migrates no
artifact and restarts no service. It opens no model checkpoint and no weather file. Where a
check cannot be performed, it says so -- an unknown is reported as an unknown, never as a pass.

Every finding carries the same five parts as the A13-04 readiness report, and uses that module's
check functions rather than a second implementation of them: what was checked, what was found,
what follows from it, one thing to do, and **who** should do it -- a colleague correcting a path
or a date, or the maintainer restoring a service or an environment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from eforecast.config import readiness
from eforecast.config.readiness import ATTENTION, BLOCKING, CANNOT_START, NEEDS_ATTENTION, OK, READY

#: Who acts on a finding. The distinction matters: a colleague should not be told to restore a
#: service, and a maintainer should not be left guessing which case a path belongs to.
COLLEAGUE, MAINTAINER = 'colleague', 'maintainer'

EXIT = {READY: 0, NEEDS_ATTENTION: 1, CANNOT_START: 2}

#: Artifacts are sampled, not inventoried: the lab holds on the order of 100k retained files.
ARTIFACT_SAMPLE = 60


def _finding(check, severity, checked, finding, consequence, remedy, who=MAINTAINER, field=None):
    entry = readiness._finding(check, field, severity, checked, finding, consequence, remedy)
    entry['who'] = who
    return entry


# ---------------------------------------------------------------------------- workspace

def workspace() -> list[dict[str, Any]]:
    """Where users, tenants and cases live, and whether this account can write there."""
    from eforecast.config.service import WORKSPACE_REMEDY, WorkspaceNotConfigured, workspace_root
    try:
        root = workspace_root()
    except WorkspaceNotConfigured as error:
        return [_finding('workspace', BLOCKING, 'the configured workspace directory',
                         str(error).splitlines()[0], 'nothing can be read or saved',
                         ' '.join(WORKSPACE_REMEDY.splitlines()[:2]).strip(), COLLEAGUE)]
    if not root.exists():
        return [_finding('workspace', BLOCKING, f'the workspace {root}',
                         'the directory does not exist',
                         'cases, users and run records have nowhere to live',
                         f'Create {root}, or set EFORECAST_WORKSPACE to a directory that exists.',
                         COLLEAGUE)]
    if not os.access(root, os.W_OK):
        return [_finding('workspace', BLOCKING, f'the workspace {root}',
                         'this account cannot write there',
                         'saving a case or starting a run will fail',
                         f'Give this account write access to {root}, or point '
                         'EFORECAST_WORKSPACE somewhere it can write.', COLLEAGUE)]
    return [_finding('workspace', OK, 'the configured workspace directory',
                     f'{root} exists and is writable', 'cases, users and run records have a home',
                     'Nothing to do.')]


# ---------------------------------------------------------------------------- interpreter

def interpreter() -> list[dict[str, Any]]:
    """The interpreter the runs will use, and whether it can do what the lab needs."""
    import importlib.metadata as metadata
    configured = os.environ.get('EFORECAST_PYTHON') or sys.executable
    findings = [_finding('interpreter', OK, 'which interpreter the installation selects for native work', configured,
                         'native work uses this interpreter; isolated model families use their approved runtimes',
                         'Nothing to do.')]
    if configured != sys.executable:
        findings.append(_finding(
            'interpreter.self', ATTENTION, 'whether the doctor is running in that interpreter',
            f'the doctor is {sys.executable}', 'the checks below describe this interpreter, '
            'which is not the one the runs use, so a package it lacks may still be present there',
            f'Run the doctor with the configured interpreter: {configured} -m eforecast.config.doctor'))
    versions = {}
    for name in ('pandas', 'numpy', 'scikit-learn', 'torch'):
        try:
            versions[name] = metadata.version(name)
        except Exception:                                       # noqa: BLE001
            versions[name] = None
    absent = [name for name, value in versions.items() if value is None]
    findings.append(_finding(
        'interpreter.core', BLOCKING if absent else OK, 'the core numerical packages',
        ', '.join(f'{k} {v or "ABSENT"}' for k, v in versions.items()),
        'no stage can run without these' if absent else 'the core stack is present',
        f'Ask the maintainer to install {", ".join(absent)}.' if absent else 'Nothing to do.'))
    findings.append(readiness._estimator_finding() | {'who': MAINTAINER})
    return findings


def families(document=None) -> list[dict[str, Any]]:
    """Each model family, and whether it can run here. A spec lookup, never an import."""
    findings = []
    wanted = sorted(readiness.FAMILY_DISTRIBUTIONS)
    if document is not None:
        selected = [name for name, on in (readiness._at(document, 'methods.methods_cpu') or {}).items() if on]
        wanted = sorted(set(selected) & set(readiness.FAMILY_DISTRIBUTIONS)) or wanted
    for method in wanted:
        module, distribution = readiness.FAMILY_DISTRIBUTIONS[method]
        present = readiness._available(module)
        findings.append(_finding(
            'families', OK if present else ATTENTION, f'whether {method} can run here',
            f'{distribution} {"is installed" if present else "is not installed"}',
            f'{method} trials can start' if present else
            f'{method} is unavailable; other methods are unaffected',
            'Nothing to do.' if present else
            f'Ask the maintainer to install {distribution}, or clear {method} in the case.',
            MAINTAINER if not present else MAINTAINER, 'methods.methods_cpu'))
    grib = [module for module in readiness.GRIB_MODULES if not readiness._available(module)]
    findings.append(_finding(
        'families.grib', ATTENTION if grib else OK, 'whether weather files can be read here',
        f'{", ".join(grib)} missing' if grib else 'pygrib and eccodes are installed',
        'cases with weather inputs cannot be prepared' if grib else 'weather extraction can run',
        f'Ask the maintainer to install {", ".join(grib)}.' if grib else 'Nothing to do.'))
    return findings


# ---------------------------------------------------------------------------- services

def services(timeout=3.0) -> list[dict[str, Any]]:
    """Is the API answering, and is a coordinator actually holding its service lock?

    Neither question is answered by looking for a process name. A process called ``python``
    with the right arguments may be a stale shell, a container, or another user's; and a
    coordinator that has exited may leave its name in somebody's scrollback. The API is asked
    over HTTP, and the coordinator is identified by the lifetime lock it holds -- the same lock
    ``run_coordinator.ensure_service`` uses to decide whether to start another one.
    """
    findings = [_api_finding(timeout)]
    findings.extend(_coordinator_finding())
    return findings


def _api_finding(timeout):
    import urllib.error
    import urllib.request
    host = os.environ.get('EFORECAST_HOST', '127.0.0.1')
    host = '127.0.0.1' if host in ('0.0.0.0', '::', '') else host
    port = os.environ.get('EFORECAST_PORT', '8000')
    url = f'http://{host}:{port}/api/me'
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            code = response.status
    except urllib.error.HTTPError as error:
        code = error.code                                       # answering, just not to us
    except Exception as error:                                  # noqa: BLE001
        return _finding('services.api', ATTENTION, f'whether the web UI answers on {url}',
                        f'no answer ({type(error).__name__})',
                        'the pages and the case editor are unavailable; runs already started '
                        'continue, because the coordinator is a separate process',
                        'Ask the maintainer to start it: sudo systemctl start eforecast-webui')
    return _finding('services.api', OK, f'whether the web UI answers on {url}',
                    f'HTTP {code}', 'the pages and the case editor are reachable', 'Nothing to do.')


def _coordinator_finding():
    """Held lock means running. Acquirable means nothing is running for this workspace."""
    from eforecast.config.service import WorkspaceNotConfigured, Workspace
    try:
        scope = Workspace()
    except WorkspaceNotConfigured:
        return [_finding('services.coordinator', ATTENTION, 'whether a run coordinator is running',
                         'the workspace is not configured, so its service identity is unknown',
                         'whether submitted runs would be picked up cannot be determined',
                         'Fix the workspace finding above, then re-check.', COLLEAGUE)]
    try:
        held, folder = _service_lock_held(scope)
    except Exception as error:                                  # noqa: BLE001
        return [_finding('services.coordinator', ATTENTION, 'whether a run coordinator is running',
                         f'the service lock could not be inspected ({type(error).__name__}: '
                         f'{str(error)[:90]})',
                         'whether submitted runs would be picked up is unknown -- this is not '
                         'evidence that it is down',
                         'Ask the maintainer to check the coordinator and its executor directory.')]
    if held:
        return [_finding('services.coordinator', OK, 'whether a run coordinator is running',
                         f'one holds the service lock for this workspace ({folder.name})',
                         'submitted runs are picked up and supervised', 'Nothing to do.')]
    return [_finding('services.coordinator', ATTENTION, 'whether a run coordinator is running',
                     'nothing holds the service lock for this workspace',
                     'a submitted run stays durably recorded but nothing advances it, so work '
                     'appears to wait forever',
                     'Ask the maintainer to start it: sudo systemctl restart eforecast-webui '
                     '(the service starts the coordinator).')]


def _service_lock_held(scope):
    """``True`` when another process holds it. Acquiring briefly proves only that none does."""
    import hashlib
    from eforecast.config.run_model import pinned
    from eforecast.config.executors import LocalExecutor, locked
    executor = LocalExecutor(workspace=scope.root)
    marker = pinned({'workspace': str(scope.root.resolve()),
                     'store': os.environ.get('EFORECAST_RUN_STORE_PATH'),
                     'dsn': os.environ.get('EFORECAST_RUN_STORE_DSN')
                     or os.environ.get('EFORECAST_ALLOCATOR_DSN'),
                     'namespace': os.environ.get('EFORECAST_RUN_STORE_NAMESPACE', 'default')})
    folder = executor.directory / 'services' / hashlib.sha256(marker.encode()).hexdigest()
    if not folder.is_dir():
        return False, folder
    with locked(folder) as descriptor:
        return descriptor is None, folder


# ---------------------------------------------------------------------------- model cache

def model_cache() -> list[dict[str, Any]]:
    """Are the prewarmed weights present? Listing the cache, never loading or downloading."""
    from eforecast import model_library
    try:
        hub = Path(os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE')
                   or Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub')
        root = model_library.root()
    except Exception as error:                                  # noqa: BLE001
        return [_finding('model_cache', ATTENTION, 'where prewarmed model weights live',
                         f'the cache location could not be resolved ({type(error).__name__})',
                         'whether a foundation model can start offline is unknown',
                         'Ask the maintainer to check HF_HOME and EFORECAST_MODEL_CACHE_DIR.')]
    if not hub.is_dir():
        return [_finding('model_cache', ATTENTION, f'the prewarmed model cache {hub}',
                         'the directory does not exist',
                         'a foundation model would have to download its weights, which an '
                         'offline run refuses',
                         f'Ask the maintainer to prewarm the models into {hub}.')]
    models = sorted(entry.name[len('models--'):].replace('--', '/')
                    for entry in hub.iterdir() if entry.name.startswith('models--'))
    return [_finding('model_cache', OK, f'the prewarmed model cache {hub}',
                     f'{len(models)} models present' + (f' (library root {root})' if root else ''),
                     'foundation models can start without downloading', 'Nothing to do.'),
            _finding('model_cache.models', OK, 'which models are prewarmed',
                     ', '.join(models[:8]) + ('...' if len(models) > 8 else '') or 'none',
                     'a case selecting one of these can run offline',
                     'Nothing to do.')]


# ---------------------------------------------------------------------------- retained artifacts

def artifacts(root=None, sample=ARTIFACT_SAMPLE) -> list[dict[str, Any]]:
    """Can the retained models still be read here, and do they warn?

    Wraps ``artifact_compat`` rather than reimplementing it, and **samples**: the lab holds on
    the order of a hundred thousand retained files, so a full inventory is not a command a
    colleague runs. A sample that finds nothing wrong is reported as a sample.
    """
    if root is None:
        return [_finding('artifacts', ATTENTION, 'whether retained models can still be read',
                         'no model root was given, so none was checked',
                         'whether an existing model still loads here is unknown',
                         'Run the doctor with --artifacts <model root>, or with --case <name> '
                         'so it can find that case\'s own models.', COLLEAGUE)]
    root = Path(root)
    if not root.is_dir():
        return [_finding('artifacts', ATTENTION, f'the model root {root}',
                         'the directory does not exist',
                         'this case has no retained models to read, which is normal before its '
                         'first run',
                         'Nothing to do unless you expected models here.', COLLEAGUE)]
    from eforecast.common_utils import artifact_compat
    found = sorted(root.rglob('*.pickle'))
    if not found:
        return [_finding('artifacts', OK, f'retained models under {root}',
                         'none yet', 'nothing to read; this case has not produced models',
                         'Nothing to do.', COLLEAGUE)]
    chosen = found[:sample]
    records = [artifact_compat.inspect(str(path)) for path in chosen]
    unreadable = [r for r in records if r['status'] == 'unreadable']
    shimmed = [r for r in records if r['status'] == 'compatibility_read']
    warned = [r for r in records if r.get('library_version_warnings')]
    scope = (f'{len(chosen)} of {len(found)}' if len(found) > len(chosen) else f'all {len(found)}')
    findings = [_finding(
        'artifacts', BLOCKING if unreadable else OK,
        f'whether retained models under {root.name} can be read by this interpreter',
        f'{scope} checked: {len(records) - len(unreadable) - len(shimmed)} read directly, '
        f'{len(shimmed)} through the compatibility reader, {len(unreadable)} unreadable',
        'an unreadable model cannot be used for a forecast and cannot be rolled back to'
        if unreadable else 'the sampled models load here',
        f'Ask the maintainer about {Path(unreadable[0]["path"]).name}: '
        f'{unreadable[0].get("error", "")[:90]}' if unreadable else 'Nothing to do.')]
    if warned:
        findings.append(_finding(
            'artifacts.estimators', ATTENTION, 'version warnings from the retained estimators',
            f'{len(warned)} of {len(chosen)} sampled files warn',
            'they load and, for the compared estimator classes, answer identically; a class '
            'that was never compared could differ silently',
            'No action while the configured scikit-learn stays in the compared range; the '
            'estimator finding above reports that.'))
    return findings


# ---------------------------------------------------------------------------- worker image

def image() -> list[dict[str, Any]]:
    """The default worker image, whether it is clean, and whether its witness is findable.

    Includes the trap A13-02 ran into: ``witness_file()`` keys off ``EFORECAST_WORKSPACE``, so a
    witness recorded in an ordinary terminal lands where the service never looks, and activation
    then reports "no recorded witness" right after a successful recording.
    """
    from eforecast import image_runtime
    from eforecast.training.command_line import docker_task
    reference = os.environ.get('EFORECAST_WORKER_IMAGE') or docker_task.DEFAULT_IMAGE
    identity = image_runtime.image_id(reference)
    if identity is None:
        return [_finding('image', ATTENTION, f'whether the worker image {reference} exists here',
                         'the image cannot be inspected; Docker may be unavailable or the image '
                         'not built',
                         'container trials cannot start; native runs are unaffected',
                         f'Ask the maintainer to build {reference}, or check that Docker is running.')]
    findings = [_finding('image', OK, f'the worker image {reference}',
                         f'{identity[:19]}...', 'this is the image a container trial would use',
                         'Nothing to do.')]
    served = image_runtime.witness_file()
    witness = image_runtime.load(reference)
    if witness is not None:
        findings.append(_finding('image.witness', OK, f'a recorded runtime witness for {reference}',
                                 f'recorded for this id (python {witness["hardware"]["python"]})',
                                 'a pinned profile can be matched for container work',
                                 'Nothing to do.'))
        return findings
    elsewhere = _witness_recorded_elsewhere(identity)
    if elsewhere:
        findings.append(_finding(
            'image.witness', ATTENTION, f'a recorded runtime witness for {reference}',
            f'a witness for this id exists in {elsewhere}, but not in {served}, which is the '
            'file this configuration reads',
            'pinned profiles cannot be activated for container trials even though the witness '
            'was recorded -- the recording went to a file nothing reads',
            f'Re-record with the workspace set: EFORECAST_WORKSPACE={os.environ.get("EFORECAST_WORKSPACE", "<the service workspace>")} '
            f'python -m eforecast.image_runtime record --image {reference}'))
        return findings
    findings.append(_finding(
        'image.witness', ATTENTION, f'a recorded runtime witness for {reference}',
        f'nothing is recorded for this id in {served}',
        'pinned profiles cannot be activated for container trials; runs proceed on the '
        'established policy',
        f'Ask the maintainer to record one: python -m eforecast.image_runtime record '
        f'--image {reference}'))
    return findings


def _witness_recorded_elsewhere(identity):
    """The other place ``witness_file`` resolves to, when the workspace is not set."""
    from eforecast import image_runtime
    candidates = [Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'eforecast'
                  / 'image-runtime.json']
    served = image_runtime.witness_file()
    for path in candidates:
        if path == served or not path.is_file():
            continue
        try:
            records = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if identity in (records.get('images') or {}):
            return path
    return None


# ---------------------------------------------------------------------------- model runtimes

def model_runtimes(document=None) -> list[dict[str, Any]]:
    """Whether the foundation families a case keeps can actually be served -- A16-02.

    Image presence, witness, and credential presence only.  No checkpoint is downloaded and
    no model package is imported: a colleague opening the UI must not pull thirteen sets of
    weights, and the images are 6 GB each.

    Scoped to the case when one is named, because a case that keeps no zero-shot member does
    not need either image and should not be told about them.
    """
    from eforecast import image_runtime, runtime_launcher
    from eforecast.config import secrets_source

    IMAGE_METHODS = runtime_launcher.CASE_METHODS
    wanted = sorted(IMAGE_METHODS)
    if document is not None:
        selected = [name for name, on in (readiness._at(document, 'methods.methods_cpu') or {}).items() if on]
        wanted = sorted(set(selected) & set(IMAGE_METHODS))
        if not wanted:
            return [_finding('model_runtimes', OK, 'whether this case needs a model image',
                             'it keeps no zero-shot foundation member',
                             'no model image is needed to train or serve it', 'Nothing to do.',
                             COLLEAGUE, 'methods.methods_cpu')]

    findings: list[dict[str, Any]] = []
    needed: dict[str, list[str]] = {}
    for name in wanted:
        try:
            reference = runtime_launcher.image_for(IMAGE_METHODS[name])
        except Exception as error:                                          # noqa: BLE001
            findings.append(_finding(
                'model_runtimes', ATTENTION, f'which image serves {name}', str(error)[:120],
                f'{name} cannot be dispatched', 'Ask the maintainer to pin an image for it.'))
            continue
        needed.setdefault(reference, []).append(name)

    for reference, names in sorted(needed.items()):
        served = ', '.join(names)
        identity = image_runtime.image_id(reference)
        if identity is None:
            findings.append(_finding(
                'model_runtimes', ATTENTION, f'whether the model image {reference} exists here',
                'the image cannot be inspected; Docker may be unavailable or it is not built',
                f'{served} cannot run: these families are deliberately not installed on the '
                'host, so the image is the only way to run them',
                f'Ask the maintainer to build it: docker build -f docker/Dockerfile.'
                f'{"lagllama" if "lagllama" in reference else "foundation"} -t {reference} .'))
            continue
        findings.append(_finding(
            'model_runtimes', OK, f'the model image for {served}',
            f'{reference} is {identity[:19]}...',
            'these families can be dispatched to it', 'Nothing to do.'))
        if image_runtime.load(reference, entrypoint=runtime_launcher.ENTRYPOINT_WITNESS) is None:
            findings.append(_finding(
                'model_runtimes.witness', ATTENTION, f'a recorded witness for {reference}',
                'nothing is recorded for this image id',
                'the run record cannot state which runtime produced a forecast',
                f'Ask the maintainer to record one: python -m eforecast.image_runtime record '
                f'--image {reference} --entrypoint {runtime_launcher.ENTRYPOINT_WITNESS}'))

    for name in wanted:
        for variable, secret in runtime_launcher.required_credentials(IMAGE_METHODS[name]):
            present = secrets_source.present(secret) or bool(os.environ.get(variable))
            findings.append(_finding(
                'model_runtimes.credential', OK if present else ATTENTION,
                f'whether the {name} credential is available',
                f'{secret} {"is in the secret source" if present else "is not stored"}',
                f'{name} can be called' if present else
                f'{name} is a hosted model and will fail without its key; the other families '
                'are unaffected',
                'Nothing to do.' if present else
                f'Store it once: bin/eforecast secret set {secret}',
                MAINTAINER, 'methods.methods_cpu'))
        note = runtime_launcher.licence_note(IMAGE_METHODS[name])
        if note:
            findings.append(_finding(
                'model_runtimes.licence', ATTENTION, f'the weight licence of {name}', note,
                'the lab may use it for research; published or revenue-generating use needs '
                'a separate agreement with the rights holder',
                'Keep this result to research use, or choose a family with an Apache-2.0 '
                'checkpoint.', MAINTAINER, 'methods.methods_cpu'))
    return findings


def image_coherence(document=None) -> list[dict[str, Any]]:
    """Is every image a case needs built from the code the selected release names? -- A16-11.

    A wrong forecast from a stale image is the failure this answers: the A13-02 candidate
    predated three fixes and would have produced backcasts rather than an error. Scoped to
    the case when one is named, so an unrelated optional family is not this run's problem.
    """
    from eforecast import image_release

    release = image_release.selected()
    if release is None:
        names = image_release.available()
        return [_finding(
            'image_coherence', ATTENTION, 'which image release this installation runs against',
            f'{len(names)} release(s) are captured and none is selected'
            if names else 'no release has been captured',
            'images cannot be compared against anything, so a stale image would go unnoticed',
            'Ask the maintainer to capture one (python -m eforecast.image_release capture '
            '--release <name>) and set EFORECAST_RELEASE to it.')]

    wanted = image_release.required_for(document) if document is not None else None
    try:
        report = image_release.check(release, runtimes=wanted or None)
    except image_release.ReleaseError as error:
        return [_finding('image_coherence', ATTENTION,
                         f'the image release {release!r}', str(error)[:160],
                         'coherence cannot be established',
                         'Ask the maintainer to capture the release again.')]

    findings = []
    for name, entry in sorted(report['runtimes'].items()):
        if entry['status'] == image_release.COHERENT:
            findings.append(_finding(
                'image_coherence', OK, f'whether {name} matches release {release}',
                f'{entry.get("reference")} is the recorded image and its inputs match',
                'work dispatched to it runs the code this release was qualified with',
                'Nothing to do.'))
            continue
        reasons = ', '.join(sorted({item['reason'] for item in entry['reasons']}))
        detail = '; '.join(item['detail'] for item in entry['reasons'])[:300]
        remedy = (image_release.BUILD_COMMAND.get(name) if entry['rebuild']
                  else f'python -m eforecast.image_runtime record --image '
                       f'{entry.get("reference")}' if entry['requalify'] else None)
        findings.append(_finding(
            'image_coherence', ATTENTION, f'whether {name} matches release {release}',
            f'{entry["status"]}: {reasons} -- {detail}',
            'a run dispatched to this image would not be running the code this release was '
            'qualified with, and the symptom is a wrong number rather than an error',
            f'Ask the maintainer to {remedy}.' if remedy else
            'Ask the maintainer to capture a release that describes it.'))
    for item in report.get('checkpoints', {}).get('moved') or []:
        findings.append(_finding(
            'image_coherence.checkpoint', ATTENTION,
            f'the checkpoint {item["repository"]}', item['detail'],
            'the weights differ from the release; this is not a source change and no image '
            'needs rebuilding for it',
            'Ask the maintainer to restore the recorded revision into the runtime cache, or '
            'capture a release that records this one.'))
    return findings


def topology() -> list[dict[str, Any]]:
    """What this installation is, and what it therefore cannot survive.

    Not a problem to fix: a statement, in the place people look. One control service and one
    store is the right trade for a lab, but it has to be *said*, because the alternative is
    somebody planning around redundancy that was never there. The doctor is where it belongs --
    `end_to_end.md` says it too, and the UI is where a colleague is actually standing.
    """
    from eforecast.config import sql_store
    dsn = os.environ.get('EFORECAST_RUN_STORE_DSN') or os.environ.get('EFORECAST_ALLOCATOR_DSN')
    backend = os.environ.get('EFORECAST_RUN_STORE', 'postgres' if dsn else 'sqlite')
    where = 'one PostgreSQL database' if backend == 'postgres' else 'one SQLite file'
    found = [_finding(
        'topology.control', OK, 'how many control services and stores this installation has',
        f'one API and coordinator, {where}',
        'several workers do not make this highly available: losing the store stops new work '
        'and loses nothing already recorded, but recovery is a restore, not a failover',
        'Nothing to do -- but keep the backup section of docs/administrator.md true: an approved model that '
        'is nowhere else is one disk failure from being gone.', who=MAINTAINER)]
    place = os.environ.get('EFORECAST_BACKUP_DIR')
    if not place:
        # Said here as well as under `backup`, because this is the sentence that explains why
        # the other finding matters.
        found.append(_finding(
            'topology.recovery', ATTENTION, 'whether this installation could be rebuilt',
            'no backup destination is configured, so there is nothing to restore from',
            'with one store and one disk, an unbacked installation has no recovery path at all',
            'Set EFORECAST_BACKUP_DIR to a path on a separate physical disk '
            '(docs/administrator.md, backup and restore).', who=MAINTAINER))
    return found


def _run_store_present():
    """Does the configured run store already exist? Resolved exactly as ``configured_store`` does."""
    if (os.environ.get('EFORECAST_RUN_STORE_DSN') or os.environ.get('EFORECAST_ALLOCATOR_DSN')
            or os.environ.get('EFORECAST_RUN_STORE') == 'postgres'):
        return True                                             # PostgreSQL: connecting only reads
    configured = os.environ.get('EFORECAST_RUN_STORE_PATH')
    if configured:
        return Path(configured).exists()
    state = Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local' / 'state'))
    return (state / 'eforecast' / 'runs.sqlite3').exists()


# ---------------------------------------------------------------------------- the report

def client(url, *, timeout=3.0) -> list[dict[str, Any]]:
    """Can this machine reach the lab? -- A14-04. Checks nothing about a local installation.

    A colleague with a browser needs the server, not CUDA. Running the full doctor on such a
    machine reports a dozen findings about an ML stack it is never going to have, which is
    worse than silence: it buries the one thing that matters. So this asks only what the
    browser route depends on, and says plainly that it has validated nothing local -- a
    remote server being healthy is not evidence about this machine.
    """
    import urllib.error
    import urllib.request

    base = url.rstrip('/')
    findings = []
    try:
        with urllib.request.urlopen(base + '/', timeout=timeout) as response:
            code, body = response.status, response.read(2048)
    except urllib.error.HTTPError as error:
        code, body = error.code, b''                    # answering, just not with a page
    except Exception as error:                          # noqa: BLE001
        findings.append(_finding(
            'client.reachable', BLOCKING, f'whether the lab answers on {base}',
            f'no answer ({type(error).__name__}: {str(error)[:80]})',
            'this machine cannot use the lab at all through the browser',
            f'Check the address, then that this machine is on the lab network. If both are '
            f'right, ask the maintainer whether the service is running and whether the '
            f'firewall admits this machine: sudo systemctl status eforecast-webui',
            COLLEAGUE))
        findings.append(_finding(
            'client.scope', ATTENTION, 'what this check covered',
            'the lab was not reached, so nothing was established',
            'no conclusion about this machine or about the lab',
            'Nothing to do until the address above answers.', COLLEAGUE))
        return findings

    findings.append(_finding(
        'client.reachable', OK, f'whether the lab answers on {base}', f'HTTP {code}',
        'this machine can reach the lab', 'Nothing to do.', COLLEAGUE))

    # 401 or a redirect to the sign-in page is the healthy answer when EFORECAST_AUTH is on;
    # treating it as a fault would send a colleague to the maintainer over working security.
    if code in (401, 403) or (300 <= code < 400):
        findings.append(_finding(
            'client.pages', OK, 'whether the lab asks this machine to sign in',
            f'HTTP {code}', 'the lab is reachable and protected; sign in through the browser',
            'Nothing to do.', COLLEAGUE))
    elif code == 200 and b'<' in body:
        findings.append(_finding(
            'client.pages', OK, 'whether the lab serves its pages', 'it returned a page',
            'the UI will load in a browser', 'Nothing to do.', COLLEAGUE))
    else:
        findings.append(_finding(
            'client.pages', ATTENTION, 'whether the lab serves its pages',
            f'HTTP {code} without a page',
            'the address answers but the UI may not load',
            'Ask the maintainer whether the pages were built: cd webui && npm ci && npm run build'))

    findings.append(_finding(
        'client.scope', OK, 'what this check covered',
        'the lab is reachable from this machine',
        'nothing local was checked: no interpreter, workspace, model library or GPU, because '
        'the browser route needs none of them. A healthy lab is not evidence that this '
        'machine could run a case by itself',
        'Nothing to do. For a machine that will run cases locally, follow the standalone '
        'path in deploy/README.md and run the doctor without --client.', COLLEAGUE))
    return findings


# ---------------------------------------------------------------------------- backup

def protection(destination=None, case=None, now=None) -> list[dict[str, Any]]:
    """A15-04: the last complete backup and the last verified restore, reported **separately**.

    Separately because they fail separately. A fresh backup nobody has restored is untested;
    an old restore of a stale generation proves only that something once worked. One combined
    "backups: ok" would hide whichever of the two is worse, which is the reassurance this check
    exists to withhold.
    """
    from eforecast.config import model_backup
    place = destination or os.environ.get('EFORECAST_BACKUP_DIR')
    if not place:
        return [_finding(
            'backup', ATTENTION, 'where approved models are backed up',
            'no backup destination is configured',
            'rollback is a pointer operation, so it assumes the previous bundle is still on a '
            'disk that did not fail with the first one; nothing here checks that today',
            'Set EFORECAST_BACKUP_DIR to a path on a separate physical disk, then take a '
            'generation. Another directory on the same disk does not survive the disk.')]

    findings = []
    source = os.environ.get('EFORECAST_WORKSPACE') or '.'
    for item in model_backup.destination_findings(source, place):
        findings.append(_finding(
            'backup.destination', OK if item['state'] == 'ok' else BLOCKING,
            'whether the destination survives losing the source disk', item['detail'],
            'this is the outage the backup is for' if item['state'] != 'ok'
            else 'losing the source disk does not lose the backup',
            item.get('remedy', 'Nothing to do.')))

    cases = [case] if case else sorted(
        path.name for path in Path(place).iterdir() if path.is_dir()) if Path(place).is_dir() else []
    if not cases:
        findings.append(_finding(
            'backup.generations', ATTENTION, 'protected cases',
            f'{place} holds no case backups',
            'no approved model here could be recovered from this destination',
            'Take a generation for each case whose model matters.'))
        return findings

    for name in cases:
        state = model_backup.freshness(place, name, now=now)
        for kind in ('backup', 'restore'):
            entry = state[kind]
            severity = {'ok': OK, 'attention': ATTENTION, 'fail': BLOCKING}[entry['state']]
            findings.append(_finding(
                f'backup.{kind}', severity,
                f'the last {"complete backup" if kind == "backup" else "verified restore"} '
                f'of {name}',
                entry['detail'],
                'this case could be recovered' if entry['state'] == 'ok' else
                'recovery of this case is unproven' if kind == 'restore' else
                'this case has no recent protection',
                entry['remedy']))
    return findings


# ---------------------------------------------------------------------------- monitoring

def monitoring(timeout=3.0) -> list[dict[str, Any]]:
    """A16-06: the dispatch transport, the cluster, storage headroom and the scrape endpoint.

    Read-only and bounded, like every other check here. Nothing in this function connects to a
    broker or an API server for longer than ``timeout``, and nothing it prints can contain a
    credential: the broker URL is reported as its host and port, never its userinfo.

    The distinction this check exists to preserve is the same one A16-06's exposition makes: a
    dependency that cannot be reached is *unknown*, and an unknown is reported as an unknown.
    """
    from eforecast.config import fleet_metrics
    findings: list[dict[str, Any]] = []

    # --- where this installation dispatches, which is a decision and not a guess (A16-08)
    try:
        from eforecast.config import dispatch_route
        store = _read_only_store()
        decided = dispatch_route.current(store) if store is not None else None
        if decided is None:
            findings.append(_finding(
                'monitoring.route', OK, 'where work is dispatched',
                'there is no run store yet, so nothing has been decided',
                'a fresh installation runs locally', 'Nothing to do.'))
        elif (blocked_routes := dispatch_route.unavailable_routes(store)):
            findings.append(_finding(
                'monitoring.route', BLOCKING, 'where work is dispatched',
                'Recorded routes cannot execute managed runs: ' + ', '.join(v['scope'] for v in blocked_routes),
                dispatch_route.execution_problem(blocked_routes[0]['venue']),
                'Drain the named scope, then use: eforecast route <scope> local'))
        elif decided['default']:
            findings.append(_finding(
                'monitoring.route', OK, 'where work is dispatched',
                'locally on this machine (no cutover has been decided)',
                'this is the tested path; a broker URL alone does not change it',
                'Nothing to do.'))
        elif dispatch_route.execution_problem(decided['venue']):
            findings.append(_finding(
                'monitoring.route', BLOCKING, 'where work is dispatched',
                f"{decided['venue']} is recorded but cannot execute managed runs",
                dispatch_route.execution_problem(decided['venue']),
                'Drain existing work, then roll back with: eforecast route installation local'))
        else:
            findings.append(_finding(
                'monitoring.route', OK, 'where work is dispatched',
                f'{decided["venue"]} for {decided["scope"]}, revision {decided["revision"]}, '
                f'decided by {decided["decided_by"]}'
                + (f' -- {decided["note"]}' if decided['note'] else ''),
                'runs accepted from now on carry this venue; those already accepted keep theirs',
                'Roll back with: eforecast route <scope> local'))
    except Exception as error:                                  # noqa: BLE001
        findings.append(_finding(
            'monitoring.route', ATTENTION, 'where work is dispatched',
            f'the decision could not be read ({type(error).__name__}: {str(error)[:80]})',
            'a route nobody can read is a cutover nobody can audit',
            'Check the run store, then bin/eforecast route.'))

    # --- the dispatch transport
    configured, where = None, None
    try:
        from eforecast.config import dispatch_broker
        configured = dispatch_broker.broker_url()
        where = _redacted_url(configured)
    except Exception as error:                                  # noqa: BLE001
        findings.append(_finding(
            'monitoring.broker', ATTENTION, 'whether a dispatch broker is configured',
            'none is configured', 'dispatch records are written and kept, but nothing publishes '
            'them, so eligible work waits instead of running',
            f'Set EFORECAST_BROKER_URL or store the secret: bin/eforecast secret set broker_url. '
            f'({str(error)[:80]})'))
    if configured:
        findings.append(_finding(
            'monitoring.broker', OK, 'which dispatch broker is configured', where,
            'this is where eligible work is published', 'Nothing to do.'))

    # --- the outbox, which is the durable record whether or not a broker exists
    try:
        from eforecast.config import dispatch_outbox
        store = _read_only_store()
        if store is None:
            findings.append(_finding(
                'monitoring.dispatch', OK, 'the dispatch backlog',
                'there is no run store yet, so nothing has been dispatched',
                'a fresh installation has no backlog, which is not the same as a stuck one',
                'Nothing to do.'))
            raise _NothingToRead
        lines, note = fleet_metrics.dispatch_samples(store)
        if note:
            raise RuntimeError(note)
        rows = {line.split('"')[1]: line.rsplit(' ', 1)[1] for line in lines
                if line.startswith('eforecast_dispatch_rows{')}
        age = next((float(line.rsplit(' ', 1)[1]) for line in lines
                    if line.startswith('eforecast_dispatch_oldest_age_seconds')), 0.0)
        waiting = int(rows.get('pending', '0'))
        stuck = waiting and age > 600
        # A configured broker URL is not a running publisher. `dispatch_broker.drain` and
        # `.dispatch` are what move these rows, and no installed service runs either yet
        # (R27-12). Saying "check the broker" to someone whose broker is fine, while the real
        # answer is that nothing is draining the outbox on this installation, sends them to
        # the wrong place. Evidence, not configuration: a row that was ever published or
        # claimed proves something drains this outbox.
        drained = any(int(rows.get(state, '0')) for state in ('published', 'claimed', 'failed'))
        findings.append(_finding(
            'monitoring.dispatch', ATTENTION if stuck else OK, 'the dispatch backlog',
            f'{waiting} waiting to publish, oldest {age:.0f}s; '
            f'{rows.get("claimed", "0")} handed off, {rows.get("failed", "0")} failed',
            'work that stays here is recorded and recoverable, but it is not running'
            if stuck else 'nothing is stuck waiting to be published',
            ('Nothing has ever been published or claimed from this outbox, so no publisher or '
             'consumer is running here, and a broker URL alone does not move these rows. '
             'Routing this installation through a broker is a deliberate cutover, described '
             'in docs/administrator.md; until it is done, run the work locally.'
             if stuck and not drained else
             'Check the broker above, then the publisher.') if stuck else 'Nothing to do.'))
        undeliverable = dispatch_outbox.undeliverable(store, limit=64)
        if undeliverable:
            findings.append(_finding(
                'monitoring.undeliverable', ATTENTION,
                'deliveries that were not dispatch messages',
                f'{len(undeliverable)} distinct, {sum(r["deliveries"] for r in undeliverable)} '
                f'arrivals; most recent reason: {undeliverable[-1]["reason"][:70]}',
                'something is publishing to the dispatch exchange that this version cannot '
                'read; those deliveries are recorded and dropped, so no work is lost',
                'Usually a version mismatch between publisher and consumer. Compare the '
                'release of whatever publishes with this installation.'))
    except _NothingToRead:
        pass                                            # already reported, and nothing was opened
    except Exception as error:                                  # noqa: BLE001
        findings.append(_finding(
            'monitoring.dispatch', ATTENTION, 'the dispatch backlog',
            f'it could not be read ({type(error).__name__}: {str(error)[:90]})',
            'nothing can be said about queued work; this is an unknown, not an empty queue',
            'Check the run store. A backlog that cannot be read is also a backlog that cannot '
            'be published.'))

    # --- the cluster, only if one is configured
    try:
        from eforecast.config.kubernetes_api import Cluster
        cluster = Cluster.available(timeout=timeout)
    except Exception:                                           # noqa: BLE001
        cluster = None
    if cluster is None:
        findings.append(_finding(
            'monitoring.cluster', OK, 'whether a Kubernetes cluster is configured',
            'none is configured, so work runs locally',
            'the local executor is the only path, which is a complete configuration',
            'Nothing to do.'))
    else:
        try:
            version = cluster.version().get('gitVersion', 'unknown')
            from eforecast.config.kubernetes_executor import KubernetesExecutor
            queued = KubernetesExecutor(cluster).queued()
            findings.append(_finding(
                'monitoring.cluster', OK, 'the configured Kubernetes cluster',
                f'{cluster.server} is {version}, namespace {cluster.namespace}, '
                f'{queued} worker Job(s) unfinished',
                'this is where routed attempts run', 'Nothing to do.'))
        except Exception as error:                              # noqa: BLE001
            findings.append(_finding(
                'monitoring.cluster', BLOCKING, 'the configured Kubernetes cluster',
                f'{cluster.server} could not be reached '
                f'({type(error).__name__}: {str(error)[:80]})',
                'running Jobs are unaffected, but none can be launched, observed or cancelled, '
                'and an attempt already handed off stays recoverable rather than lost',
                'Check the API server and the executor token; the doctor does not retry.'))

    # --- storage headroom, as the ratio the alert uses
    lines, note = fleet_metrics.storage_samples(os.environ.get('EFORECAST_WORKSPACE') or '.')
    if note:
        findings.append(_finding(
            'monitoring.storage', BLOCKING, 'workspace free space',
            f'the filesystem could not be stated ({note[:80]})',
            'usually the workspace is not mounted, in which case nothing can run',
            'Check the mount and EFORECAST_WORKSPACE.'))
    else:
        free = next(float(line.rsplit(' ', 1)[1]) for line in lines
                    if line.startswith('eforecast_storage_free_bytes'))
        total = next(float(line.rsplit(' ', 1)[1]) for line in lines
                     if line.startswith('eforecast_storage_total_bytes'))
        share = (free / total) if total else 0.0
        findings.append(_finding(
            'monitoring.storage', ATTENTION if share < 0.10 else OK, 'workspace free space',
            f'{free / 1024 ** 3:.0f} GiB free of {total / 1024 ** 3:.0f} GiB ({share:.0%})',
            'stages and trials will start failing on writes, leaving no partial artifact'
            if share < 0.10 else 'there is room for the stages and trials to write',
            'Free space or move retained artifacts off the workspace.' if share < 0.10
            else 'Nothing to do.'))

    # --- the scrape endpoint, which is what the dashboards are fed by
    port = int(os.environ.get('EFORECAST_FLEET_METRICS_PORT', '9109') or 0)
    if not port:
        findings.append(_finding(
            'monitoring.scrape', OK, 'the metrics endpoint', 'it is turned off (port 0)',
            'the dashboards will have no application data, which they show as no data',
            'Nothing to do, unless you expect Grafana to show this installation.'))
    else:
        reachable, detail = _scrape_health(port, timeout)
        findings.append(_finding(
            'monitoring.scrape', OK if reachable else ATTENTION, 'the metrics endpoint',
            detail,
            'Prometheus can read this installation' if reachable else
            'the dashboards fed by it show no data; they are not showing healthy zeroes, but '
            'someone who does not look at the alert may read a blank panel as a quiet system',
            'Nothing to do.' if reachable else
            f'Start it: {sys.executable} -m eforecast.config.metrics_server --port {port}'))
    return findings


class _NothingToRead(Exception):
    """There was nothing to open, and the doctor opened nothing."""


class _ReadOnlyStore:
    """Just enough of the run store's interface for the metrics queries, and no migration.

    The doctor is read-only by construction, and `configured_store` is not: its constructor runs
    the schema migration, which on a missing store *creates* it. That is how a doctor run came to
    leave a new `runs.sqlite3` behind -- `tests/test_doctor.py::test_the_doctor_changes_nothing`
    caught it. This opens SQLite with `mode=ro` so the kernel enforces what the docstring claims,
    and for PostgreSQL only ever issues SELECTs.
    """

    def __init__(self, connection, namespace, placeholder):
        self._connection, self.namespace, self.placeholder = connection, namespace, placeholder

    def _execute(self, cursor, sql, params=()):
        return cursor.execute(sql.replace('?', self.placeholder), params)

    @contextmanager
    def _transaction(self, write=False):
        cursor = self._connection.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def close(self):
        try:
            self._connection.close()
        except Exception:                                       # noqa: BLE001
            pass


def _read_only_store():
    """A read-only handle on the configured run store, or None when there is nothing to read."""
    namespace = os.environ.get('EFORECAST_RUN_STORE_NAMESPACE',
                               os.environ.get('EFORECAST_ALLOCATOR_NAMESPACE', 'default'))
    dsn = os.environ.get('EFORECAST_RUN_STORE_DSN') or os.environ.get('EFORECAST_ALLOCATOR_DSN')
    backend = os.environ.get('EFORECAST_RUN_STORE', 'postgres' if dsn else 'sqlite')
    if backend == 'postgres':
        if not dsn:
            return None
        import psycopg2
        return _ReadOnlyStore(psycopg2.connect(dsn, connect_timeout=3), namespace, '%s')
    if not _run_store_present():
        return None
    import sqlite3
    state = Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local' / 'state'))
    path = os.environ.get('EFORECAST_RUN_STORE_PATH', state / 'eforecast' / 'runs.sqlite3')
    connection = sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True, timeout=3)
    return _ReadOnlyStore(connection, namespace, '?')


def _redacted_url(url):
    """``scheme://host:port/path`` -- never the userinfo, which is a credential."""
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(url)
        host = parts.hostname or '?'
        port = f':{parts.port}' if parts.port else ''
        return f'{parts.scheme}://{host}{port}{parts.path or ""}'
    except ValueError:
        return 'a URL that could not be parsed'


def _scrape_health(port, timeout):
    """Whether the exposition answers, and how many series it offers. Bounded, read-only."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',
                                    timeout=min(timeout, 5.0)) as answer:
            body = answer.read().decode('utf-8', 'replace')
    except (urllib.error.URLError, OSError, ValueError) as error:
        return False, f'nothing is answering on 127.0.0.1:{port} ({str(error)[:70]})'
    series = [line for line in body.splitlines() if line and not line.startswith('#')]
    down = [line for line in series if line.startswith('eforecast_source_up') and
            line.rsplit(' ', 1)[1] == '0']
    note = f'; {len(down)} source(s) reporting unreadable' if down else ''
    return True, f'answering on 127.0.0.1:{port} with {len(series)} series{note}'


def report(*, case=None, artifact_root=None, timeout=3.0, now=None, client_url=None) -> dict[str, Any]:
    """Every check, one verdict, and the readiness of a named case when one is given.

    A check that raises becomes an ``attention`` finding naming the failure, never a silent
    pass and never a crash: a doctor that dies is the one tool that must not.

    ``client_url`` switches to the browser-client role (A14-04): only what reaching the lab
    depends on, and an explicit statement that nothing local was checked.
    """
    if client_url:
        findings = client(client_url, timeout=timeout)
        payload = {'checked_at': readiness._moment(now), 'case': None, 'findings': findings,
                   'role': 'browser client', 'lab': client_url.rstrip('/'),
                   'interpreter': None}
        payload['verdict'] = readiness.verdict(findings)
        payload['counts'] = {severity: sum(1 for f in findings if f['severity'] == severity)
                             for severity in (BLOCKING, ATTENTION, OK)}
        return payload
    document, case_note = _case_document(case)
    findings: list[dict[str, Any]] = []
    if case_note:
        findings.append(case_note)
    steps = (('workspace', lambda: workspace()),
             ('interpreter', lambda: interpreter()),
             ('families', lambda: families(document)),
             ('services', lambda: services(timeout)),
             ('model_cache', lambda: model_cache()),
             ('artifacts', lambda: artifacts(artifact_root or _case_models(document))),
             ('image', lambda: image()),
             ('model_runtimes', lambda: model_runtimes(document)),
             ('image_coherence', lambda: image_coherence(document)),
             ('monitoring', lambda: monitoring(timeout)),
             ('protection', lambda: protection()),
             ('topology', lambda: topology()))
    for name, step in steps:
        try:
            findings.extend(step())
        except Exception as error:                              # noqa: BLE001
            findings.append(_finding(
                name, ATTENTION, f'the {name} check',
                f'the check itself failed ({type(error).__name__}: {str(error)[:110]})',
                'this part of the installation is unknown, which is not the same as healthy',
                'Ask the maintainer to run the doctor with the configured interpreter and '
                'report this message.'))
    payload = {'checked_at': readiness._moment(now), 'case': case, 'findings': findings,
               'interpreter': os.environ.get('EFORECAST_PYTHON') or sys.executable}
    if document is not None:
        # The same module the UI calls, so the two cannot disagree about one case.
        payload['readiness'] = readiness.report(document, case=case, siblings=_siblings())
    # The verdict and the counts cover everything the report shows, installation and case
    # alike: a count that excluded the case findings would contradict the list beside it.
    everything = findings + (payload.get('readiness', {}).get('findings') or [])
    payload['verdict'] = readiness.verdict(everything)
    payload['counts'] = {severity: sum(1 for f in everything if f['severity'] == severity)
                         for severity in (BLOCKING, ATTENTION, OK)}
    return payload


def _siblings():
    """Other cases' model folders, from this installation's own workspace, or ``None``."""
    from eforecast.config.service import Workspace
    try:
        return readiness.case_model_paths(Workspace())
    except Exception:                                           # noqa: BLE001
        return None


def _case_document(case):
    if not case:
        return None, None
    from eforecast.config.service import WorkspaceNotConfigured, Workspace
    try:
        scope = Workspace()
        path = scope.case_path(case) if hasattr(scope, 'case_path') else scope.root / 'cases' / case
    except WorkspaceNotConfigured:
        return None, _finding('case', ATTENTION, f'the case {case}',
                              'the workspace is not configured, so the case cannot be found',
                              'the case-specific checks below were not performed',
                              'Fix the workspace finding, then re-check.', COLLEAGUE)
    except Exception as error:                                  # noqa: BLE001
        return None, _finding('case', ATTENTION, f'the case {case}',
                              f'it could not be located ({type(error).__name__})',
                              'the case-specific checks were not performed',
                              'Check the case name, or run without --case.', COLLEAGUE)
    try:
        from eforecast.config.config_files import load_document
        document = load_document(path)
    except Exception as error:                                  # noqa: BLE001
        return None, _finding('case', ATTENTION, f'the case {case}',
                              f'it could not be read ({type(error).__name__}: {str(error)[:90]})',
                              'the case-specific checks were not performed',
                              f'Check that {path} holds a readable case.', COLLEAGUE)
    if not document:
        return None, _finding('case', ATTENTION, f'the case {case}',
                              f'nothing was found at {path}',
                              'the case-specific checks were not performed',
                              'Check the case name with the case list in the UI.', COLLEAGUE)
    return document, None


def _case_models(document):
    """The case's own model root, so --case also checks that case's retained artifacts."""
    if document is None:
        return None
    static_data = readiness.resolved(document)
    return (static_data or {}).get('path_model')


# ---------------------------------------------------------------------------- printing

HEADLINE = {READY: 'Everything checked is working.',
            NEEDS_ATTENTION: 'Working, with things that need attention.',
            CANNOT_START: 'Something must be fixed before work can run.'}
MARK = {OK: 'ok  ', ATTENTION: 'WARN', BLOCKING: 'STOP'}


def render(payload, *, verbose=False) -> str:
    lines = [f'eforecast doctor - {payload["checked_at"]}']
    if payload.get('role'):
        # A14-04: say which role was reported on, so a browser-client result is never read
        # as a verdict about a local installation.
        lines += [f'role       : {payload["role"]}', f'lab        : {payload.get("lab")}']
    else:
        lines.append(f'interpreter: {payload["interpreter"]}')
    if payload.get('case'):
        lines.append(f'case       : {payload["case"]}')
    lines += ['', HEADLINE[payload['verdict']], '']
    groups = [(BLOCKING, 'Must be fixed'), (ATTENTION, 'Needs attention')]
    if verbose:
        groups.append((OK, 'Checked and fine'))
    everything = payload['findings'] + (payload.get('readiness', {}).get('findings') or [])
    for severity, title in groups:
        chosen = [f for f in everything if f['severity'] == severity]
        if not chosen:
            continue
        lines.append(f'{title}:')
        for finding in chosen:
            who = finding.get('who')
            lines.append(f'  [{MARK[severity]}] {finding["check"]}: {finding["finding"]}')
            if severity != OK:
                lines.append(f'         so: {finding["consequence"]}')
                lines.append(f'         do: {finding["remedy"]}'
                             + (f'   ({who})' if who else ''))
        lines.append('')
    counts = payload['counts']
    lines.append(f'{counts[OK]} checks passed, {counts[ATTENTION]} need attention, '
                 f'{counts[BLOCKING]} must be fixed.')
    if payload['verdict'] != READY:
        lines.append('If a run already failed, see docs/troubleshooting.md.')
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='eforecast doctor', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', help='also check one case and its retained models')
    parser.add_argument('--artifacts', help='a model root whose retained artifacts to sample')
    parser.add_argument('--json', action='store_true', help='machine-readable output')
    parser.add_argument('--verbose', action='store_true', help='also list the checks that passed')
    parser.add_argument('--timeout', type=float, default=3.0, help='seconds to wait for the web UI')
    parser.add_argument('--client', metavar='URL',
                        help='check only that this machine can reach the lab at URL, for a '
                             'colleague who uses the browser and runs nothing locally')
    parser.add_argument('--snapshot', action='store_true',
                        help='also record this reading where the web UI and new runs read it '
                             '(the only thing this command ever writes). With --case it records '
                             "that case's reading, not the installation-wide one")
    args = parser.parse_args(argv)
    try:
        payload = report(case=args.case, artifact_root=args.artifacts, timeout=args.timeout,
                         client_url=args.client)
    except Exception as error:                                  # noqa: BLE001
        print(f'the doctor could not run: {type(error).__name__}: {error}', file=sys.stderr)
        return 3
    if args.snapshot:
        # The default stays read-only: this command's whole value is that it changes nothing.
        # With the flag it writes one file, so a cron entry or a maintainer can keep the
        # reading the page and new runs use from going stale.
        from eforecast.config import health
        try:
            # Scoped like the reading itself: a `--case` run answers a different question, so
            # it must not overwrite the installation-wide reading the header and new runs use.
            recorded = health.write(health.record_of(payload, case=args.case), case=args.case)
            print(f'reading recorded in {recorded}', file=sys.stderr)
        except OSError as error:
            print(f'the reading could not be recorded: {error}', file=sys.stderr)
    print(json.dumps(payload, indent=1, sort_keys=True, default=str) if args.json
          else render(payload, verbose=args.verbose))
    return EXIT[payload['verdict']]


if __name__ == '__main__':
    raise SystemExit(main())
