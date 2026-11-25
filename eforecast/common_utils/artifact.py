# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Whether an artifact on disk is finished work, and what produced it.

A trial folder used to be admitted to prediction and selection on the presence of
``net_weights.pickle``. The network writes weights while it is preparing and loading, before
training has finished, so a trial that crashed, was cancelled or failed could still be predicted
with and selected -- which is the discovery half of F15/AR07.

A manifest is the artifact saying what it is: the state it reached, what produced it, and the size
and digest of each file it published. It is written last and atomically. A complete result can
still supply legacy completion evidence if manifest publication was interrupted; that evidence
does not establish payload integrity. Attempt-local publication is a separate A07 increment.

Three answers, not two:

``verified``   a manifest is present, its version is known, its state is usable, and every file it
               lists is present with the recorded size (digests require ``verify_digests``);
``legacy``     no manifest. Readers may use existing completion evidence with a warning; absence
               alone does not establish the artifact's age;
``rejected``   a manifest that says the attempt did not finish, or whose files no longer match it.
"""
import hashlib
import json
import logging
import os
import re
import socket
import uuid

log = logging.getLogger(__name__)

MANIFEST_FILE = 'artifact_manifest.json'
VERSION = 1
USABLE_STATES = ('complete', 'pruned')

# What each producer must have published for the artifact to be usable, and what it may publish.
# A kind that is not listed here (``trial``, written before the producers were named) keeps the
# older rule: the manifest is judged on the files it happens to name.
KINDS = {
    'gpu_trial': {'required': ('trial_result.json', 'net_weights.pickle', 'net_weights.pt', 'parameters.pickle'),
                  'optional': ('results_eval.pickle', 'results.csv', 'error.txt', 'training_scope.json')},
    'cpu_trial': {'required': ('trial_result.json', 'net_weights.pickle'),
                  'optional': ('training_scope.json', 'error.txt')},
    'graph_export': {'required': ('net_weights.pickle', 'predictions_eval.pickle'), 'optional': ()},
    'graph_bundle': {'required': ('site_graph.pt', 'meta.json'), 'optional': ('history.csv',)},
}


def payload_rules(kind, state):
    """``(required, optional)`` for a producer. A failure publishes its record; the rest is optional."""
    rule = KINDS.get(kind)
    if rule is None:
        return (), ()
    if state in USABLE_STATES:
        return rule['required'], rule['optional']
    failed = tuple(name for name in rule['required'] if name.endswith('result.json'))
    return failed, tuple(name for name in rule['required'] if name not in failed) + rule['optional']


def worker_identity(path=None, run_id=None, task=None, attempt=None):
    """Who produced this artifact: the run, its task, the trial folder and the attempt.

    The orchestrator passes the first three in the environment, to a local worker and to a
    container alike; a producer inside the parent process passes them directly; a worker started
    by hand simply records what it knows.
    """
    trial = os.path.basename(os.path.normpath(path)) if path else None
    attempt = os.environ.get('EFORECAST_ATTEMPT') if attempt is None else attempt
    identity = {'run_id': run_id or os.environ.get('EFORECAST_RUN_ID'),
                'task': task or os.environ.get('EFORECAST_TASK_ID'),
                'trial': trial, 'attempt': int(attempt) if str(attempt).isdigit() else attempt}
    identity['attempt_id'] = ':'.join(str(identity[key]) for key in ('run_id', 'task', 'trial', 'attempt'))
    return identity


def digest(path, blocks=8 * 1024 * 1024):
    sha = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(blocks), b''):
            sha.update(block)
    return sha.hexdigest()


def manifest_file(path):
    return os.path.join(path, MANIFEST_FILE)


def describe(path, names):
    """Describe every named payload; a missing required file is a publication error."""
    out = {}
    for name in names:
        full = os.path.join(path, name)
        out[name] = {'bytes': os.path.getsize(full), 'sha256': digest(full)}
    return out


def publish(path, kind, state, files=(), optional_files=(), **fields):
    """Write the manifest of a finished artifact, atomically and last.

    Call it only when the files it describes are already on disk: the manifest is the claim that
    they are complete, so writing it earlier would make an interrupted publication look finished.
    Missing required payloads raise ``FileNotFoundError``. An ``OSError`` while reading payloads
    or writing/replacing the manifest is logged and returns ``None``; temporary-file cleanup is
    best effort and does not replace that publication diagnostic with another filesystem error.
    """
    required, optional = payload_rules(kind, state)
    files, optional_files = tuple(files) or required, tuple(optional_files) or optional
    names = tuple(files) + tuple(name for name in optional_files if os.path.isfile(os.path.join(path, name)))
    missing = [name for name in files if not os.path.isfile(os.path.join(path, name))]
    if missing:
        # A publisher naming a file that is not there is not publishing finished work.
        raise FileNotFoundError(f"{', '.join(missing)} named for publication but not in {path}")
    temporary = os.path.join(path, f'.{MANIFEST_FILE}.{uuid.uuid4().hex}.tmp')
    try:
        # Reading the payload is inside the try: a file that cannot be read now, or disappears while
        # it is read, leaves the artifact without a completion claim rather than failing its caller.
        payload = {'version': VERSION, 'kind': kind, 'state': state,
                   'host': socket.gethostname(), 'pid': os.getpid(),
                   'files': describe(path, names), **fields}
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=1, default=str)
        os.replace(temporary, manifest_file(path))
    except OSError as error:
        log.warning('cannot publish the manifest of %s (%r): the artifact keeps no completion claim', path, error)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            log.warning('cannot remove temporary manifest %s (%r)', temporary, cleanup_error)
        return None
    return payload


def read(path):
    from .graph_bundle import bundle_root
    try:
        path = bundle_root(path)
    except (OSError, ValueError, TypeError):
        return {'version': None, 'state': 'unreadable'}
    return _read_manifest(path)


def _read_manifest(path):
    file = manifest_file(path)
    if not os.path.isfile(file):
        return None
    try:
        with open(file, encoding='utf-8') as stream:
            payload = json.load(stream)
    except (OSError, ValueError) as error:
        log.warning('cannot read the manifest of %s (%r)', path, error)
        return {'version': None, 'state': 'unreadable'}
    return payload if isinstance(payload, dict) else {'version': None, 'state': 'unreadable'}


def inspect(path, usable_states=USABLE_STATES, *, allow_staged=False):
    """``(verdict, reason)`` with verdict ``verified`` / ``legacy`` / ``rejected``."""
    from .graph_bundle import bundle_root
    try:
        path = bundle_root(path)
    except (OSError, ValueError, TypeError):
        return 'rejected', 'site-graph pointer cannot be resolved'
    return _inspect(path, _read_manifest(path), usable_states, allow_staged=allow_staged)


def _inspect(path, payload, usable_states=USABLE_STATES, *, allow_staged=False):
    if payload is None:
        return 'legacy', 'no manifest: completion needs legacy evidence'
    if type(payload.get('version')) is not int or payload['version'] != VERSION:
        return 'rejected', f"manifest version {payload.get('version')!r} is not {VERSION}"
    if not isinstance(payload.get('kind'), str) or not payload['kind']:
        return 'rejected', 'manifest kind must be a nonempty string'
    # Written by a version that recorded an allocation with each trial; still read, so that
    # trials trained then stay usable.
    allocation = payload.get('allocation')
    if 'allocation' in payload:
        if not isinstance(allocation, dict):
            return 'rejected', 'manifest allocation must be a mapping'
        for field in ('id', 'run_id', 'task_id', 'attempt_id', 'host', 'boot_id'):
            if not isinstance(allocation.get(field), str) or not allocation[field].strip():
                return 'rejected', f'allocation {field} must be a nonempty identity'
        if (type(allocation.get('fence')) is not int or allocation['fence'] <= 0
                or type(payload.get('fence')) is not int or payload['fence'] != allocation['fence']
                or payload.get('run_id') != allocation['run_id']):
            return 'rejected', 'manifest allocation fence/run identity differs'
    if payload.get('publication_phase') == 'staged' and not allow_staged:
        return 'rejected', 'attempt is staged, not published to its trial'
    if not allow_staged:
        from .graph_bundle import publication_error
        reason = publication_error(payload)
        if reason:
            return 'rejected', reason
    state = payload.get('state')
    if state not in usable_states:
        return 'rejected', f'the attempt ended as {state!r}'
    files = payload.get('files')
    if not isinstance(files, dict) or not files:
        return 'rejected', 'manifest files must be a nonempty mapping'
    for name in payload_rules(payload['kind'], state)[0]:
        if name not in files:   # the producer is known, so its bundle is known too
            return 'rejected', f"the manifest of a {payload['kind']} does not name {name}"
    for name, recorded in files.items():
        if not isinstance(name, str) or not name or os.path.isabs(name) or '..' in name.split('/') or '\x00' in name:
            return 'rejected', f'invalid payload name {name!r}'
        if not isinstance(recorded, dict):
            return 'rejected', f'{name} metadata must be a mapping'
        if type(recorded.get('bytes')) is not int or recorded['bytes'] < 0:
            return 'rejected', f'{name} bytes must be a nonnegative integer'
        if not isinstance(recorded.get('sha256'), str) or not re.fullmatch(r'[0-9a-fA-F]{64}', recorded['sha256']):
            return 'rejected', f'{name} sha256 must be a 64-character hexadecimal digest'
        full = os.path.join(path, name)
        try:
            if not os.path.isfile(full):
                return 'rejected', f'{name} is named in the manifest and missing'
            size = os.path.getsize(full)
        except OSError as error:
            return 'rejected', f'{name} cannot be inspected ({error})'
        if size != recorded['bytes']:
            return 'rejected', f'{name} is {size} bytes, the manifest says {recorded["bytes"]}'
    return 'verified', f'state {state}'


def verify_digests(path, *, allow_staged=False):
    """The expensive half of :func:`inspect`: every digest, for a release check rather than a tick."""
    from .graph_bundle import bundle_root
    try:
        path = bundle_root(path)
    except (OSError, ValueError, TypeError):
        return False, 'site-graph pointer cannot be resolved'
    payload = _read_manifest(path)
    verdict, reason = _inspect(path, payload, allow_staged=allow_staged)
    if verdict != 'verified':
        return False, reason
    for name, recorded in payload['files'].items():
        full = os.path.join(path, name)
        try:
            actual = digest(full)
        except OSError as error:
            return False, f'{name} cannot be read ({error})'
        if actual != recorded['sha256'].lower():
            return False, f'{name} does not match its recorded digest'
    return True, 'every file matches'
