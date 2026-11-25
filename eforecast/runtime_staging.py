# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Where an attempt stages its inputs, and how it reuses what is already there -- A16-03.

Three properties the task asks for, and one it implies.

**Portable.** An attempt's directory is named from its *identity* and nothing else, so no
owner-specific absolute path is baked into it. The base comes from the environment or the
workspace; everything below is `case/run/attempt`, which is the same on any machine.

**Logical identity is separate from where the bytes sit.** The contract carries
`dataset_id`, `partition_id`, `view_id` and `scaler_id`, and an array descriptor carries a
name, a digest and a shape -- never a path. So the same inputs staged under two different
roots produce the *same request fingerprint*: moving a job cannot change its split or scaler
identity. That is asserted in the tests rather than assumed.

**Staged by hash, reused when complete.** An artifact already present with the digest the
payload would have is not written again. The manifest is published last and only after every
artifact has been re-read and re-digested, so a manifest exists only when the set behind it
is complete.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

MANIFEST = 'manifest.json'
VERSION = 1

#: Identity fields are used as directory names, so they are sanitised rather than trusted.
_SAFE = re.compile(r'[^A-Za-z0-9_.-]')


class StagingError(RuntimeError):
    """The attempt's staging area cannot be established."""


def base_root(root=None):
    """The base every attempt directory sits under.

    `EFORECAST_RUNTIME_SCRATCH` first, then the workspace's run-state, then a temporary
    directory. The workspace is preferred over the home directory because a lab machine's
    workspace is the disk with room on it.
    """
    if root is not None:
        return Path(root)
    named = os.environ.get('EFORECAST_RUNTIME_SCRATCH')
    if named:
        return Path(named)
    workspace = os.environ.get('EFORECAST_WORKSPACE')
    if workspace:
        return Path(workspace) / '.run-state' / 'scratch'
    import tempfile
    return Path(tempfile.gettempdir()) / 'eforecast-scratch'


def _part(value, fallback):
    text = _SAFE.sub('-', str(value or '')).strip('-.')
    return text[:64] or fallback


def attempt_paths(identity, *, root=None, create=True):
    """The directories one attempt owns: its root, its inputs and its output.

    Named only from the identity, so the same attempt resolves to the same relative place on
    any machine and nothing outside this attempt can be addressed from it.
    """
    if not isinstance(identity, dict):
        raise StagingError('identity: expected the contract identity of an attempt')
    parts = [_part(identity.get(key), fallback) for key, fallback in
             (('case_id', 'case'), ('run_id', 'run'), ('attempt_id', 'attempt'))]
    from eforecast.runtime_contract import fingerprint
    # Sanitisation/truncation is not an identity: a/b and a-b otherwise share bytes,
    # and a successor fence must never overwrite its predecessor's staging directory.
    parts[-1] += '-' + fingerprint(identity)[:24]
    attempt = base_root(root).joinpath(*parts)
    paths = {'root': attempt, 'input': attempt / 'input', 'output': attempt / 'output',
             'scratch': attempt / 'scratch'}
    if create:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    return paths


# ----------------------------------------------------------------- the manifest

def read_manifest(task_dir):
    """What a previous staging of this directory completed, or an empty record."""
    path = Path(task_dir) / MANIFEST
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict) or document.get('version') != VERSION or not isinstance(document.get('artifacts'), dict):
        return {}
    return document['artifacts']


def write_manifest(task_dir, artifacts):
    """Publish the manifest, after re-reading every artifact it names.

    Published last and only on success, so a manifest never describes a set that is not
    there: the check is a re-read of the bytes on disk, not a memory of having written them.
    """
    from eforecast.runtime_adapters import transport

    verified = {}
    for name, descriptor in sorted(artifacts.items()):
        path = transport.array_path(task_dir, descriptor['artifact_id'])
        if not os.path.exists(path):
            raise StagingError(f'{name}: {descriptor["artifact_id"]} is missing, so the '
                               f'manifest would describe a set that is not there')
        digest = transport._digest(path)
        if digest != descriptor['sha256']:
            raise StagingError(f'{name}: {descriptor["artifact_id"]} on disk does not match '
                               f'what was staged; the manifest is not published')
        verified[name] = dict(descriptor)
    transport.write_document(task_dir, MANIFEST,
                             {'version': VERSION, 'artifacts': verified})
    return verified


def reusable(task_dir, name, descriptor):
    """Whether a staged artifact is already present, complete and the one wanted.

    Both halves matter: the manifest says a previous staging *finished*, and the digest says
    the bytes are still the ones it finished with.
    """
    from eforecast.runtime_adapters import transport

    recorded = read_manifest(task_dir).get(name)
    if not recorded or recorded.get('sha256') != descriptor.get('sha256'):
        return False
    path = transport.array_path(task_dir, descriptor['artifact_id'])
    try:
        return os.path.exists(path) and transport._digest(path) == descriptor['sha256']
    except OSError:
        return False
