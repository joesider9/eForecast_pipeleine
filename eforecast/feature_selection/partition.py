# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Partition records belong to selector payloads, independently of clustering's record.

A record (``feature_selectors.partition.json``) names the partition a payload was fitted against
and a digest of the payload itself, so a selector cannot be reused under a partition it never saw.
The payload and its record are two files and cannot be replaced in one step, so a record also
carries ``replaces``: the digest and the record of the payload it is replacing.  An interrupted
publication therefore leaves a record that describes a payload which never landed, and the payload
that is on disk is recognised as the previous one and keeps its own identity -- the feature stage
stays resumable, which it must be, because it is the expensive stage.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import uuid

import joblib

from eforecast.datasets.data_preprocessing.data_split import partition_file, partition_fingerprint

log = logging.getLogger(__name__)
SELECTORS_FILE = 'feature_selectors.pickle'
SELECTOR_PARTITION_FILE = 'feature_selectors.partition.json'
IDENTITY_KEYS = ('split_policy', 'masks_sha256', 'cluster_masks_sha256')
REBUILD_HINT = ('Rebuild the feature stage with --recreate and retrain dependent models in a new model '
                'version, or restore the original partition and policy.')


def selector_partition_identity(static_data, global_masks, cluster_masks):
    identity = partition_fingerprint(static_data, global_masks) if global_masks is not None else {
        'split_policy': static_data.get('split_policy', 'frozen'), 'masks_sha256': None}
    identity['cluster_masks_sha256'] = partition_fingerprint(static_data, cluster_masks)['masks_sha256']
    return identity


def _payload_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _publish_record(path, record):
    # A partial JSON file must not erase the provenance of a cached selector.
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=1)
            stream.write('\n')
        if path.exists():
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_record(path):
    try:
        with path.open(encoding='utf-8') as stream:
            stored = json.load(stream)
        if not isinstance(stored, dict) or stored.get('version') != 1 or not stored.get('selector_sha256'):
            raise ValueError('invalid selector partition record')
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot read the selector partition record {path}: {error}. {REBUILD_HINT}') from error
    return stored


def _record_of_the_payload(stored, payload, path):
    """The record that describes what is actually on disk, or a refusal.

    ``stored`` describes the payload of the last publication that *started*.  If the digests differ,
    the publication did not finish: the payload is still the one it was replacing, which ``replaces``
    identifies, and that payload keeps its own record.  ``None`` means the payload on disk predates
    the records.  Anything else is a payload replaced behind the records' back, and is refused.
    """
    digest = _payload_digest(payload)
    if stored['selector_sha256'] == digest:
        return stored
    previous = stored.get('replaces') or {}
    if previous.get('selector_sha256') != digest:
        raise ValueError(f'The selector payload {payload} does not match its partition record {path}, '
                         f'and is not the payload that record was replacing either: it was changed outside '
                         f'the feature stage. {REBUILD_HINT}')
    log.warning('%s records a publication that did not finish (an interrupted feature stage): the payload '
                'on disk is the one it was replacing, so it keeps its own provenance and the stage can be '
                'rerun without --recreate; the next selector published rewrites the record', path)
    return previous.get('record')


def check_selector_partition(cluster_path, identity):
    """Validate the selector's own record; legacy payloads stay readable with a warning."""
    folder = Path(cluster_path)
    path = folder / SELECTOR_PARTITION_FILE
    payload = folder / SELECTORS_FILE
    if not path.exists():
        log.warning('%s has no selector partition record; legacy feature selectors are reused on trust', payload)
        return None
    stored = _record_of_the_payload(_read_record(path), payload, path)
    if stored is None:
        log.warning('%s has no record of its own; legacy feature selectors are reused on trust', payload)
        return None
    differences = {key: (stored.get(key), identity[key]) for key in IDENTITY_KEYS
                   if stored.get(key) is not None and stored[key] != identity[key]}
    if differences:
        changes = ', '.join(f'{key} was {was!r}, now {now!r}' for key, (was, now) in differences.items())
        raise ValueError(f'The feature selectors in {folder} belong to another partition: {changes} '
                         f'(recorded in {path}). {REBUILD_HINT}')
    if stored.get('provenance') != 'fitted':
        log.warning('%s was adopted from legacy artifacts on trust; its original fitting provenance '
                    'was not verified', payload)
    return stored


def _replaced(folder, payload):
    """``{'selector_sha256', 'record'}`` of the payload about to be replaced, or ``None``.

    Without this a publication interrupted between the two files would leave the previous payload
    unidentifiable, and the only way forward would be to refit the whole stage.  The nested record
    is stripped of its own ``replaces`` so a long-lived model folder does not accumulate a chain.
    """
    if not payload.exists():
        return None
    path = folder / SELECTOR_PARTITION_FILE
    record = None
    if path.exists():
        try:
            record = {key: value for key, value in _read_record(path).items() if key != 'replaces'}
        except ValueError as error:
            log.warning('cannot carry over the record being replaced in %s (%r)', path, error)
    return {'selector_sha256': _payload_digest(payload), 'record': record}


def save_selectors(cluster_path, selectors, *, identity=None, provenance='fitted'):
    """Publish a selector payload and its own identity without changing the pickle format."""
    folder = Path(cluster_path)
    payload = folder / SELECTORS_FILE
    temporary = folder / f'.feature_selectors.{uuid.uuid4().hex}.tmp'
    try:
        # Exclusive creation uses the caller's umask, like the original joblib.dump: a worker must
        # not turn readable model artifacts into owner-only files when publishing a replacement.
        with temporary.open('xb') as stream:
            joblib.dump(selectors, stream)
        if payload.exists():
            temporary.chmod(stat.S_IMODE(payload.stat().st_mode))
        if identity is not None:
            record = dict(identity, version=1, provenance=provenance,
                          selector_sha256=_payload_digest(temporary),
                          replaces=_replaced(folder, payload))
            # Publish the record first: interruption between the two replacements is detected by
            # the checksum, instead of mistaking a newly written payload for an untracked legacy one.
            # What the record replaces is named in it, so the interruption is recoverable.
            _publish_record(folder / SELECTOR_PARTITION_FILE, record)
        os.replace(temporary, payload)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def forget_selector_partition(cluster_path):
    """Drop the record of a payload that was replaced outside the feature stage.

    A payload with no record is reused on trust with a warning; a payload contradicting a record is
    refused.  Anything that writes selectors without an identity of this model must therefore leave
    no record behind.
    """
    path = Path(cluster_path) / SELECTOR_PARTITION_FILE
    if path.exists():
        path.unlink()
        log.warning('removed the selector partition record %s: the payload beside it was not published by '
                    'the feature stage, so it is reused on trust', path)


def preserve_global_selector_partition(static_data, global_path):
    """Pin untracked global selectors before clustering replaces their masks and parent record.

    This adds metadata only. Unknown legacy policy is kept unknown, and the record explicitly says
    it was adopted on trust; the old selector and any trained weights are left intact.
    """
    folder = Path(global_path)
    payload, path = folder / SELECTORS_FILE, folder / SELECTOR_PARTITION_FILE
    if not payload.exists() or path.exists():
        return
    parent = {}
    try:
        with open(partition_file(static_data), encoding='utf-8') as stream:
            loaded = json.load(stream)
        if isinstance(loaded, dict):
            parent = loaded
    except (OSError, ValueError):
        pass
    policy = parent.get('split_policy')
    legacy_static = dict(static_data, split_policy=policy)
    try:
        old_masks = joblib.load(folder / 'cv_mask.pickle')
        identity = selector_partition_identity(legacy_static, old_masks, old_masks)
    except Exception as error:  # noqa: BLE001 -- older model trees may not retain their masks
        log.warning('cannot recover the old global masks for %s (%r); preserving whatever the old '
                    'partition record knows', payload, error)
        identity = {'split_policy': policy, 'masks_sha256': parent.get('masks_sha256'),
                    'cluster_masks_sha256': parent.get('masks_sha256')}
    record = dict(identity, version=1, provenance='legacy_adopted', selector_sha256=_payload_digest(payload))
    _publish_record(path, record)
    log.warning('preserved the previous partition of legacy global selectors in %s before rebuilding '
                'clustering; the original fitting provenance remains unverified', path)
