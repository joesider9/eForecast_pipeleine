# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A15-04: the artifacts a forecast cannot be rebuilt without, and a restore that proves it.

A15-02 made rollback a pointer operation, which is right and which quietly assumes the previous
bundle is still on disk. This task owns that assumption.

The shape of it:

* a **generation** is a directory holding copies plus `generation.json`, and the manifest is
  written **last**, after every digest has been verified. A generation without its manifest is
  an interrupted copy, not a backup, and nothing will restore from one;
* the previous complete generation is never touched while a new one is being written, so an
  interruption costs the new copy and not the last good one;
* the destination must be on a **different device** from the source. Another directory on the
  same disk protects against deleting a file and against nothing else, which is the failure
  this task exists to survive;
* **the acceptance is the restore.** A backup nobody has restored is a belief.

Completion comes from A07/A09 (`artifact`, `model_bundle`), not from a second opinion: a
partially written bundle must not enter a backup looking complete.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

VERSION = 1
GENERATION = 'generation.json'
PARTIAL = '.incomplete'
#: Read in chunks: a model checkpoint is not something to hold in memory twice.
BLOCK = 1024 * 1024


class BackupError(ValueError):
    """The backup or restore cannot proceed, and nothing has been changed."""


class IncompleteGeneration(BackupError):
    """This generation was never finished, so it is not a thing to restore from."""


def digest_of(path, blocks=BLOCK):
    out = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(blocks):
            out.update(chunk)
    return out.hexdigest()


# ---------------------------------------------------------------------- the destination


def destination_findings(source, destination):
    """Whether this destination protects against losing the source disk.

    `st_dev` rejects a second directory on the same filesystem. Different device numbers
    alone do not prove different physical disks: two partitions can share a failure domain.
    An operator still needs to verify the destination is physically independent.
    """
    source, destination = Path(source), Path(destination)
    probe = destination
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        same = os.stat(source).st_dev == os.stat(probe).st_dev
    except OSError as error:
        return [{'state': 'fail', 'detail': f'the destination cannot be stated ({error})',
                 'remedy': 'mount it before backing up; a backup to a path that is not there '
                           'is not a backup'}]
    if same:
        return [{'state': 'fail',
                 'detail': f'{destination} is on the same device as {source}',
                 'remedy': 'use a separate physical disk or an independent destination. '
                           'Another directory on the same disk does not survive the disk.'}]
    return [{'state': 'ok',
             'detail': f'{destination} is on a different device from {source}'}]


# ---------------------------------------------------------------------- what is protected


def inventory(root, *, approval=None, case=None, static_data=None):
    """The files this case cannot be rebuilt without, each with its digest.

    Built from the bundle manifest, because that is already the agreed statement of what a
    forecast needs (A15-01) -- not from walking a directory and hoping. Anything the bundle
    does not name is not in the protected set, and that is a decision a reader can check.
    """
    from eforecast import model_bundle
    root = Path(root)
    manifest = model_bundle.read(root)
    absent = model_bundle.missing(manifest, root)
    if absent:
        raise IncompleteGeneration(
            'this bundle is not complete, so it must not enter a backup looking as if it were: '
            + '; '.join(f'{item["component"]}: {item["finding"]}' for item in absent[:3]))

    if approval is not None:
        from copy import deepcopy
        from eforecast.bundle_serving import identity, legacy_identity, reference_digest
        bound = deepcopy(manifest)
        if approval.get('identity_version', 1) == 2:
            bound['identity_version'] = 2
            reference = model_bundle.read_fixture(root)
            if not reference:
                raise IncompleteGeneration('approved bundle has lost its reference')
            bound['reference_sha256'] = reference_digest(reference)
            actual = identity(bound)
        else:
            actual = legacy_identity(bound)
        if actual != approval.get('bundle'):
            raise IncompleteGeneration('these artifacts are not the bundle the supplied approval names')

    files = []
    for item in manifest['components']:
        path = root / item['path']
        files.append({'path': item['path'], 'kind': item['kind'], 'bytes': path.stat().st_size,
                      'sha256': digest_of(path), 'runtime': item['runtime'],
                      'checkpoint': item.get('checkpoint')})
    for name in (model_bundle.FIXTURE_FILE, 'bundle.json'):
        path = root / name
        if path.is_file():
            files.append({'path': name, 'kind': 'manifest', 'bytes': path.stat().st_size,
                          'sha256': digest_of(path), 'runtime': None, 'checkpoint': None})
    return {
        'case': case or manifest.get('case'),
        'qualification': 'qualified-contract' if manifest.get('identity_version') == 2 and manifest.get('reference_sha256') else 'legacy-unqualified',
        'bundle': manifest,
        'files': sorted(files, key=lambda item: item['path']),
        'runtimes': sorted({item['runtime'] for item in files if item['runtime']}),
        'checkpoints': sorted({item['checkpoint'] for item in files if item['checkpoint']}),
        # The control state this artifact set belongs to. A restore that brings back artifacts
        # without it cannot say which approval they were, and one that brings back the pointer
        # without the artifacts is a silent hole. They travel together.
        'approval': approval,
    }


def _secret_shaped(files):
    """Credential material has its own recovery procedure (A16-10) and never rides along here."""
    from eforecast.config import secrets_source
    return [item['path'] for item in files
            if secrets_source.secret_shaped_name(Path(item['path']))]


# ---------------------------------------------------------------------- backing up


def back_up(root, destination, *, case=None, approval=None, by='', now=None, label=None):
    """Copy the protected set into a new generation, verify it, then publish the manifest.

    Returns the generation manifest. The previous complete generation is untouched throughout:
    if this is interrupted, the directory stays `.incomplete` and the last good one still is.
    """
    source = Path(root)
    destination = Path(destination)
    findings = destination_findings(source, destination)
    bad = [item for item in findings if item['state'] != 'ok']
    if bad:
        raise BackupError(f'{bad[0]["detail"]} -- {bad[0]["remedy"]}')

    held = inventory(source, approval=approval, case=case)
    leaking = _secret_shaped(held['files'])
    if leaking:
        raise BackupError(
            f'these look like credential material and must not enter a model backup: '
            f'{", ".join(leaking)}. Credentials have their own recovery procedure (A16-10).')

    moment = time.time() if now is None else now
    name = label or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(moment))
    case_root = destination / (held['case'] or 'unnamed')
    working = case_root / (name + PARTIAL)
    final = case_root / name
    if final.exists():
        raise BackupError(f'generation {name} already exists at {final}')
    working.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    copied = 0
    for item in held['files']:
        target = working / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / item['path'], target)
        # Verified from the copy, not from the original: the point is that *this* file arrived.
        written = digest_of(target)
        if written != item['sha256']:
            raise BackupError(
                f'{item["path"]} did not copy intact ({written[:12]} != {item["sha256"][:12]}); '
                f'the generation stays incomplete and the previous one is untouched')
        copied += item['bytes']

    manifest = {
        'version': VERSION, 'generation': name, 'case': held['case'],
        'created_at': moment, 'by': by,
        'seconds': round(time.monotonic() - started, 3), 'bytes': copied,
        'files': held['files'], 'runtimes': held['runtimes'],
        'checkpoints': held['checkpoints'], 'approval': held['approval'],
        'source': str(source),
    }
    # Published last, and only now: until this file exists the directory is an interrupted copy.
    (working / GENERATION).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(working, final)
    return manifest


def generations(destination, case):
    """Every generation for a case, newest last, each marked complete or not."""
    case_root = Path(destination) / case
    if not case_root.is_dir():
        return []
    found = []
    for path in sorted(case_root.iterdir()):
        if not path.is_dir():
            continue
        complete = (path / GENERATION).is_file()
        entry = {'generation': path.name.removesuffix(PARTIAL), 'path': str(path),
                 'complete': complete and not path.name.endswith(PARTIAL)}
        if entry['complete']:
            # What the copy recorded about itself, for the listing: how large it is.
            try:
                manifest = json.loads((path / GENERATION).read_text())
                entry.update(bytes=manifest.get('bytes'), files=len(manifest.get('files') or []),
                             created_at=manifest.get('created_at'))
            except (OSError, ValueError, TypeError):
                pass                                    # `verify` says what is wrong with it
        found.append(entry)
    return found


def last_complete(destination, case):
    """The newest generation that was actually finished, or None."""
    done = [item for item in generations(destination, case) if item['complete']]
    return done[-1] if done else None


def read_generation(path):
    try:
        manifest = json.loads((Path(path) / GENERATION).read_text())
    except FileNotFoundError:
        raise IncompleteGeneration(
            f'{path} has no {GENERATION}: it is an interrupted copy, not a backup') from None
    except ValueError as error:
        raise BackupError(f'{path}/{GENERATION} is not readable JSON: {error}') from None
    if not isinstance(manifest, dict) or manifest.get('version') != VERSION:
        raise BackupError(f'{path} is not a version {VERSION} generation')
    from eforecast.bundle_serving import safe_path
    try:
        if not isinstance(manifest.get('files'), list) or not manifest['files']:
            raise ValueError('generation contains no file inventory')
        for item in manifest['files']:
            safe_path(path, item['path'])
            if not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64:
                raise ValueError('invalid generation digest')
    except (KeyError, TypeError, ValueError) as error:
        raise BackupError(f'invalid generation inventory: {error}') from error
    return manifest


def verify(path):
    """Check a generation against its own manifest. Returns findings, never raises for content."""
    path = Path(path)
    try:
        manifest = read_generation(path)
    except BackupError as error:
        return [{'state': 'fail', 'file': GENERATION, 'detail': str(error)}]
    findings = []
    for item in manifest['files']:
        target = path / item['path']
        if not target.is_file():
            findings.append({'state': 'fail', 'file': item['path'],
                             'detail': 'in the manifest but not in the generation'})
            continue
        found = digest_of(target)
        if found != item['sha256']:
            findings.append({'state': 'fail', 'file': item['path'],
                             'detail': f'digest differs ({found[:12]} != {item["sha256"][:12]})'})
    if not findings:
        findings.append({'state': 'ok', 'file': '',
                         'detail': f'{len(manifest["files"])} file(s) match the manifest'})
    return findings


# ---------------------------------------------------------------------- restoring


def restore(path, into, *, verify_first=True):
    """Copy a generation back, verifying every file. Returns a report with the time it took.

    The restore writes only into `into`. Nothing about the original location is touched, read
    or required -- which is the property the drill then checks by making it unavailable.
    """
    path, into = Path(path), Path(into)
    manifest = read_generation(path)
    if verify_first:
        broken = [item for item in verify(path) if item['state'] != 'ok']
        if broken:
            raise BackupError(
                f'this generation does not match its own manifest, so restoring it would '
                f'spread the damage: {broken[0]["file"]}: {broken[0]["detail"]}')
    into.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    restored = []
    for item in manifest['files']:
        from eforecast.bundle_serving import safe_path
        target = safe_path(into, item['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path / item['path'], target)
        found = digest_of(target)
        if found != item['sha256']:
            raise BackupError(f'{item["path"]} did not restore intact; the restore is incomplete')
        restored.append(item['path'])
    return {'generation': manifest['generation'], 'case': manifest['case'],
            'into': str(into), 'files': restored, 'bytes': manifest['bytes'],
            'seconds': round(time.monotonic() - started, 3),
            'approval': manifest.get('approval'), 'runtimes': manifest.get('runtimes') or [],
            'restored_at': time.time()}


def agreement(report, *, approval):
    """Do the restored artifacts and the restored control state agree?

    A run record pointing at an artifact the restore did not bring back is a silent hole; an
    artifact with no record cannot be attributed. Both are reported rather than resolved,
    because resetting a ledger to make a restore look clean is the one thing never to do.
    """
    findings = []
    recorded = report.get('approval') or {}
    if not recorded:
        findings.append({'state': 'fail', 'detail':
                         'this generation records no approval, so the restored artifacts '
                         'cannot be attributed to a version anyone approved',
                         'remedy': 'back up through the approval, not by copying a directory'})
    if approval is None:
        findings.append({'state': 'fail', 'detail':
                         'the control state has no approval for this case, but artifacts were '
                         'restored for one',
                         'remedy': 'restore the control state too; artifacts with no record '
                                   'cannot be attributed and must be fenced before dispatch'})
    elif recorded and approval.get('bundle') != recorded.get('bundle'):
        findings.append({'state': 'fail', 'detail':
                         f'the control state points at bundle '
                         f'{str(approval.get("bundle"))[:12]} and the restore brought back '
                         f'{str(recorded.get("bundle"))[:12]}',
                         'remedy': 'restore the generation the control state names, or roll the '
                                   'pointer to what was restored -- never edit the ledger to '
                                   'make them agree'})
    if not findings:
        findings.append({'state': 'ok', 'detail':
                         f'the restored artifacts are the bundle the control state approves '
                         f'(revision {approval.get("revision")})'})
    return findings


RESTORES = 'restores.json'
#: How long evidence stays meaningful. A backup nobody has restored in two months is a belief
#: again, and a restore nobody has repeated says only that it worked once.
BACKUP_STALE_DAYS = 7
RESTORE_STALE_DAYS = 60


def record_restore(destination, case, report, *, agreement_findings=(), now=None):
    """Keep the evidence that a restore was actually performed, and whether it agreed.

    Written beside the generations rather than in the run store on purpose: the question this
    answers -- "has anyone proved we can get this back?" -- must be answerable when the store
    is the thing that was lost.
    """
    path = Path(destination) / case / RESTORES
    path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if path.is_file():
        try:
            history = json.loads(path.read_text())
        except ValueError:
            history = []
    entry = {'generation': report['generation'], 'at': now or time.time(),
             'seconds': report['seconds'], 'bytes': report['bytes'],
             'files': len(report['files']), 'into': report['into'],
             'agreed': all(item['state'] == 'ok' for item in agreement_findings)
                       if agreement_findings else None}
    history.append(entry)
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(history[-64:], indent=2, sort_keys=True))
    os.replace(temporary, path)
    return entry


def last_verified_restore(destination, case):
    """The most recent restore actually performed for this case, or None."""
    path = Path(destination) / case / RESTORES
    try:
        history = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None
    return history[-1] if history else None


def freshness(destination, case, *, now=None):
    """Last complete backup and last verified restore, **separately**.

    Separately because they fail separately and mean different things: a fresh backup nobody
    has restored is untested, and an old restore of a stale backup proves only that something
    once worked. Reporting one number for both would hide whichever is worse.
    """
    moment = now or time.time()
    out = {'case': case, 'destination': str(destination)}
    complete = last_complete(destination, case)
    if complete is None:
        out['backup'] = {'state': 'fail', 'detail': 'no complete backup generation exists',
                         'remedy': 'take one: a rollback is a pointer operation and assumes '
                                   'the previous bundle is still somewhere'}
    else:
        try:
            manifest = read_generation(complete['path'])
            age = (moment - manifest['created_at']) / 86400.0
            stale = age > BACKUP_STALE_DAYS
            out['backup'] = {
                'state': 'attention' if stale else 'ok',
                'detail': f'generation {manifest["generation"]} from '
                          f'{time.strftime("%Y-%m-%d", time.gmtime(manifest["created_at"]))} '
                          f'({age:.1f} days old, {len(manifest["files"])} files)',
                'remedy': 'take a new generation' if stale else 'Nothing to do.'}
        except BackupError as error:
            out['backup'] = {'state': 'fail', 'detail': str(error),
                             'remedy': 'take a new generation'}
    restore_entry = last_verified_restore(destination, case)
    if restore_entry is None:
        out['restore'] = {'state': 'fail',
                          'detail': 'no restore has ever been performed from this destination',
                          'remedy': 'restore a generation into a scratch directory and validate '
                                    'it; a backup nobody has restored is a belief'}
    else:
        age = (moment - restore_entry['at']) / 86400.0
        stale = age > RESTORE_STALE_DAYS
        out['restore'] = {
            'state': 'attention' if stale else 'ok',
            'detail': f'generation {restore_entry["generation"]} restored '
                      f'{time.strftime("%Y-%m-%d", time.gmtime(restore_entry["at"]))} '
                      f'({age:.1f} days ago, {restore_entry["files"]} files in '
                      f'{restore_entry["seconds"]}s)'
                      + ('' if restore_entry.get('agreed') is not False
                         else '; the control state did NOT agree'),
            'remedy': 'repeat the restore drill' if stale else 'Nothing to do.'}
    return out


def main(argv=None):
    """`eforecast backup` -- copy, check and restore an approved bundle and its dependencies.

        bin/eforecast backup copy    BUNDLE DESTINATION [--case NAME] [--label NAME]
        bin/eforecast backup verify  GENERATION
        bin/eforecast backup restore GENERATION INTO
        bin/eforecast backup list    DESTINATION CASE

    A16-08 asks a maintainer to demonstrate a restore, so it must not require writing Python.
    The destination must be on a different physical device; a restored bundle is replayed
    before it is trusted, which `eforecast approve` does for you.
    """
    import argparse
    import getpass
    import os
    import sys

    parser = argparse.ArgumentParser(prog='eforecast backup',
                                     description=main.__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    copy = commands.add_parser('copy', help='copy a bundle and its dependencies to a destination')
    copy.add_argument('bundle')
    copy.add_argument('destination')
    copy.add_argument('--case', default=None)
    copy.add_argument('--label', default=None)
    check = commands.add_parser('verify', help='check a generation against what it recorded')
    check.add_argument('generation')
    back = commands.add_parser('restore', help='restore a generation into a folder')
    back.add_argument('generation')
    back.add_argument('into')
    listing = commands.add_parser('list', help='the generations a destination holds for a case')
    listing.add_argument('destination')
    listing.add_argument('case')
    arguments = parser.parse_args(argv)
    who = os.environ.get('EFORECAST_USER') or getpass.getuser()

    try:
        if arguments.command == 'copy':
            made = back_up(arguments.bundle, arguments.destination, case=arguments.case,
                           by=who, label=arguments.label)
            print(f"copied {len(made['files'])} file(s), {made['bytes']} bytes -> "
                  f"{arguments.destination}/{made['case']}/{made['generation']}")
            print('Check it before trusting it: bin/eforecast backup verify '
                  f"{arguments.destination}/{made['case']}/{made['generation']}")
            return 0
        if arguments.command == 'verify':
            findings = verify(arguments.generation)
            for item in findings:
                print(f"{item['state']:9s} {item.get('file', '')}  {item.get('detail', '')}")
            return 0 if all(item['state'] == 'ok' for item in findings) else 2
        if arguments.command == 'restore':
            report = restore(arguments.generation, arguments.into)
            print(f"restored {len(report['files'])} file(s), {report['bytes']} bytes in "
                  f"{report['seconds']:.1f}s -> {arguments.into}")
            # Record that it was actually done. `freshness` reports "no restore has ever been
            # performed from this destination" as blocking, and nothing wrote the evidence it
            # reads -- so a maintainer who ran the drill through this command could never clear
            # it, and the one finding that asks for proof was unanswerable.
            destination = Path(arguments.generation).resolve().parent.parent
            try:
                entry = record_restore(destination, report['case'], report)
                print(f"drill recorded: {destination}/{report['case']}/{RESTORES}"
                      f" ({entry['files']} file(s) in {entry['seconds']}s)")
            except OSError as error:
                print(f'the restore worked but could not be recorded: {error}', file=sys.stderr)
            print('Qualify it before use: bin/eforecast approve <case> ' + str(arguments.into))
            return 0
        listed = generations(arguments.destination, arguments.case)
        if not listed:
            print(f'no backup of {arguments.case} under {arguments.destination}')
        for item in listed:
            if not item.get('complete', True):
                size = 'INCOMPLETE: an interrupted copy'
            elif item.get('bytes') is None:
                size = 'size not recorded'
            else:
                size = f"{item['bytes']} bytes in {item.get('files')} file(s)"
            print(f"{str(item.get('generation')):24s} {size}")
        return 0
    except Exception as error:                          # noqa: BLE001 -- a command, not a trace
        print(f'nothing was changed: {type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
