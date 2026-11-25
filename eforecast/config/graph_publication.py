# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Sorted site ownership and recoverable multi-site graph publication."""
from contextlib import contextmanager, ExitStack
import json
import logging
import os
from pathlib import Path
import shutil
import uuid

from eforecast.common_utils import artifact, file_stamps
from eforecast.common_utils.graph_bundle import POINTER, GENERATIONS, bundle_root, identity
from eforecast.folder_lock import folder_lock as file_critical_section
from eforecast.training.command_line.resource_reporting import ResourceUnavailable
from .executors import atomic_json
from . import run_reporting

log = logging.getLogger(__name__)
WRITER = 'site_graph_writer.json'


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read(path):
    try:
        value = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'invalid graph ownership/publication record {path}')
    return value


def rollback(journal_path):
    """Called with every recorded resource lock held, after the worker has exited.

    A committed filesystem publication wins even if its acknowledgement was lost.
    No rollback can replace a successor's directory or pointer.
    """
    journal_path = Path(journal_path)
    record = read(journal_path)
    if record is None:
        return
    if Path(record['commit']).exists():
        return
    for move in reversed(record['moves']):
        destination, staged, backup = map(Path, (move['destination'], move['staged'], move['backup']))
        if destination.exists():
            manifest = read(destination / artifact.MANIFEST_FILE) or {}
            ours = (manifest.get('graph_publication') or {}).get('id') == record['id']
            if backup.exists() and not ours:
                raise ResourceUnavailable(f'graph rollback cannot replace another writer at {destination}')
            if ours:
                if staged.exists():
                    shutil.rmtree(staged)
                os.replace(destination, staged)
        if backup.exists():
            os.replace(backup, destination)
        sync_directory(destination.parent)
    pointer = Path(record['out']) / POINTER
    current = read(pointer)
    if current and current.get('generation') == record['id']:
        if record['old_pointer'] is None:
            pointer.unlink()
        else:
            atomic_json(pointer, record['old_pointer'])
    record['state'] = 'rolled_back'
    atomic_json(journal_path, record)


class Ownership:
    def __init__(self, paths):
        self.paths = sorted({str(Path(p).resolve()) for p in paths})
        self.id, self.open = uuid.uuid4().hex, False
        self.stages = {}

    def check(self):
        run_reporting.check_cancelled()
        if not self.open:
            raise ResourceUnavailable('site-graph ownership is no longer held')
        for path in self.paths:
            value = read(Path(path) / WRITER)
            stage = read(Path(path) / 'writer.json')
            if not value or value.get('id') != self.id or identity(stage) != self.stages[path]:
                raise ResourceUnavailable('stale site-graph owner cannot publish')

    def journal(self, path):
        self.check()
        for resource in self.paths:
            value = read(Path(resource) / WRITER)
            value['journal'] = str(path)
            atomic_json(Path(resource) / WRITER, value)


@contextmanager
def ownership(paths):
    """Use the existing stage writer locks, not an independent lock namespace.

    An interrupted publication can mention sites omitted by the next command.
    Expand to its complete lock set before rolling it back; reread under locks.
    """
    resources = {str(Path(p).resolve()) for p in paths}
    for _ in range(8):
        journals = set()
        expanded = set(resources)
        for path in resources:
            previous = read(Path(path) / WRITER) or {}
            if previous.get('journal'):
                journal = read(previous['journal'])
                if journal:
                    journals.add(previous['journal'])
                    expanded.update(journal['resources'])
        resources = expanded
        with ExitStack() as locks:
            for path in sorted(resources):
                locks.enter_context(file_critical_section(path, timeout=2))
            extra = set()
            for path in resources:
                previous = read(Path(path) / WRITER) or {}
                if previous.get('journal'):
                    journal = read(previous['journal'])
                    if journal:
                        journals.add(previous['journal'])
                        extra.update(journal['resources'])
            if not extra.issubset(resources):
                resources.update(extra)
                continue
            for journal in sorted(journals):
                rollback(journal)
            held = Ownership(resources)
            for path in held.paths:
                held.stages[path] = identity(read(Path(path) / 'writer.json'))
                previous = read(Path(path) / WRITER) or {}
                atomic_json(Path(path) / WRITER, {'id': held.id, 'fence': int(previous.get('fence', 0)) + 1,
                                                'stage_generation': held.stages[path], 'journal': None})
            held.open = True
            try:
                yield held
            finally:
                held.open = False
            return
    raise ResourceUnavailable('site-graph resource set changed repeatedly; retry after the other publication settles')


def publish(workers, handle, result, out, destinations, held, inputs_check):
    """Copy and hash the worker's output, rename all exports, then commit.

    Until the one commit exists, modern exports are explicitly unready. Ordinary
    errors roll back immediately; a crashed host is repaired by the next owner.
    Old bundle files and old export directories remain available as history.
    """
    root = workers.root(handle)
    _, spec = workers.executor._spec(handle)
    context = spec['compute']['context']
    launch = context['launch_id']
    out = Path(out).resolve()
    generation = out / GENERATIONS / launch
    commit = generation / 'commit.json'
    journal_path = generation / 'publication.json'
    if commit.exists():
        return {'model': str(generation / 'bundle'), 'commit': str(commit), 'exports': list(destinations.values())}
    held.check()
    inputs_check()
    generation.mkdir(parents=True, exist_ok=True)
    if generation.is_symlink():
        raise ValueError('unsafe graph generation')
    artifacts, moves, stamps = {}, [], {}
    staged_roots = []
    try:
        targets = {'bundle': generation / 'bundle'}
        for site, destination in destinations.items():
            destination = Path(destination).resolve()
            staged = destination.with_name('.graph-stage-' + launch)
            backup = destination.with_name('.graph-backup-' + launch)
            targets['exports/' + site] = staged
            moves.append({'destination': str(destination), 'staged': str(staged), 'backup': str(backup)})
            staged_roots.append(staged)
        for source_name, target in targets.items():
            source = root / 'artifacts' / source_name
            if not source.is_dir():
                raise ValueError(f'worker omitted graph output {source_name}')
            if target.is_symlink():
                raise ValueError('unsafe graph staging directory')
            target.mkdir(parents=True, exist_ok=True)
            for file in source.iterdir():
                name = file.relative_to(root).as_posix()
                if not file.is_file() or file.is_symlink() or name not in result['files']:
                    raise ValueError('unverified graph publication payload')
                copied = target / file.name
                if copied.is_symlink():
                    raise ValueError('unsafe graph staging file')
                shutil.copyfile(file, copied)
                if artifact.digest(copied) != result['files'][name]['sha256']:
                    raise ValueError('graph copy differs from accepted worker bytes')
                with copied.open('rb') as stream:
                    os.fsync(stream.fileno())
            logical = str(generation / 'bundle') if source_name == 'bundle' else str(destinations[source_name.split('/')[1]])
            manifest = read(target / artifact.MANIFEST_FILE)
            manifest['graph_publication'] = {'id': launch, 'commit': str(commit), 'logical_path': logical}
            atomic_json(target / artifact.MANIFEST_FILE, manifest)
            valid, reason = artifact.verify_digests(str(target), allow_staged=True)
            if not valid:
                raise ValueError(f'incomplete graph publication: {reason}')
            artifacts[logical] = identity(manifest)
            for path in target.iterdir():
                # A stamp, so that a sync client marking the file it just uploaded is not
                # taken for a payload that changed (``file_stamps``).
                stamps[path] = file_stamps.stamp(path, follow=False)
        held.check()
        inputs_check()
        record = {'version': 1, 'id': launch, 'out': str(out), 'commit': str(commit),
                  'resources': held.paths, 'old_pointer': read(out / POINTER), 'moves': moves, 'state': 'prepared'}
        atomic_json(journal_path, record)
        held.journal(journal_path)
        fallback = str(bundle_root(out).resolve())
        body = {'version': 1, 'id': launch, 'artifacts': artifacts,
                'producer': context, 'dependencies': result['value']['dependencies']}
        # These directories are pending (commit absent), so no reader can take
        # them for finished.
        for move in moves:
            held.check()
            destination, staged, backup = map(Path, (move['destination'], move['staged'], move['backup']))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                raise ValueError('graph backup collision; previous history was not replaced')
            if destination.exists():
                os.replace(destination, backup)
                sync_directory(destination.parent)
            os.replace(staged, destination)
            stamps.update({destination / path.name: expected for path, expected in list(stamps.items()) if path.parent == staged})
            stamps = {path: expected for path, expected in stamps.items() if path.parent != staged}
            sync_directory(destination.parent)
        for path in [generation, *generation.parents]:
            sync_directory(path)
        held.check()
        inputs_check()
        for path, expected in stamps.items():
            if not file_stamps.same(expected, file_stamps.stamp(path, follow=False)):
                raise ValueError('graph payload changed before publication')
        atomic_json(out / POINTER, {'version': 1, 'generation': launch, 'fallback': fallback})
        held.check()
        atomic_json(commit, body)  # sole group completion decision; always last
        record['state'] = 'committed'
        atomic_json(journal_path, record)
        return {'model': str(generation / 'bundle'), 'commit': str(commit), 'exports': list(destinations.values())}
    except BaseException:
        if journal_path.exists():
            rollback(journal_path)
        raise
