# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The explicit maintenance conversion of extracted weather files, in a worker of its own."""
import hashlib
import logging
from pathlib import Path
import shutil
import uuid

from eforecast.folder_lock import folder_lock
from eforecast.training.command_line.resource_reporting import ResourceUnavailable
from .compute_workers import ComputeWorkers, failure_reason
from . import run_reporting

log = logging.getLogger(__name__)


def convert(folder, max_step_hours=60, compress=3, dry_run=False, *, workers=None):
    from eforecast.nwp_extraction.convert_nwp_pickles import NAME
    from eforecast.nwp_extraction.conversion_worker import observation
    folder = Path(folder).absolute()
    if not folder.exists():
        raise FileNotFoundError(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    workers = workers or ComputeWorkers()
    with folder_lock(folder / '.stage-artifacts', timeout=2):
        sources, symlinks = {}, 0
        for path in sorted(folder.iterdir()):
            if NAME.match(path.name):
                if path.is_symlink():
                    symlinks += 1
                else:
                    sources[path.name] = observation(path)
        if not sources:
            log.info('weather conversion: no regular input files; %d symlinks skipped', symlinks)
            return 0, symlinks, 0, 0  # nothing to read, so no worker is started
        uid = uuid.uuid4().hex
        scope = hashlib.sha256(str(folder).encode()).hexdigest()
        request = workers.request('weather:' + uid, 'convert:' + scope, uid)
        payload = {'folder': str(folder), 'sources': sources, 'symlinks': symlinks,
                   'max_step_hours': max_step_hours, 'compress': compress, 'dry_run': dry_run}
        copies = []
        destination = folder / ('.conversion-stage-' + uid)
        run_reporting.check_cancelled()
        handle = workers.prepare(request, 'eforecast.nwp_extraction.conversion_worker:execute', payload,
                                 execution={'device': 'cpu', 'threads': 1})
        try:
            result = workers.wait(handle, timeout=None)
            if result['state'] != 'success':
                raise ResourceUnavailable(f'weather conversion {failure_reason(result)}; source files preserved')
            root, files = workers.root(handle), result.get('files') or {}
            if files:
                destination.mkdir()
            for name in files:
                filename = Path(name).name
                if filename not in sources or name != 'artifacts/' + filename:
                    raise ValueError('foreign weather conversion output')
                temporary = destination / filename
                copies.append((temporary, folder / filename))
                shutil.copyfile(root / name, temporary)
                from eforecast.common_utils.lineage import digest_file
                if temporary.stat().st_size != files[name]['bytes'] or digest_file(str(temporary)) != files[name]['sha256']:
                    raise ValueError('weather copy changed')
            # Check all sources before the first swap. This utility retains its
            # existing per-file atomic/idempotent semantics, not a case receipt.
            run_reporting.check_cancelled()
            for name, expected in sources.items():
                path = folder / name
                if path.is_symlink() or observation(path) != expected:
                    raise ValueError('weather source changed before publication')
            import os
            for temporary, target in copies:
                os.replace(temporary, target)
            counts = tuple(result['value']['counts'])
            log.info('weather conversion: converted=%d skipped=%d bytes=%d -> %d', *counts)
            workers.discard_result(handle)
            return counts
        except BaseException:
            workers.cancel(handle, force=True)
            raise
        finally:
            for temporary, _ in copies:
                temporary.unlink(missing_ok=True)
            if destination.exists():
                destination.rmdir()
