# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Fetch raw provider files again when the ones on disk hold no bytes.

A download that stops partway leaves zero-byte files behind. They are names, not weather: the
reader gets nothing out of them, and the first readable step after one cannot be de-accumulated,
because the difference against the step before it then spans two intervals instead of one. The
archive they came from still has them, so the first thing to try is fetching them again.

Only a name with no bytes is ever replaced. A file that has content is never overwritten, and the
download lands in a scratch directory beside the day folders first, so an interrupted refill
cannot leave a partial payload where a reader would find it. Nothing here raises: a refill that
cannot run leaves a gap to report, which is not a reason to stop a run.

``EFORECAST_NWP_REFILL=off`` keeps a run from reaching the network at all.
``EFORECAST_NWP_REMOTE`` names the rclone remote, ``gdrive:ECMWF`` by default.
"""
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from eforecast.nwp_extraction import sources

MODE = 'EFORECAST_NWP_REFILL'
MODES = ('drive', 'off')
REMOTE = 'EFORECAST_NWP_REMOTE'
#: Seconds for the whole refill, not per day. This runs inside the stage's admission lock, so a
#: provider that has stopped answering must cost a bounded wait and then a reported gap.
BUDGET = 600
log = logging.getLogger(__name__)


def _rclone(day, names, into, budget):
    """The owner's own Drive, one named file at a time rather than the whole day folder."""
    listing = into / '.files-from'
    listing.write_text('\n'.join(names) + '\n')
    remote = f"{os.environ.get(REMOTE) or 'gdrive:ECMWF'}/{day}"
    done = subprocess.run(['rclone', 'copy', remote, str(into), '--files-from', str(listing),
                           '--stats', '0'], capture_output=True, text=True, timeout=budget)
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout).strip()[:300])
    return f'rclone {remote}'


def _gdown(day, names, into, budget):
    """The public shared folder the extractor already downloads from, whole day folders only."""
    import pandas as pd
    from eforecast.nwp_extraction.ecmwf_extractor import DownLoader
    log.info('refill %s: no file-by-file route, so the whole day folder is fetched', day)
    DownLoader(date=pd.Timestamp(day), path_nwp=str(into)).download()
    return 'gdown public folder'


def routes_of():
    """One route, not a chain. ``rclone`` fetches the named files and can be given a deadline;
    the public-folder route fetches whole days and cannot, so it stands in only when there is no
    ``rclone`` to ask -- not whenever ``rclone`` says no to one file."""
    return (_rclone,) if shutil.which('rclone') else (_gdown,)


def restore(files, *, mode=None, routes=None):
    """Fetch every zero-byte file in ``files`` again; report what is still empty afterwards."""
    mode = (mode or os.environ.get(MODE) or 'drive').strip().lower()
    _, empty = sources.readable(files)
    report = {'empty': list(empty), 'refilled': [], 'unfilled': list(empty), 'source': None}
    if not empty:
        return report
    if mode != 'drive':
        log.warning('%d raw NWP file(s) hold no bytes and %s=%s, so none were fetched again',
                    len(empty), MODE, mode)
        return report
    wanted = {}
    for path in empty:
        day, inside = sources.day_of(path)
        if day is None:
            log.warning('refill: %s is not inside a %%Y%%m%%d source folder, so it cannot be placed', path)
            continue
        wanted.setdefault(day, {})[str(inside)] = Path(path)
    chosen = routes if routes is not None else routes_of()
    deadline = time.monotonic() + BUDGET
    for day, targets in sorted(wanted.items()):
        budget = deadline - time.monotonic()
        if budget <= 0:
            log.warning('refill: %ds spent without finishing; %s is left for a later run',
                        BUDGET, day.name)
            break
        report['source'] = _restore_day(day, targets, chosen, budget, report) or report['source']
    report['unfilled'] = [path for path in empty if path not in set(report['refilled'])]
    if report['refilled']:
        log.info('refill: %d of %d empty raw NWP file(s) fetched again from %s',
                 len(report['refilled']), len(empty), report['source'])
    return report


def _restore_day(day, targets, chosen, budget, report):
    scratch = None
    try:
        # Beside the day folders, never inside one: a reader walks the day folder, and the
        # replacement has to land with one rename on the same filesystem.
        scratch = Path(tempfile.mkdtemp(dir=day.parent, prefix='.nwp-refill-'))
    except OSError as error:
        log.warning('refill %s: no writable scratch beside the day folder (%r)', day.name, error)
        return None
    try:
        used = None
        for route in chosen:
            try:
                used = route(day.name, sorted(targets), scratch, budget)
                break
            except Exception as error:                      # noqa: BLE001 -- try the next route
                log.warning('refill %s: %s failed (%r)', day.name, getattr(route, '__name__', route), error)
        if used is None:
            return None
        for inside, target in sorted(targets.items()):
            found = next((p for p in sorted(scratch.rglob(Path(inside).name))
                          if p.is_file() and p.stat().st_size), None)
            if found is None:
                continue
            if target.is_file() and target.stat().st_size:
                continue            # refilled by something else while this ran; leave its bytes alone
            try:
                os.replace(found, target)
            except OSError as error:
                log.warning('refill %s: could not place %s (%r)', day.name, inside, error)
                continue
            report['refilled'].append(str(target))
        return used
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def before_inputs(static):
    """Refill what holds no bytes before the extract stage records the roster it will read.

    A refilled file is a real input; a file left empty is a recorded gap. Doing this first is what
    keeps the two apart. Never raises -- failing to look is not a reason to stop a run.
    """
    try:
        from eforecast.config.stage_artifacts import raw_nwp_sources
        return restore(raw_nwp_sources(static))
    except Exception as error:                              # noqa: BLE001
        log.warning('could not check the raw NWP sources for empty files (%r)', error)
        return None
