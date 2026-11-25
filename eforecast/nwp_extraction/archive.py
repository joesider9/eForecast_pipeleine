# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Which folder of a group's weather archive holds the daily files of one case.

A group of cases shares its extracted weather: one file a day, written once and read by every
case of the group. A file holds the variables it was extracted with, and nothing asked which: a
day whose file existed was "already there". That held while a group had one kind of case. A group
with a wind case and a photovoltaic park found each other's files -- wind speed and pressure
where radiation and cloud cover were needed -- and failed three stages later, on a variable that
"is missing from the weather files".

So every set of variables has a folder of its own, named after them:
``nwp/Cloud+Flux+High_Cloud+Low_Cloud+Medium_Cloud``. Two cases that ask for the same variables
share it, as before. The sets are kept apart rather than merged into one file because a file's
steps depend on its variables: the first step after a missing provider file is dropped whole when
an accumulated field (radiation) is among them, so one case joining a group would change what
another one reads.

Files that an earlier version wrote directly into ``nwp`` stay where they are, and serve the
cases that need exactly their variables. Which variables those are is read from one of the
files, once, and kept beside them in ``variables.json``.
"""
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

MARKER = 'variables.json'
SAMPLES = 3                 # daily files tried before the folder is taken as it always was
_read = {}                  # (file, its mtime) -> the variables it holds


def name(variables):
    """The folder of a set of variables: their names in order, joined."""
    return '+'.join(sorted(set(variables)))


def variables_of(nwps):
    """The variables one daily file holds, whichever format it is in."""
    if nwps.get('variables'):
        return sorted(set(nwps['variables']))
    from eforecast.nwp_extraction.nwp_pickles import NON_RUN_KEYS
    for run in sorted(key for key in nwps if key not in NON_RUN_KEYS):
        for values in nwps[run].values():
            return sorted(key for key in values if key not in ('lat', 'long'))
    return []


def resident(folder, *, record=False):
    """The variables of the daily files written directly into ``folder``.

    ``None`` when it holds no daily file, and ``False`` when it holds some and none of the first
    few can be read: nothing can be said about them then, and they are left to whoever asks.
    """
    folder = Path(folder)
    try:
        return sorted(set(json.loads((folder / MARKER).read_text())['variables']))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        with os.scandir(folder) as entries:
            files = sorted(entry.path for entry in entries
                           if entry.is_file() and entry.name.endswith('.pickle') and '_' in entry.name)
    except OSError:
        return None
    if not files:
        return None
    for path in files[:SAMPLES]:
        try:
            key = (path, os.stat(path).st_mtime_ns)
            if key not in _read:
                import joblib
                _read[key] = variables_of(joblib.load(path))
            held = _read[key]
        except Exception as error:                              # noqa: BLE001 -- a file that cannot be read says nothing
            log.warning('weather archive %s: %s cannot be read (%r)', folder, os.path.basename(path), error)
            continue
        if record:
            try:
                temporary = folder / f'.{MARKER}.{os.getpid()}.tmp'
                temporary.write_text(json.dumps({'variables': held, 'read_from': os.path.basename(path)}) + '\n')
                os.replace(temporary, folder / MARKER)
            except OSError:
                pass                                            # read again next time; nothing depends on the note
        return held
    return False


def folder(group_nwp, variables, *, record=False):
    """The folder for the daily files of ``variables`` inside a group's ``nwp`` folder.

    ``record`` lets the reading of an earlier version's files be noted beside them; a dry run
    passes False and changes nothing.
    """
    wanted = sorted(set(variables))
    if not wanted:
        return str(group_nwp)                                   # a case without weather keeps nothing there
    held = resident(group_nwp, record=record)
    if held is False or held == wanted:
        return str(group_nwp)
    return os.path.join(str(group_nwp), name(wanted))
