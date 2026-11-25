# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A13-04: what a colleague is told before starting a long run.

Three outcomes, not an evidence taxonomy: **ready**, **needs attention**, **cannot start**.
Each finding names the field it belongs to, what was checked, what was found, what follows from
it, and one concrete thing to do.  The display stays simple; the care is in the checks.

Bounded by construction.  Only the index column of the target file is read, under a row cap; an
NWP archive contributes one directory listing, never a grib message; model availability is a
spec lookup, never an import; nothing is trained, fitted, downloaded or launched.  Reading this
report leaves every source and artifact exactly as it was.

The report carries ``checked_at`` and ``inputs_digest``.  The digest covers the fields the
findings actually depend on, so a caller can tell whether a verdict still describes the document
in front of it: a timestamp alone cannot authorise a stale verdict at submission.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

#: Only the index is read, and only this many rows of it.
MAX_INDEX_ROWS = 500_000

#: Above this share of empty slots the period is not a dataset any more but a scattering of rows.
#: Below it the period is merely shorter than its dates suggest, which is worth saying and not
#: worth refusing: these checks gate nothing by themselves, so a blocking verdict has to be true.
MOSTLY_EMPTY_SHARE = 50.0

#: Which distribution provides each CPU method that needs one.
FAMILY_DISTRIBUTIONS = {'Chronos2': ('chronos', 'chronos-forecasting'),
                        'TimesFM': ('timesfm', 'timesfm'),
                        'TiRex': ('tirex', 'tirex-ts'),
                        'CatBoost': ('catboost', 'catboost'),
                        'lightgbm': ('lightgbm', 'lightgbm')}

#: Reading GRIB needs these, whatever else is installed.
GRIB_MODULES = ('pygrib', 'eccodes')

#: The scikit-learn versions the retained estimators were compared against (A13-03).
QUALIFIED_SKLEARN = ((1, 8), (1, 10))

BLOCKING, ATTENTION, OK = 'blocking', 'attention', 'ok'
CANNOT_START, NEEDS_ATTENTION, READY = 'cannot start', 'needs attention', 'ready'


def _finding(check, field, severity, checked, finding, consequence, remedy):
    return {'check': check, 'field': field, 'severity': severity, 'checked': checked,
            'finding': finding, 'consequence': consequence, 'remedy': remedy}


def _at(document, path, default=None):
    node: Any = document
    for part in path.split('.'):
        if not isinstance(node, dict):
            return default
        node = node.get(part)
    return default if node is None else node


# ---------------------------------------------------------------------------- the target series

def _target_index(path, resolution):
    """The timestamps of the target file, and nothing else.

    ``usecols=[0]`` keeps this one column wide however many variables the file carries, and the
    row cap keeps it bounded however long it is.  A truncated read is reported as truncated
    rather than presented as the whole series.
    """
    import pandas as pd
    frame = pd.read_csv(path, header=0, index_col=0, usecols=[0], parse_dates=True,
                        nrows=MAX_INDEX_ROWS + 1)
    index = pd.DatetimeIndex(frame.index)
    truncated = len(index) > MAX_INDEX_ROWS
    return index[:MAX_INDEX_ROWS], truncated


def _columns(path):
    import pandas as pd
    return list(pd.read_csv(path, header=0, nrows=0).columns)


def _target_findings(document, derived):
    import pandas as pd
    findings = []
    from eforecast.common_utils.resolutions import pandas_freq
    filename = _at(document, 'project.filename')
    try:
        resolution = pandas_freq(_at(document, 'project.ts_resolution', 'H'))
    except ValueError as error:
        # A readiness check reports; it does not crash on a document it was asked to judge.
        return [_finding('target.resolution', 'project.ts_resolution', BLOCKING,
                         'the time step the case runs at',
                         str(error),
                         'no index can be built at that step, so every dataset stage refuses',
                         'Set the resolution to 1min, 15min, H or D.')], None
    if not filename:
        return [_finding('target.file', 'project.filename', BLOCKING,
                         'the main time series file of the case',
                         'no file is configured',
                         'no target can be read, so no dataset and no model can be built',
                         'Choose the csv file holding the measured target series.')], None
    path = Path(str(filename))
    if not path.is_file():
        return [_finding('target.file', 'project.filename', BLOCKING,
                         f'the main time series file {path.name}',
                         f'{path} does not exist or is not a file',
                         'the target cannot be read, so the run stops at its first stage',
                         'Correct the path, or point it at a copy of the series on this machine.')], None
    try:
        index, truncated = _target_index(path, resolution)
        columns = _columns(path)
    except Exception as error:                                  # noqa: BLE001
        return [_finding('target.file', 'project.filename', BLOCKING,
                         f'reading the timestamps of {path.name}',
                         f'{type(error).__name__}: {str(error)[:160]}',
                         'the target cannot be read, so the run stops at its first stage',
                         'Check that the first column holds timestamps and the file is a readable csv.')], None

    source = _at(document, 'data.target_variable.source') or _at(document, 'project.name')
    if source and source not in columns:
        findings.append(_finding(
            'target.column', 'data.target_variable.source', BLOCKING,
            'the column holding the target, among the file\'s columns',
            f'{source!r} is not in {columns[:8]}{"..." if len(columns) > 8 else ""}',
            'the dataset builder refuses the case before any stage runs',
            f'Set the target source to one of the file\'s columns, or rename the column to {source!r}.'))

    if len(index) == 0:
        findings.append(_finding('target.coverage', 'project.filename', BLOCKING,
                                 'how much history the file covers',
                                 'the file has no rows',
                                 'there is nothing to train on',
                                 'Point the case at a series that contains measurements.'))
        return findings, None

    first, last = index.min(), index.max()
    span_days = round((last - first).total_seconds() / 86400.0, 1)
    coverage = {'first': str(first), 'last': str(last), 'rows': int(len(index)),
                'span_days': span_days, 'truncated': bool(truncated)}
    try:        # the rows a model can be trained on: the evaluation period is kept away from training
        moment = pd.Timestamp(_at(document, 'project.evaluation_start'))
        if first <= moment <= last:
            coverage['rows_before_evaluation'] = int((index < moment).sum())
    except Exception:                                           # noqa: BLE001 -- none said, or not a date: judged below
        pass
    findings.append(_finding(
        'target.coverage', 'project.filename', ATTENTION if truncated else OK,
        'the first and last timestamp of the target series',
        f'{len(index)} rows from {first} to {last} ({span_days} days)'
        + (f'; only the first {MAX_INDEX_ROWS} rows were read' if truncated else ''),
        'this is the period a run can learn from and be evaluated on'
        if not truncated else 'the checks below describe the part that was read, not the whole file',
        'Nothing to do.' if not truncated else
        'Open the deeper inspection if the later part of the file also needs checking.'))

    if not index.is_monotonic_increasing:
        findings.append(_finding('target.order', 'project.filename', ATTENTION,
                                 'whether the timestamps increase',
                                 'the timestamps are not in increasing order',
                                 'the import sorts and rounds them, so rows may not line up as written',
                                 'Sort the file by timestamp so what is stored matches what is used.'))

    duplicates = int(index.duplicated().sum())
    if duplicates:
        findings.append(_finding(
            'target.duplicates', 'project.ts_resolution', ATTENTION,
            'repeated timestamps at the case resolution',
            f'{duplicates} of {len(index)} timestamps repeat',
            'the import keeps the last row of each repeated timestamp and discards the others',
            'Remove the repeated rows, or confirm that keeping the last one is what you want.'))

    # The slot a row falls in, as the import assigns it (``fix_timeseries_dates``).
    slots = index.floor(resolution)
    off_grid = int((slots != index).sum())
    if off_grid:
        shared = len(slots) - len(slots.unique())
        findings.append(_finding(
            'target.sub_resolution', 'project.ts_resolution', OK,
            f'rows stamped between the slots of the {resolution} grid',
            f'{off_grid} of {len(index)} rows are not on a {resolution} slot, and {shared} of them '
            f'share a slot with another row',
            'the import averages the rows of a slot, so such a value is the mean over the slot and '
            'not the reading stamped at its start',
            f'Nothing to do if the mean over each {resolution} slot is what the case means. Set '
            'Ts resolution to the resolution of the source to keep the finer rows as slots of '
            'their own.'))
    expected = pd.date_range(slots.min(), slots.max(), freq=resolution)
    absent = expected.difference(slots.unique())
    dst_omitted = 0
    if absent.tz is None and len(absent):
        try:
            # A naive repeated hour cannot identify either autumn occurrence;
            # spring's skipped hour has no instant at all. Leaving these slots
            # absent is legitimate, unlike losing an ordinary observation.
            unresolved = absent.tz_localize(_at(document, 'project.local_timezone', 'UTC'),
                                           ambiguous='NaT', nonexistent='NaT').isna()
            dst_omitted = int(unresolved.sum())
            absent = absent[~unresolved]
        except (ValueError, KeyError, TypeError):
            pass  # _timezone_findings supplies the invalid-zone diagnostic
    missing = len(absent)
    if dst_omitted:
        findings.append(_finding(
            'target.dst_omitted', 'project.local_timezone', OK,
            'omitted daylight-saving slots in the source timezone',
            f'{dst_omitted} ambiguous or nonexistent clock slots have no observation',
            'the import leaves these slots empty, preserving lag spacing; targets needing '
            'these observations are excluded when datasets are rebuilt',
            'Do not fill these slots with guessed measurements. Existing datasets and models '
            'are unchanged; use Rebuild to apply source changes to a new run.'))
    if missing > 0:
        share = round(100.0 * missing / max(len(expected), 1), 2)
        empty = share > MOSTLY_EMPTY_SHARE
        findings.append(_finding(
            'target.gaps', 'project.filename', BLOCKING if empty else ATTENTION,
            f'missing slots on the {resolution} grid between the first and last timestamp',
            f'{missing} of {len(expected)} slots have no row ({share}% of the period)',
            'most of the period has no data, so training and evaluation would run on a fraction '
            'of what the dates suggest' if empty else
            f'those rows become empty and are dropped later, so {round(100.0 - share, 2)}% of the '
            'period is what a run can learn from, and a gap longer than the lags removes the rows '
            'around it as well',
            'Fill the gaps in the source, or move the start date past the sparse part.'))
    elif not dst_omitted and len(expected) == len(slots.unique()):
        findings.append(_finding('target.resolution', 'project.ts_resolution', OK,
                                 f'that the series is a complete {resolution} grid',
                                 f'{len(expected)} slots, none missing',
                                 'the declared resolution matches the file',
                                 'Nothing to do.'))

    findings.extend(_timezone_findings(document, index))
    findings.extend(_lag_findings(document, derived, coverage, index, resolution))
    return findings, coverage


def _timezone_findings(document, index):
    # Source timestamps are in local_timezone, just as the dataset/NWP readers
    # interpret them. site_timezone describes the physical site, not the file.
    field = 'project.local_timezone'
    zone = _at(document, field, 'UTC')
    try:
        if index.tz is not None:
            index.tz_convert(zone)  # preserve recorded instants; do not localize twice
        else:
            index.tz_localize(zone, nonexistent='raise', ambiguous='raise')
        return [_finding('target.dst', field, OK,
                         f'the source timestamps in Local timezone ({zone})',
                         'the recorded timezone/offset identifies each instant'
                         if index.tz is not None else 'every timestamp exists once in that source timezone',
                         'no daylight-saving ambiguity was found in the timestamps checked',
                         'Nothing to do.')]
    except Exception as error:                                  # noqa: BLE001
        text = str(error)
        ambiguous, nonexistent = 'ambiguous' in text.lower(), 'nonexistent' in text.lower()
        kind = ('are ambiguous when the clocks go back' if ambiguous
                else 'do not exist when the clocks go forward' if nonexistent
                else 'cannot be placed in that zone')
        # A clock that changes twice a year produces a repeated hour and a missing one, and a
        # source kept in that clock inherits both. The import leaves those slots empty on purpose
        # (target.dst_omitted says so), so this is reported rather than flagged: there is nothing
        # to repair, only a zone to confirm. A timestamp that fits the zone nowhere is different.
        return [_finding(
            'target.dst', field, OK if ambiguous or nonexistent else ATTENTION,
            f'the source timestamps in Local timezone ({zone})',
            f'some timestamps {kind} ({text[:110]})',
            'weather alignment may shift, omit or fail on these source times; '
            'this check does not establish that any retained dataset was changed',
            'Check Local timezone against the clock used by the source file, not the site location. '
            + ('CET in this runtime changes to UTC+2 in summer. If the source stays at UTC+1 all year, '
               'the fixed-offset name is Etc/GMT-1; do not relabel it without confirming that convention. '
               if zone == 'CET' else '')
            + 'If the source clock changes seasonally, repeated hours need explicit offsets or '
            'another documented way to distinguish them. Review affected datasets before changing the source timezone.')]


def _lag_findings(document, derived, coverage, index, resolution):
    import pandas as pd
    findings = []
    past = [int(v) for v in (_at(document, 'data.global_past_lags') or [])]
    future = [int(v) for v in (_at(document, 'data.global_future_lags') or [])]
    steps = int((derived or {}).get('steps') or 0)
    needed_before = max([abs(v) for v in past if v < 0] + [0])
    needed_after = max([abs(v) for v in future if v > 0] + [steps] + [0])
    available = len(index.round(resolution).unique())
    if needed_before + needed_after >= available:
        findings.append(_finding(
            'target.lags', 'data.global_past_lags', BLOCKING,
            'that the configured lags fit inside the available history',
            f'the lags need {needed_before} steps before and {needed_after} after each sample, '
            f'but only {available} steps exist in total',
            'no sample would have a complete input window, so the dataset would be empty',
            'Shorten the lags, or extend the series.'))
    elif needed_before or needed_after:
        lost = needed_before + needed_after
        findings.append(_finding(
            'target.lags', 'data.global_past_lags', OK,
            'that the configured lags fit inside the available history',
            f'{needed_before} steps before and {needed_after} after each sample; '
            f'{available - lost} of {available} rows can form a complete sample',
            f'the first {needed_before} and last {needed_after} rows cannot be used, which is normal',
            'Nothing to do.'))
    start = _at(document, 'project.evaluation_start')
    if start:
        try:
            moment = pd.Timestamp(start)
        except Exception:                                       # noqa: BLE001
            moment = None
        if moment is not None and not (index.min() <= moment <= index.max()):
            findings.append(_finding(
                'target.evaluation_start', 'project.evaluation_start', BLOCKING,
                'that the evaluation period lies inside the series',
                f'{moment} is outside {index.min()} .. {index.max()}',
                'the later evaluation period would select no rows, so no score could be produced',
                'Move the evaluation start inside the covered period.'))
    return findings


# ---------------------------------------------------------------------------- the weather archive

def _nwp_findings(document, coverage):
    """One directory listing per archive. No grib message is opened here."""
    import pandas as pd
    from eforecast.nwp_extraction import sources
    findings = []
    models = _at(document, 'project.nwp_models') or []
    if not models:
        return findings
    horizon_type = _at(document, 'project.horizon_type')
    horizon = int(_at(document, 'project.horizon') or 0)
    # Every keyword and string form of pd.Timedelta raises a NumPy deprecation under this
    # pandas/NumPy pair; the two-argument form is the one that does not.
    ahead = pd.Timedelta(int(horizon), 'h') if horizon_type == 'short_term' else pd.Timedelta(1, 'D')
    for position, model in enumerate(models):
        if not isinstance(model, dict):
            continue
        name = model.get('model') or f'model {position}'
        field = f'project.nwp_models.{position}.path_nwp_source'
        root = model.get('path_nwp_source')
        if model.get('path_nwp_gdrive') is not None:
            findings.append(_drive_finding(name, position, model))
            continue
        if not root:
            findings.append(_finding('nwp.archive', field, BLOCKING,
                                     f'where the {name} forecasts are stored',
                                     'no archive folder is configured',
                                     'the weather inputs cannot be built, so the run stops at the nwp stage',
                                     f'Point this entry at the folder holding the {name} files.'))
            continue
        candidates = [Path(p) for p in sources.roots(str(root))]
        present = [p for p in candidates if p.is_dir()]
        if not present:
            findings.append(_finding('nwp.archive', field, BLOCKING,
                                     f'the {name} archive folder',
                                     f'none of {[str(p) for p in candidates]} is a directory',
                                     'the weather inputs cannot be built, so the run stops at the nwp stage',
                                     'Mount the archive disk, or correct the folder.'))
            continue
        days = sorted({entry.name for folder in present for entry in folder.iterdir()
                       if entry.is_dir() and len(entry.name) == 8 and entry.name.isdigit()})
        if not days:
            findings.append(_finding('nwp.archive', field, BLOCKING,
                                     f'day folders under the {name} archive',
                                     f'{present[0]} contains no YYYYMMDD day folder',
                                     'no forecast day can be read, so the weather inputs stay empty',
                                     'Extract the archive into day folders, or point at the folder that has them.'))
            continue
        first, last = pd.Timestamp(days[0]), pd.Timestamp(days[-1])
        findings.append(_finding('nwp.archive', field, OK,
                                 f'the days available in the {name} archive',
                                 f'{len(days)} day folders from {first.date()} to {last.date()}',
                                 'these are the days whose forecasts a run can use',
                                 'Nothing to do.'))
        if coverage is None:
            continue
        required = pd.Timestamp(coverage['last']).normalize() + ahead
        if last.normalize() < required.normalize():
            findings.append(_finding(
                'nwp.horizon', field, BLOCKING,
                f'that the {name} archive reaches the end of the run period',
                f'the archive stops {last.date()}; this run needs forecasts through {required.date()}',
                'the last part of the period has no weather input, so those samples are dropped '
                'and the evaluation covers less than the dates suggest',
                f'Extend the {name} archive to {required.date()}, or choose an earlier end date.'))
        missing = len(pd.date_range(first, last, freq='D')) - len(days)
        if missing > 0:
            share = round(100.0 * missing / max(len(pd.date_range(first, last, freq='D')), 1), 1)
            findings.append(_finding(
                'nwp.gaps', field, ATTENTION,
                f'missing days inside the {name} archive',
                f'{missing} days between {first.date()} and {last.date()} have no folder ({share}%)',
                'samples on those days lose their weather input and are dropped from training',
                'Fetch the missing days, or accept the shorter usable period.'))
    return findings


def _drive_finding(name, position, model):
    """A provider whose days come from a shared Google Drive folder when no local folder has them.

    Nothing is listed or downloaded here: this report reaches no network. So it says what the
    extract stage will do, and that the stage, not this report, names a day the folder lacks.
    An archive that is not on this disk is not a reason to refuse a case that says where it is.
    """
    from eforecast.nwp_extraction import gdrive, sources
    field = f'project.nwp_models.{position}.path_nwp_gdrive'
    link = model.get('path_nwp_gdrive')
    try:
        gdrive.folder_id(link)
    except ValueError:
        return _finding('nwp.archive', field, BLOCKING,
                        f'the Google Drive folder of the {name} forecasts',
                        f'{str(link)!r} is not a Google Drive folder link',
                        'the days that are not on this machine cannot be downloaded, so the run stops at extract',
                        'Paste the address of the shared folder (https://drive.google.com/drive/folders/...), '
                        'or choose the local folder as the source.')
    if not _available('gdown'):
        return _finding('nwp.archive', field, BLOCKING,
                        f'that this interpreter can download the {name} forecasts',
                        'gdown is not installed, and the shared folder is listed with it',
                        'the days that are not on this machine cannot be downloaded, so the run stops at extract',
                        'Ask the maintainer to install gdown as requirements.txt pins it.')
    root = model.get('path_nwp_source')
    held = {entry.name for folder in (sources.roots(str(root)) if root else []) if Path(folder).is_dir()
            for entry in Path(folder).iterdir() if entry.is_dir() and sources.DAY_NAME.fullmatch(entry.name)}
    return _finding('nwp.archive', field, OK,
                    f'where the {name} forecasts come from',
                    f'{len(held)} day folder(s) are in {root or "no local folder"}; a day that is not there is '
                    f'downloaded from Google Drive when extract runs, read into its daily file, and removed',
                    'whether the shared folder holds every day of the period is not checked here: the extract '
                    'stage names the days it could not get, and a later run asks only for those',
                    'Nothing to do.')


def _weather_lag_findings(document):
    """Whether the lags of a weather variable can be met by what a forecast holds.

    A daily file keeps the hours up to ``MAX_STEP_HOURS`` after each analysis, and a sample needs
    every one of its lags. In a ``long_term`` case a row is a predicted hour, one to seventy-three
    hours after its issue day begins, and a lag is counted from that hour: the newest forecast a
    row may use is that day's 12:00 analysis, so nothing later than about twelve hours plus the
    kept steps after the day begins has weather. Lags that are a list of every hour of the
    horizon -- what a short_term case rightly has, counted from its issue time -- leave a
    long_term case without a single sample.
    """
    from eforecast.nwp_extraction import sources
    findings = []
    if not (_at(document, 'project.nwp_models') or []):
        return findings
    long_term = _at(document, 'project.horizon_type') == 'long_term'
    kept = int(sources.MAX_STEP_HOURS)
    reach = 12 + kept if long_term else kept
    for position, variable in enumerate(_at(document, 'data.variables') or []):
        if not isinstance(variable, dict) or variable.get('type') != 'nwp':
            continue
        lags = variable.get('lags')
        if not isinstance(lags, list) or not lags:
            continue
        try:
            furthest = max(int(lag) for lag in lags)
        except (TypeError, ValueError):
            continue
        name, field = variable.get('name') or f'variable {position}', f'data.variables.{position}.lags'
        if long_term:
            covered = reach - furthest                          # the predicted hours that have every lag
            if covered < 1:
                findings.append(_finding(
                    'nwp.lags', field, BLOCKING,
                    f'that the lags of {name} can be met by a forecast',
                    f'{name} asks for weather up to {furthest} hours after each predicted hour, and the forecasts '
                    f'a row may use end about {reach} hours after its issue day begins',
                    'no predicted hour has weather for every lag, so the nwp stage builds no sample and stops',
                    'Use lags around the predicted hour: [-1, 0, 1] is what long_term cases use. In a long_term '
                    'case a lag is counted from each predicted hour, not from the issue time.'))
            elif furthest > 2:
                findings.append(_finding(
                    'nwp.lags', field, ATTENTION,
                    f'that the lags of {name} can be met by a forecast',
                    f'{name} asks for weather up to {furthest} hours after each predicted hour, so only about the '
                    f'first {covered} of the 73 predicted hours of an issue day have every lag',
                    'the later hours get no sample: they are neither trained on nor evaluated',
                    'Use lags around the predicted hour ([-1, 0, 1]) to keep the whole three days.'))
        elif furthest > reach:
            findings.append(_finding(
                'nwp.lags', field, BLOCKING,
                f'that the lags of {name} can be met by a forecast',
                f'{name} asks for weather {furthest} hours after the issue time, and a forecast is kept for '
                f'{kept} hours after its analysis',
                'no sample has weather for every lag, so the nwp stage builds no sample and stops',
                f'Keep the lags of {name} within {kept} hours.'))
    return findings


# ---------------------------------------------------------------------------- the configured runtime

def _available(module):
    """Is the module importable here? Asked of the import system, never by importing."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _runtime_findings(document):
    findings = []
    from eforecast.runtime_registry import effective_python, resolve, require_available, ContractError
    interpreter = effective_python()
    findings.append(_finding('runtime.interpreter', None, OK,
                             'which interpreter the installation selects for native work',
                             interpreter,
                             'the service selects this interpreter; isolated model families use their approved runtimes',
                             'Nothing to do.'))
    # Legacy env_name/path_env are retained notes, not runtime selectors. A mismatch
    # cannot affect execution and is not an actionable readiness warning.

    try:
        binding = require_available(resolve(_at(document, 'methods', {})), python=interpreter)
        if binding['selection']:
            findings.append(_finding('runtime.selection', None, OK,
                'the approved runtimes resolved for this case',
                ', '.join(f'{method}: {runtime}' for method, runtime in sorted(binding['selection'].items())),
                'these definitions are frozen when the run is accepted; model availability is checked separately',
                'Nothing to do.'))
    except ContractError as error:
        findings.append(_finding('runtime.selection', None, BLOCKING,
            'the approved runtimes resolved for this case', str(error), 'submission is refused; no native fallback is applied',
            'Ask the maintainer to restore or configure the approved runtime for the selected models.'))

    if os.path.abspath(interpreter) != os.path.abspath(sys.executable):
        findings.append(_finding('runtime.unverified', None, ATTENTION,
            'whether this service can verify the configured interpreter', interpreter,
            'module checks in this service cannot establish availability in a different environment',
            'Run eforecast doctor with the configured interpreter before launching.'))
        findings.extend(_image_findings())
        return findings

    wanted = [name for name, on in (_at(document, 'methods.methods_cpu') or {}).items() if on]
    from eforecast.runtime_launcher import CASE_METHODS
    for method in sorted(wanted):
        module, distribution = FAMILY_DISTRIBUTIONS.get(method, (None, None))
        if module is None:
            # a family with no host distribution at all -- TimeGPT, Timestral, T0, Lag-Llama --
            # is served only by a pinned image, and that is what must be reported
            if method in CASE_METHODS:
                findings.append(_image_served_finding(method, 'it', interpreter))
            continue
        if _available(module):
            findings.append(_finding('runtime.families', 'methods.methods_cpu', OK,
                                     f'that the dependency for {method} is available',
                                     f'{distribution} is installed',
                                     'module availability is not a qualified forecast or checkpoint check',
                                     'Nothing to do.'))
        else:
            findings.append(_image_served_finding(method, distribution, interpreter))

    if _at(document, 'project.nwp_models'):
        absent = [module for module in GRIB_MODULES if not _available(module)]
        findings.append(_finding(
            'runtime.grib', 'project.nwp_models', BLOCKING if absent else OK,
            'that this interpreter can read grib weather files',
            f'{", ".join(absent)} missing' if absent else 'pygrib and eccodes are installed',
            'the weather extraction stage cannot run' if absent else 'weather extraction can run',
            f'Ask the maintainer to install {", ".join(absent)}.' if absent else 'Nothing to do.'))

    findings.append(_estimator_finding())
    findings.extend(_image_findings())
    return findings


def _image_served_finding(method, distribution, interpreter):
    """What to do about a family the configured interpreter does not have.

    This used to be one BLOCKING finding advising the maintainer to install the package.
    For the A16-02 families that advice is actively harmful: `foundationforecast` resolves at
    pandas 2.3.3 and `nixtla` requires pandas<3, so installing either into the configured
    interpreter would undo the A13-05 pandas 3 upgrade and bring back the pickle
    incompatibility A13-01 closed. The right answer is the pinned image, not the host.
    """
    from eforecast import image_runtime, runtime_launcher

    served = runtime_launcher.CASE_METHODS.get(method)
    if served is None:
        return _finding(
            'runtime.families', 'methods.methods_cpu', BLOCKING,
            f'that {method} can run in the configured interpreter',
            f'{distribution} is not installed in {interpreter or "this interpreter"}',
            f'every {method} trial fails on import; other selected methods are unaffected',
            f'Ask the maintainer to install {distribution} in the configured interpreter, '
            f'or clear {method} for now.')
    try:
        reference = runtime_launcher.image_for(served)
        identity = image_runtime.image_id(reference)
    except Exception:                                                   # noqa: BLE001
        reference, identity = None, None
    if identity is not None:
        return _finding(
            'runtime.families', 'methods.methods_cpu', OK,
            f'how {method} will run, since this interpreter does not carry it',
            f'the pinned image {reference} serves it ({identity[:19]}...)',
            'the work is dispatched to that image; nothing is installed on the host, because '
            'these families require pandas<3 and the host runs pandas 3',
            'Nothing to do.')
    return _finding(
        'runtime.families', 'methods.methods_cpu', BLOCKING,
        f'that {method} can run anywhere this installation can reach',
        f'{distribution} is not in {interpreter or "this interpreter"}, and the image that '
        f'serves it ({reference or "unpinned"}) is not built here',
        f'every {method} trial fails; other selected methods are unaffected',
        f'Ask the maintainer to build the model image (bin/eforecast images refresh). Do NOT '
        f'install {distribution} in the configured interpreter: it requires pandas<3 and '
        f'would break every retained pandas 3 artifact.')


def _estimator_finding():
    """The qualified reader range for retained estimators (A13-03)."""
    try:
        import sklearn
        version = sklearn.__version__
        parts = tuple(int(p) for p in version.split('.')[:2])
    except Exception:                                           # noqa: BLE001
        return _finding('runtime.estimators', None, ATTENTION,
                        'the scikit-learn version that will read retained models',
                        'scikit-learn is not importable here',
                        'retained scalers and cluster models cannot be read',
                        'Ask the maintainer to install scikit-learn in the configured interpreter.')
    low, high = QUALIFIED_SKLEARN
    inside = low <= parts < high
    return _finding(
        'runtime.estimators', None, OK if inside else ATTENTION,
        'the scikit-learn version that will read retained models',
        f'{version}' + ('' if inside else f', outside the compared range {low[0]}.{low[1]}..{high[0]}.{high[1]}'),
        'retained cluster models and scalers were compared against this range and answer identically'
        if inside else 'retained models load but were never compared against this version, so a '
                       'silent numerical difference is possible',
        'Nothing to do.' if inside else
        'Ask the maintainer to re-run the retained-estimator comparison, or to pin a compared version.')


def _image_findings():
    """A container trial needs a witness for the exact image it will launch."""
    reference = os.environ.get('EFORECAST_WORKER_IMAGE')
    if not reference:
        return []
    try:
        from eforecast import image_runtime
        identity = image_runtime.image_id(reference)
        witness = image_runtime.load(reference) if identity else None
    except Exception as error:                                  # noqa: BLE001
        return [_finding('runtime.image', None, ATTENTION,
                         f'the recorded runtime of worker image {reference}',
                         f'{type(error).__name__}: {str(error)[:120]}',
                         'a container trial cannot be matched against a profile',
                         'Ask the maintainer to check the image and its recorded witness.')]
    if identity is None:
        return [_finding('runtime.image', None, BLOCKING,
                         f'that worker image {reference} exists here',
                         'the image cannot be inspected',
                         'container trials cannot start',
                         f'Ask the maintainer to build or pull {reference}.')]
    if witness is None:
        return [_finding(
            'runtime.image', None, ATTENTION,
            f'a recorded runtime witness for {reference}',
            f'nothing is recorded for the id this tag now points at ({identity[:19]}...)',
            'pinned execution profiles cannot be activated for container trials; the run still '
            'works with the established policy',
            f'Ask the maintainer to run: python -m eforecast.image_runtime record --image {reference}')]
    return [_finding('runtime.image', None, OK,
                     f'a recorded runtime witness for {reference}',
                     f'recorded for {identity[:19]}... (python {witness["hardware"]["python"]})',
                     'container trials can be matched against a pinned profile',
                     'Nothing to do.')]


# ---------------------------------------------------------------------------- case identity

#: Enumerating other cases is bounded: a lab has a handful, not a catalogue.
MAX_SIBLING_CASES = 50


def _release_findings(document):
    """Are the images this case needs the ones the selected release describes? -- A16-11.

    The witness check elsewhere asks whether an image id was qualified. This asks the other
    half: whether that id was built from the code the release names. Both can pass
    independently, and a run needs both -- a witnessed image built from stale source still
    produces the stale answer.
    """
    from eforecast import image_release

    wanted = image_release.required_for(document)
    if not wanted:
        return []
    release = image_release.selected()
    if release is None:
        return [_finding(
            'runtime.release', 'methods.methods_cpu', ATTENTION,
            'which image release this run would be checked against',
            'none is selected, and the families this case keeps run in images',
            'the images cannot be compared with anything, so a stale one would go unnoticed '
            'and produce a wrong number rather than an error',
            'Ask the maintainer to capture a release and set EFORECAST_RELEASE.')]
    try:
        # ask_docker=False: this answers a web request, and the established invariant is that
        # readiness runs no subprocess. The build inputs that are files are the valuable half
        # anyway -- an edited adapter that was never rebuilt is the case this catches. The
        # image id and its witness are the doctor's to check.
        report_payload = image_release.check(release, runtimes=wanted, ask_docker=False)
    except image_release.ReleaseError as error:
        return [_finding(
            'runtime.release', 'methods.methods_cpu', ATTENTION,
            f'the image release {release!r}', str(error)[:160],
            'coherence cannot be established for this run',
            'Ask the maintainer to capture the release again.')]
    allowed, blocking = image_release.activatable(report_payload, wanted)
    if allowed:
        return [_finding(
            'runtime.release', 'methods.methods_cpu', OK,
            f'that the build inputs for this case\'s images match release {release}',
            ', '.join(wanted),
            'the source these images are built from is the source the release names; that '
            'the built images are those exact ids is checked by eforecast doctor',
            'Nothing to do.')]
    return [_finding(
        'runtime.release', 'methods.methods_cpu', BLOCKING,
        f'that the build inputs for this case\'s images match release {release}',
        '; '.join(blocking)[:240],
        'this run would not execute the code the release was qualified with; a half-rebuilt '
        'image set must not serve a run',
        'Ask the maintainer to rebuild the named image(s) and re-check '
        '(python -m eforecast.image_release check --release '
        f'{release}). Recording a fresh witness does not fix stale bytes.')]


def _identity_findings(document, static_data, case=None, siblings=None):
    """Does this case address a model folder that already belongs to another case?

    The folder is ``<group>/<type>/<project name>/<horizon type>/model_ver<N>``, so two cases
    with the same project name, type, horizon type and version share it -- and the second one to
    run overwrites the first's frozen masks, global scaler fits and trained models. Copying a
    case used to produce exactly that, and a hand-edited name still can.

    ``siblings`` is ``{case name: path_model}``, supplied by the caller. It is a parameter rather
    than something this function discovers for two reasons: enumerating "every case" would mean
    the caller's own visibility is ignored, so a finding could name a case the reader is not
    entitled to know exists; and a check that reaches for a global workspace answers differently
    depending on an environment variable the caller did not choose. Without it this says nothing.
    """
    path_model = (static_data or {}).get('path_model')
    if not path_model or not siblings:
        return []
    clashes = sorted(name for name, other in siblings.items()
                     if name != case and other and other == path_model)
    if not clashes:
        return [_finding('project.identity', 'project.name', OK,
                         'that no other case writes to the same model folder',
                         f'this case owns {path_model}',
                         'its datasets, scalers and models are its own', 'Nothing to do.')]
    return [_finding(
        'project.identity', 'project.name', BLOCKING,
        'that no other case writes to the same model folder',
        f'case {", ".join(clashes)} resolves to the same folder: {path_model}',
        "whichever runs second overwrites the other's frozen data splits, global scaler fits "
        'and trained models, and its results would describe the wrong site',
        'Give this case its own site name, type, horizon type or model version so the two stop '
        'sharing a folder. Do not run either until they differ.')]


def case_model_paths(scope, limit=MAX_SIBLING_CASES):
    """``{case name: path_model}`` for the cases this caller may see.

    Built by the caller, which knows its own workspace and permissions, and bounded because a
    lab has a handful of cases rather than a catalogue. Reads each case's configuration only.
    """
    from eforecast.config.config_files import load_document
    out = {}
    try:
        names = [folder.name for folder in scope._all_case_dirs()]        # noqa: SLF001
    except Exception:                                           # noqa: BLE001
        return out
    for name in names[:limit]:
        try:
            document = load_document(scope.case_dir(name))
            if document:
                out[name] = (resolved(document) or {}).get('path_model')
        except Exception:                                       # noqa: BLE001
            continue
    return out


# ---------------------------------------------------------------------------- applicability

def _applicability_findings(document):
    """A13-08: selections that are legal and do nothing. Never fatal to the report itself."""
    try:
        from eforecast.config import applicability
        return applicability.findings(document)
    except Exception as error:                                  # noqa: BLE001
        return [_finding('applicability', 'methods.method_params', ATTENTION,
                         'whether every selected parameter reaches the graph that runs',
                         f'the check could not run ({type(error).__name__}: {str(error)[:110]})',
                         'an ignored setting would not be reported, which is not the same as '
                         'there being none',
                         'Ask the maintainer to review the methods of this case.')]


# ---------------------------------------------------------------------------- cluster support

def _daylight_share(document, coverage):
    """About the share of a series' rows a training run keeps when the case removes night hours.

    ``None`` when it removes none, or when the site cannot be asked. A training reader keeps the
    hours from three before dawn to three after dusk at the site (``files_manager.daylight``).
    One day a week of the period is asked: the share moves with the season, not with the day.
    """
    if not _at(document, 'data.remove_night_hours') or not coverage:
        return None
    try:
        import astral
        import pandas as pd
        from astral.sun import sun
        latitude, longitude = (float(value) for value in _at(document, 'project.coordinates')[:2])
        zone = _at(document, 'project.local_timezone') or 'UTC'
        site = astral.LocationInfo('site', 'region', zone, latitude, longitude)
        kept = []
        for day in pd.date_range(pd.Timestamp(coverage['first']).normalize(),
                                 pd.Timestamp(coverage['last']).normalize(), freq='7D'):
            times = sun(site.observer, date=day.date(), tzinfo=zone)
            kept.append(min(24., (times['dusk'] - times['dawn']).total_seconds() / 3600 + 6))
        return sum(kept) / (24 * len(kept)) if kept else None
    except Exception:                                           # noqa: BLE001 -- an estimate, or none
        return None


def _cluster_sample_findings(document, coverage):
    """Whether the clusterer's sample limits can be met by the rows the series has.

    The rule initialiser refuses a case whose ``min_samples`` is more than ``max_samples_ratio``
    of its training rows (``train_utils``), and a run finds that out at its clustering stage,
    after the weather has been extracted. The series says how many rows there can be at most:
    the timestamps it holds before the evaluation period, less the night hours when the case
    removes them, less the share the split holds out. A stage can still drop rows (hours
    without weather), so a case that passes here may be refused there; one that does not pass
    cannot run.

    Found on a solar park copied from a quarter-hourly case to an hourly one: ``min_samples``
    8000 for a series of 20,232 rows. Eight months of it are the evaluation period and a fifth
    of the rest is night: the run had 6,840 training rows, of which 0.45 is 3,078. Counting
    every row of the series, the advice was "at most 6828", and a case set to that would have
    been refused again.
    """
    params = _at(document, 'methods.method_params.clustering')
    if not _at(document, 'project.is_fuzzy') or not coverage or not isinstance(params, dict):
        return []
    try:
        least, ratio = int(params['min_samples']), float(params['max_samples_ratio'])
        held = 2 * float(_at(document, 'methods.val_test_ratio', 0.125))
    except (KeyError, TypeError, ValueError):
        return []
    rows = int(coverage.get('rows') or 0)
    before = int(coverage.get('rows_before_evaluation', rows))
    share = _daylight_share(document, coverage)
    kept = int(before * share) if share else before
    training = int(kept * max(0.0, 1 - held))
    left = ((f'{before} are before the evaluation period, ' if before != rows else '')
            + (f'night hours are removed, which leaves about {kept}, and ' if share else ''))
    field = 'methods.method_params.clustering.min_samples'
    checked = 'that a cluster of min_samples rows fits in max_samples_ratio of the training rows'
    refusal = 'refuses the case ("clustering sample limits are infeasible")'
    if least <= int(ratio * training):
        return [_finding('clusters.sample_limits', field, OK, checked,
                         f'min_samples {least} fits: about {training} training rows allow clusters of up to '
                         f'{int(ratio * training)}',
                         'the rule initialiser accepts these limits, unless a stage drops many rows',
                         'Nothing to do.')]
    everything = least > int(ratio * before)
    remedy = (f'Lower min_samples to at most {int(ratio * training)}, or raise max_samples_ratio '
              f'to at least {min(1.0, least / max(training, 1)):.2f}; clusters that large leave room for few rules. '
              'Leave a margin: rows on days without weather are dropped later and are not counted here.')
    if everything:
        return [_finding('clusters.sample_limits', field, BLOCKING, checked,
                         f'min_samples is {least}, and max_samples_ratio {ratio:g} of '
                         + (f'all {rows} rows of the series' if before == rows else
                            f'the {before} rows before the evaluation period')
                         + f' is {int(ratio * before)}',
                         f'the clustering stage {refusal}, after the weather has been extracted', remedy)]
    return [_finding('clusters.sample_limits', field, ATTENTION, checked,
                     f'min_samples is {least}; about {training} of the {rows} rows of the series are training '
                     f'rows ({left}the split holds out {held:.0%}), and max_samples_ratio {ratio:g} of them is '
                     f'{int(ratio * training)}',
                     f'the clustering stage {refusal} unless nearly every row of the series becomes a '
                     'training row, and it says so only after the weather has been extracted', remedy)]


def _cluster_findings(document, static_data):
    """The one protective finding from the old diagnostic panel, and nothing else.

    This reads the partition report a previous run wrote. It never recomputes support, never
    changes a fraction, a scaler scope or the fallback policy, and when no report exists it
    says so instead of implying the clusters are fine.
    """
    if not any((_at(document, 'methods.cluster_methods') or {}).values()):
        return []
    if not static_data:
        return [_finding('clusters.support', 'methods.cluster_methods', ATTENTION,
                         'measured support for each cluster',
                         'the case cannot be resolved far enough to locate its reports',
                         'whether a cluster will fall back to global training scope is unknown',
                         'Save the case, then re-check.')]
    try:
        from eforecast.datasets.data_preprocessing import partition_report
        path = Path(partition_report.report_file(static_data))
    except Exception as error:                                  # noqa: BLE001
        return [_finding('clusters.support', 'methods.cluster_methods', ATTENTION,
                         'measured support for each cluster',
                         f'the report location cannot be resolved ({type(error).__name__})',
                         'whether a cluster will fall back to global training scope is unknown',
                         'Run the clustering stage, then re-check.')]
    if not path.is_file():
        # Reported, not flagged: only the clustering stage can write this report, so before it has
        # run there is nothing to put right. The wording still refuses to call the clusters fine.
        return [_finding('clusters.support', 'methods.cluster_methods', OK,
                         'measured support for each cluster',
                         'no partition report exists yet for this case',
                         'cluster support has not been measured, so a fallback to global training '
                         'scope cannot be ruled out',
                         'Run the clustering stage, then re-check.')]
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return [_finding('clusters.support', 'methods.cluster_methods', ATTENTION,
                         'measured support for each cluster',
                         f'the partition report cannot be read ({type(error).__name__})',
                         'cluster support is unknown',
                         'Re-run the clustering stage, then re-check.')]
    findings = []
    for row in _rows(report):
        name = row.get('cluster')
        if not name or name == 'global':
            continue
        # The report names the part ``validation``. This read ``val_days``, which no report ever
        # held, so every cluster of every case that had run its clustering stage "had 0
        # validation days" and the case could not start.
        validation = row.get('validation_days', row.get('val_days'))
        if row.get('below_min_samples') or row.get('intended_deep_scope') == 'global_if_available':
            findings.append(_finding(
                'clusters.support', 'methods.cluster_methods', ATTENTION,
                f'measured support for cluster {name}',
                f'cluster {name} has {row.get("samples")} samples '
                f'({"?" if validation is None else validation} validation days, {row.get("test_days", "?")} test days), '
                'below the configured minimum',
                f'training for {name} falls back to the global scope, so its model is fitted on all '
                'data rather than on the cluster, and its score describes a global model',
                'Accept the fallback, or widen the cluster so it has its own support. '
                'The guide changes no fractions by itself.'))
        elif not validation or not row.get('test_days'):
            findings.append(_finding(
                'clusters.support', 'methods.cluster_methods', BLOCKING,
                f'measured support for cluster {name}',
                f'cluster {name} has {validation or 0} validation and '
                f'{row.get("test_days", 0)} test days',
                'the cluster cannot be validated or scored, so selection has nothing to compare',
                'Widen the cluster, or exclude it from this run.'))
    if not findings:
        findings.append(_finding('clusters.support', 'methods.cluster_methods', OK,
                                 'measured support for each cluster',
                                 'every cluster has its own validation and test support',
                                 'each cluster trains in its own scope',
                                 'Nothing to do.'))
    return findings


def _rows(report):
    """The per-cluster rows of a partition report, whatever section holds them."""
    if isinstance(report, dict):
        for key in ('rows', 'clusters', 'coverage'):
            value = report.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        for value in report.values():
            if isinstance(value, dict):
                found = _rows(value)
                if found:
                    return found
    return []


# ---------------------------------------------------------------------------- the report

#: The fields the findings depend on. A change to any of them invalidates a verdict.
WATCHED_FIELDS = (
    'project.filename', 'project.ts_resolution', 'project.horizon', 'project.horizon_type',
    'project.site_timezone', 'project.local_timezone', 'project.evaluation_start',
    'project.nwp_models', 'project.name', 'project.owner', 'project.group', 'project.version_group',
    'data.target_variable', 'data.global_past_lags', 'data.global_future_lags',
    'methods.methods_cpu', 'methods.cluster_methods', 'methods.global_methods',
    'methods.runtime_ids', 'folders.sys_folder',
)


def inputs_digest(document) -> str:
    """A digest of exactly what the checks read from the document, plus the runtime they ran in.

    The runtime is part of it on purpose: the same document checked against a different
    interpreter or worker image is a different question, and the earlier answer must not be
    reused for it.
    """
    watched = {field: _at(document, field) for field in WATCHED_FIELDS}
    from eforecast.runtime_registry import effective_python, resolve, ContractError
    runtime = {'python': effective_python(),
               'image': os.environ.get('EFORECAST_WORKER_IMAGE')}
    try:
        runtime['binding'] = resolve(_at(document, 'methods', {}))['revision']
    except ContractError as error:
        runtime['binding_error'] = str(error)
    except (AttributeError, TypeError, KeyError, ValueError) as error:
        # A draft may hold anything a colleague typed: `methods_cpu: some-string` instead of a
        # mapping is an ordinary mistake, and the registry raises on it. This function's job is
        # to change when the document changes, so a document the registry cannot read is
        # recorded as exactly that -- crashing here would take down the readiness report the
        # colleague is reading in order to find out what is wrong.
        runtime['binding_error'] = f'unreadable method binding ({type(error).__name__}: {error})'
    payload = json.dumps({'fields': watched, 'runtime': runtime}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def verdict(findings) -> str:
    severities = {finding['severity'] for finding in findings}
    if BLOCKING in severities:
        return CANNOT_START
    if ATTENTION in severities:
        return NEEDS_ATTENTION
    return READY


def resolved(document, path=None):
    """``static_data`` when the document is complete enough, else ``None``.

    Only used to locate a case's own reports. This resolves configuration into paths; it reads
    no dataset, fits nothing and launches nothing. An incomplete draft simply yields ``None``,
    and the cluster check then says support is unknown rather than implying it is fine.
    """
    from eforecast.config.service import preview_static_data
    try:
        static_data = preview_static_data(document, path)
    except Exception:                                           # noqa: BLE001
        return None
    try:
        # ``dry_run`` adds the folder names, including ``path_model``, and creates nothing.
        from eforecast.init.initialize import create_paths
        return create_paths(dict(static_data), dry_run=True)
    except Exception:                                           # noqa: BLE001
        return static_data


def report(document, *, static_data=None, path=None, case=None, siblings=None,
           now=None) -> dict[str, Any]:
    """The readiness of one case document, as three outcomes and a list of findings."""
    from eforecast.config.service import derived_facts
    document = document or {}
    if static_data is None:
        static_data = resolved(document, path)
    findings: list[dict[str, Any]] = []
    try:
        derived = derived_facts(document)
    except Exception:                                           # noqa: BLE001
        derived = {}
    target, coverage = _target_findings(document, derived)
    findings.extend(target)
    findings.extend(_nwp_findings(document, coverage))
    findings.extend(_weather_lag_findings(document))
    findings.extend(_runtime_findings(document))
    findings.extend(_release_findings(document))
    findings.extend(_identity_findings(document, static_data, case=case, siblings=siblings))
    findings.extend(_applicability_findings(document))
    findings.extend(_cluster_sample_findings(document, coverage))
    findings.extend(_cluster_findings(document, static_data))
    counts = {severity: sum(1 for f in findings if f['severity'] == severity)
              for severity in (BLOCKING, ATTENTION, OK)}
    return {'verdict': verdict(findings), 'checked_at': _moment(now), 'checked_at_epoch': now or time.time(),
            'inputs_digest': inputs_digest(document), 'findings': findings, 'counts': counts,
            'coverage': coverage, 'bounded': {'max_index_rows': MAX_INDEX_ROWS}}


def describes(report_payload, document) -> bool:
    """Does this report still describe that document, in this runtime?"""
    return bool(report_payload) and report_payload.get('inputs_digest') == inputs_digest(document)


def _moment(now=None):
    return time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(now if now is not None else time.time()))
