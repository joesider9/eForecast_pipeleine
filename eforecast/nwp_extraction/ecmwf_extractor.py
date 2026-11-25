# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

import datetime
import logging
import os
import traceback

import joblib
import numpy as np
import pandas as pd

import pygrib
from eforecast.nwp_extraction import gdrive, sources

log = logging.getLogger(__name__)

from eforecast.nwp_extraction.nwp_pickles import (ACCUMULATED, DIRECT, NAMES, NON_NEGATIVE, PICKLE_FORMAT,  # noqa: F401
                                                    compact_pickle, merge_runs)


def one_file_per_step(files):
    """One file per analysis cycle and step, keeping the first.

    A day folder can hold a misfiled copy of another cycle -- one in this archive holds 53 files
    whose names say the 00:00 cycle beside the 12:00 cycle it is named for. Reading both copies
    let the second overwrite the first, and for an accumulated field the overwriting value was
    computed by differencing against a step already past, so a duplicate did not merely repeat a
    value: it replaced a good one with a wrong one.
    """
    out, seen = [], set()
    for path in files:
        found = sources.forecast_step(path)
        if found is not None:
            if found in seen:
                log.warning('ECMWF: %s repeats cycle %s step %sh; the first copy is kept',
                            os.path.basename(path), found[0], found[1])
                continue
            seen.add(found)
        out.append(path)
    return out


class EcmwfExtractor:
    """Daily ECMWF pickles for a project group.

    ``nwps_extract(t)`` reads the grib data of day ``t`` (a single
    ``NWP_<day>T00.grb`` file or one folder of per-step files per run) in one
    pass, de-accumulates the radiation fields step by step, derives wind speed
    and direction, and writes ``ecmwf_<ddmmyy>.pickle`` in format 2: float32
    arrays, the grid stored once, steps up to ``max_step_hours`` after each
    run's analysis time, joblib-compressed.

    ``remote`` is the shared folder a day that is in no local folder is read from
    (``gdrive.attached`` sets it): the day is downloaded, read, and its files removed again.
    """

    def __init__(self, variables, path_nwp, path_group_nwp=None, max_step_hours=sources.MAX_STEP_HOURS, compress=3,
                 download_missing=False):
        unknown = [var for var in variables if var not in NAMES]
        if unknown:
            raise ValueError(f'Unknown NWP variables {unknown}; known: {sorted(NAMES)}')
        self.variables = list(variables)
        self.path_group_nwp = path_group_nwp
        self.max_step_hours = max_step_hours
        self.compress = compress
        self.download_missing = download_missing
        self.remote = None
        path_nwp = str(path_nwp)
        self.roots = [str(path) for path in sources.roots(path_nwp)]
        self.path_nwp = next((root for root in self.roots if os.path.isdir(root)), path_nwp)
        self.needed = {field for var in self.variables for field in NAMES[var]}

    # ------------------------------------------------------------------ sources

    def day_folder(self, t):
        return sources.day_folder(self.roots, pd.Timestamp(t))

    @staticmethod
    def grib_files(path_date):
        """The grib files of one day in step order: the single file, or the sorted per-step files of every run."""
        return sources.grib_files(path_date)

    def pickle_file(self, t):
        return os.path.join(self.path_group_nwp, f"ecmwf_{pd.Timestamp(t).strftime('%d%m%y')}.pickle")

    def absent(self, days, refit=False):
        """The days of ``days`` still to extract that no local folder holds."""
        return [t for t in days if self.day_folder(t) is None and
                (refit or self.path_group_nwp is None or not os.path.exists(self.pickle_file(t)))]

    # ------------------------------------------------------------------ reading

    def read_gribs(self, files, missing=()):
        """``(runs, lat, long)`` with ``runs = {'0000': {hour key: {ecmwf field: float32 array}}}``.

        ``missing`` names the provider files of this day that hold nothing -- absent or zero-byte.
        They cost more than their own hour. An ``ACCUMULATED`` field is stored as a total since the
        analysis time and de-accumulated by subtracting the step before it, so the first readable
        step after a gap would be given two intervals of radiation and published as one. That is
        not a hole in the data, it is a plausible wrong number, and nothing downstream could tell.
        So those steps are dropped and named, and the rest of the day is kept.

        The whole step goes, not just its accumulated field: a step carrying some variables and
        not others is a shape the dataset readers do not expect, and a case that asks for no
        accumulated variable at all never reaches this at all.
        """
        gaps, unplaced = dict(), False
        for item in missing:
            found = sources.forecast_step(item)
            if found is None:
                unplaced = True        # a name that does not say which cycle and step it is
            else:
                gaps.setdefault(found[0], set()).add(found[1])
        if unplaced:
            log.warning('ECMWF: a missing provider file does not name its step, so no accumulated '
                        'field of this day can be trusted across a step boundary')
        runs = dict()
        unreliable = dict()
        previous = dict()  # (run, accumulated field) -> (step, last raw field), for de-accumulation
        lat = long = None
        for file in files:
            grb = pygrib.open(file)
            try:
                for message in grb:
                    field = message.cfVarNameECMF
                    if field not in self.needed:
                        continue
                    analysis = pd.to_datetime(message.analDate)
                    step = int(message.step)
                    if self.max_step_hours is not None and step > self.max_step_hours:
                        continue
                    run = analysis.strftime('%H%M')
                    key = (analysis + pd.DateOffset(hours=step)).strftime('%d%m%y%H%M')
                    data, lat_, long_ = message.data()
                    if lat is None:
                        lat, long = lat_, long_
                    elif lat_.shape != lat.shape:
                        raise ValueError(f'{file}: grid {lat_.shape} differs from {lat.shape}')
                    if field in ACCUMULATED:
                        raw = data
                        # Step 0 of the cycle is the analysis time itself: nothing accumulated yet,
                        # which is also the right base for the first step that is readable.
                        base_step, base = previous.get((run, field)) or (0, None)
                        previous[(run, field)] = (step, raw)
                        if base is not None and step <= base_step:
                            unreliable[(run, key)] = step    # a repeated step is not a later total
                        elif unplaced or any(base_step < gap < step for gap in gaps.get(run, ())):
                            unreliable[(run, key)] = step
                        elif base is not None:
                            data = data - base
                    runs.setdefault(run, dict()).setdefault(key, dict())[field] = np.asarray(data, dtype=np.float32)
            finally:
                grb.close()
        dropped = dict()
        for (run, key), step in unreliable.items():
            if runs.get(run, dict()).pop(key, None) is not None:
                dropped.setdefault(run, []).append(step)
        for run in [run for run, hours in runs.items() if not hours]:
            runs.pop(run)
        for run, steps in sorted(dropped.items()):
            log.warning('ECMWF cycle %s: dropped %d step(s) rather than de-accumulate across a '
                        'gap: %s', run, len(steps), ', '.join(f'{s}h' for s in sorted(steps)))
        return runs, lat, long

    def build_variables(self, runs):
        """Requested variables from the raw fields; derived winds and gusts included."""
        r2d = 45.0 / np.arctan(1.0)
        out = dict()
        for run, hours in runs.items():
            if not hours:
                continue        # every step of this cycle was dropped; it contributes nothing
            out[run] = dict()
            for key, fields in hours.items():
                values = dict()
                for var in self.variables:
                    if var in DIRECT and DIRECT[var] in fields:
                        values[var] = np.maximum(fields[DIRECT[var]], 0) if var in NON_NEGATIVE else fields[DIRECT[var]]
                    elif var in ('WS', 'WD') and {'u100', 'v100'} <= fields.keys():
                        u, v = fields['u100'], fields['v100']
                        values[var] = np.sqrt(u * u + v * v) if var == 'WS' else np.arctan2(u, v) * r2d + 180
                    elif var in ('WS10', 'WD10') and {'u10', 'v10'} <= fields.keys():
                        u, v = fields['u10'], fields['v10']
                        values[var] = np.sqrt(u * u + v * v) if var == 'WS10' else np.arctan2(u, v) * r2d + 180
                out[run][key] = values
            if 'WindGust' in self.variables:  # not available at every step: interpolate over the hours
                keys = sorted(out[run].keys(), key=lambda k: pd.to_datetime(k, format='%d%m%y%H%M'))
                shape = next(iter(hours.values()))[next(iter(next(iter(hours.values()))))].shape
                stack = np.array([out[run][k].get('WindGust', np.full(shape, np.nan, dtype=np.float32)) for k in keys],
                                 dtype=float)
                flat = pd.DataFrame(stack.reshape(stack.shape[0], -1)).interpolate('linear').values
                for i, k in enumerate(keys):
                    out[run][k]['WindGust'] = flat[i].reshape(shape).astype(np.float32)
        return out

    def extract_day(self, t):
        """Format-2 dictionary of day ``t`` or None when the day has no grib data."""
        path_date = self.day_folder(t)
        if path_date is None and self.download_missing:
            DownLoader(date=t, path_nwp=self.path_nwp).download()
            path_date = self.day_folder(t)
        if path_date is not None or self.remote is None or not self.remote.has(t):
            return self._read_day(t, path_date)
        # In no local folder and in the shared one: downloaded, read, and removed again. A day
        # that is on the disk is never fetched, and its files are never the ones removed.
        fetched = self.remote.fetch(t)
        try:
            return self._read_day(t, fetched)
        finally:
            if fetched is not None:
                self.remote.release(fetched)

    def _read_day(self, t, path_date):
        day = pd.Timestamp(t).strftime('%Y-%m-%d')
        if path_date is None:
            return None
        files = one_file_per_step(self.grib_files(path_date))
        if not files:
            return None
        usable, empty = sources.readable(files)
        if empty and self.download_missing:
            from eforecast.nwp_extraction import refill
            refill.restore(empty)
            usable, empty = sources.readable(files)
        if empty:
            shown = sorted(os.path.basename(path) for path in empty)
            log.warning('ECMWF %s: %d of %d provider file(s) hold no bytes, so their steps are '
                        'missing: %s%s', day, len(empty), len(files), ', '.join(shown[:5]),
                        f' and {len(shown) - 5} more' if len(shown) > 5 else '')
        if not usable:
            return None
        runs, lat, long = self.read_gribs(usable, missing=empty)
        if not runs:
            return None
        nwps = self.build_variables(runs)
        nwps.update({'format': PICKLE_FORMAT, 'lat': lat, 'long': long, 'variables': list(self.variables),
                     'max_step_hours': self.max_step_hours})
        return nwps

    def nwps_extract(self, t, refit=False):
        """Write the pickle of day ``t``; returns a status word (or the dictionary when no group folder is set)."""
        t = pd.Timestamp(t)
        if self.path_group_nwp is not None and os.path.exists(self.pickle_file(t)) and not refit:
            return 'exists'
        try:
            nwps = self.extract_day(t)
        except gdrive.DriveError as error:
            # Not a day without weather: the shared folder has it and it did not arrive. Its own
            # word, so that whoever asked can ask again instead of recording a day as empty.
            log.error('ECMWF %s: not downloaded from Google Drive: %s', t.strftime('%Y-%m-%d'), error)
            return 'unfetched'
        except Exception as error:
            log.error("ECMWF %s: extraction failed: %r\n%s", t.strftime('%Y-%m-%d'), error, traceback.format_exc())
            return 'error'
        if nwps is None:
            log.warning('ECMWF %s: no grib data under %s', t.strftime('%Y-%m-%d'), self.path_nwp)
            return 'missing'
        if self.path_group_nwp is None:
            return nwps
        os.makedirs(self.path_group_nwp, exist_ok=True)
        file = self.pickle_file(t)
        joblib.dump(nwps, file + '.tmp', compress=self.compress)
        os.replace(file + '.tmp', file)
        log.info('ECMWF %s: %s (%.1f MB)', t.strftime('%Y-%m-%d'), os.path.basename(file), os.path.getsize(file) / 1e6)
        return 'done'
