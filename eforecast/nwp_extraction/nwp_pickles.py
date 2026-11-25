# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Daily NWP pickle formats (no grib dependencies, importable from the readers).

* old format: ``{'0000': {hour key: {var: float64 array, 'lat', 'long'}}, '1200': {...}}``
* format 2:   ``{'format': 2, 'lat', 'long', 'variables', 'max_step_hours',
               '0000': {hour key: {var: float32 array}}, '1200': {...}}``
"""
import numpy as np
import pandas as pd

# our variable name -> ECMWF short names it is built from
NAMES = {'Uwind': ['u100'], 'Vwind': ['v100'], 'WS': ['u100', 'v100'], 'WD': ['u100', 'v100'],
         'Uwind10': ['u10'], 'Vwind10': ['v10'], 'WS10': ['u10', 'v10'], 'WD10': ['u10', 'v10'],
         'Temperature': ['t2m'], 'Cloud': ['tcc'], 'High_Cloud': ['hcc'], 'Low_Cloud': ['lcc'],
         'Medium_Cloud': ['mcc'], 'Flux': ['ssrd'], 'Pressure': ['sp'], 'Humidity': ['sh2'],
         'Clear_Sky_Flux': ['ssrdc'], 'WindGust': ['fg10_3']}
ACCUMULATED = {'ssrd', 'ssrdc'}          # de-accumulated step by step
NON_NEGATIVE = {'Flux', 'Clear_Sky_Flux'}  # accumulated radiation: grib packing noise can make the differences slightly negative
DIRECT = {name: fields[0] for name, fields in NAMES.items() if len(fields) == 1}
PICKLE_FORMAT = 2
NON_RUN_KEYS = {'format', 'lat', 'long', 'variables', 'max_step_hours'}


def analysis_time(day, run):
    """The analysis timestamp of a run key (``'0000'``, ``'1200'``) of ``day``."""
    return pd.Timestamp(day).normalize() + pd.DateOffset(hours=int(run[:2]), minutes=int(run[2:]))


def runs_of(nwps, day):
    """``{analysis timestamp: {valid key: {var: array, 'lat', 'long'}}}`` of one daily pickle.

    :func:`merge_runs` collapses the cycles -- a later analysis overwrites an earlier one for the
    same valid time -- and that is what let a 12:00 UTC analysis answer an issue at 09:00 UTC. The
    pickle keeps the runs apart, so the provenance only has to be carried instead of discarded.
    """
    format_2 = nwps.get('format') == PICKLE_FORMAT
    lat, long = (nwps.get('lat'), nwps.get('long')) if format_2 else (None, None)
    out = {}
    for run in sorted(key for key in nwps if key not in NON_RUN_KEYS):
        hours = {key: dict(values) for key, values in nwps[run].items()}
        if format_2:
            for values in hours.values():
                values['lat'], values['long'] = lat, long
        out[analysis_time(day, run)] = hours
    return out


def available_analyses(runs, as_of, availability=None):
    """The analyses of ``runs`` that exist by ``as_of``, newest first.

    ``availability`` is how long after its analysis time a cycle can actually be read. Zero keeps
    only the necessary constraint -- an analysis cannot answer an issue before it was made -- and
    is deliberately optimistic: a 00:00 UTC run is not published at 00:00 UTC. Declare the real
    delay of the provider to close that gap.
    """
    availability = pd.Timedelta(0, 'ns') if availability is None else availability
    return [analysis for analysis in sorted(runs, reverse=True)
            if analysis + availability <= pd.Timestamp(as_of)]


def merge_runs(nwps):
    """Hours of one daily pickle (either format) as ``{hour key: {var: array, 'lat', 'long'}}``.

    Format 2 stores the grid once and the runs under ``'0000'`` / ``'1200'``;
    the older format stores ``lat`` / ``long`` inside every hour.  Later runs
    override earlier ones, as ``nwps['0000'].update(nwps['1200'])`` did.
    """
    if nwps.get('format') == PICKLE_FORMAT:
        lat, long = nwps['lat'], nwps['long']
        merged = dict()
        for run in sorted(key for key in nwps if key not in NON_RUN_KEYS):
            merged.update(nwps[run])
        for hour in merged.values():
            hour['lat'] = lat
            hour['long'] = long
        return merged
    runs = sorted(nwps.keys())
    merged = dict()
    for run in runs:
        merged.update(nwps[run])
    return merged


def compact_pickle(nwps, day, max_step_hours=60):
    """Format 2 of an old-format daily pickle: float32, grid once, steps up to ``max_step_hours``."""
    lat = long = None
    out = {'format': PICKLE_FORMAT, 'max_step_hours': max_step_hours}
    variables = set()
    for run, hours in nwps.items():
        if run in NON_RUN_KEYS:
            continue
        analysis = pd.Timestamp(day).normalize() + pd.DateOffset(hours=int(run[:2]), minutes=int(run[2:]))
        out[run] = dict()
        for key, arrays in hours.items():
            valid = pd.to_datetime(key, format='%d%m%y%H%M')
            if max_step_hours is not None and (valid - analysis) > pd.Timedelta(max_step_hours, 'h'):
                continue
            if lat is None and 'lat' in arrays:
                lat, long = arrays['lat'], arrays['long']
            out[run][key] = {name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()
                             if name not in ('lat', 'long')}
            variables.update(out[run][key].keys())
    out['lat'], out['long'], out['variables'] = lat, long, sorted(variables)
    return out


