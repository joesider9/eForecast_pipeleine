# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""ECMWF source layout, shared by extraction and read-only stage lineage."""
import re
from pathlib import Path

#: ``..._<analysis>T<HHMMSS>Z_<valid>T<HHMMSS>Z_<step>h``: the per-step provider naming.
STEP_NAME = re.compile(r'_\d{8}T(\d{2})(\d{2})\d{2}Z_\d{8}T\d{6}Z_(\d+)h$')
DAY_NAME = re.compile(r'\d{8}')

#: Steps further ahead than this are never read: the horizons a case forecasts are inside it,
#: and reading the long tail of every cycle would multiply the extraction for nothing. One fact,
#: shared by the extractor that applies it and by the bookkeeping that must agree with it.
MAX_STEP_HOURS = 60


def roots(path):
    path = Path(str(path))  # same normalization as the existing extractor, including legacy unset roots
    return [path, path / 'new/data', path / 'ECMWF/new/data', path / 'ECMWF']


def day_folder(candidates, day):
    return next((str(Path(root) / day.strftime('%Y%m%d')) for root in candidates
                 if (Path(root) / day.strftime('%Y%m%d')).is_dir()), None)


def grib_files(folder):
    folder = Path(folder)
    single = sorted(p for p in folder.iterdir() if p.name.startswith('NWP_') and p.name.endswith('.grb'))
    return [str(p) for p in single] if single else [str(p) for run in sorted(folder.iterdir()) if run.is_dir()
                                                   for p in sorted(run.iterdir()) if p.is_file()]


def day_of(path):
    """``(day folder, path inside it)`` of one raw file, or ``(None, None)``."""
    path = Path(path)
    for parent in path.parents:
        if DAY_NAME.fullmatch(parent.name):
            return parent, path.relative_to(parent)
    return None, None


def forecast_step(path):
    """``(analysis cycle, step hours)`` of a per-step file, or ``None`` when its name does not say.

    The cycle comes from the name, not from the folder the file sits in: a day folder can hold a
    misfiled copy of another cycle, and the file is the authority on what it contains.
    """
    found = STEP_NAME.search(Path(path).name)
    return (f'{found.group(1)}{found.group(2)}', int(found.group(3))) if found else None


def holds_weather(folder, *, max_step_hours=MAX_STEP_HOURS):
    """Whether a day folder holds a file the extractor would actually read.

    A zero-byte payload is a name without weather in it, and a step past the horizon is never
    opened, so a day of nothing but those cannot become a pickle however often it is tried. A
    file whose name does not state its step is left for the extractor to judge and counts as
    weather here.
    """
    try:
        usable, _ = readable(grib_files(folder))
    except OSError:
        return False
    return any((forecast_step(path) or (None, 0))[1] <= max_step_hours for path in usable)


def readable(files):
    """``(usable, empty)``. A missing or zero-byte payload is a name without weather in it."""
    usable, empty = [], []
    for path in files:
        try:
            keep = Path(path).is_file() and Path(path).stat().st_size > 0
        except OSError:
            keep = False
        (usable if keep else empty).append(str(path))
    return usable, empty
