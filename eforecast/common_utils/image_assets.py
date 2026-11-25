# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Lightweight identities for the processed pixels read by raw-image networks.

Statistics-only models consume dataset_images.pickle, not these shared cache files.
Legacy caches remain observations until a producer has published their digests.
"""
from pathlib import Path


def uses_pixels(static, experiment=None):
    definitions = [experiment] if isinstance(experiment, dict) else (static.get('experiments') or {}).values()
    return any(('image' in tag or 'img' in tag) and 'stats' not in tag
               for definition in definitions for tag in definition)


def paths(static, *, dates=None):
    import joblib
    import pandas as pd
    if dates is None:
        value = joblib.load(Path(static['path_data']) / 'dataset_images.pickle')
        dates = value.get('dates')
        if not isinstance(dates, dict):
            raise ValueError('image dataset has no dates mapping')
    base = Path(static['path_image']) / 'processed'
    files = []
    for variable in static.get('variables', []):
        if variable.get('type') != 'image':
            continue
        api, name = variable['source'], variable['name']
        key = f'{api}_{name}'
        if key not in dates:
            raise ValueError(f'image dataset has no dates for {key}')
        files.extend(base / f'satellite_{api}_{name}_{day.strftime("%Y_%m_%d__%H_%M")}.pkl'
                     for day in pd.DatetimeIndex(dates[key]))
    return sorted(set(files))


def observations(static):
    result = {}
    for path in paths(static):
        try:
            info = path.stat()
            result[str(path)] = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
        except FileNotFoundError:
            result[str(path)] = None
    return result


def stamps(static):
    """``observations`` for a before/after guard: stamps that tell a moved change time apart
    from a rewritten file (``file_stamps``). The cache identities below keep the bare stat."""
    from . import file_stamps
    return {str(path): file_stamps.observe(path) for path in paths(static)}


def trial_uses_pixels(static, folder):
    if not static.get('use_image'):
        return False  # non-image workloads need no parameter decoding or shared-cache inventory
    import joblib
    file = Path(folder) / 'parameters.pickle'
    if not file.is_file():
        return False  # historical custom producer: no invented experiment provenance
    parameters = joblib.load(file)
    experiment = parameters.get('experiment') if isinstance(parameters, dict) else None
    return isinstance(experiment, dict) and uses_pixels(static, experiment)


def identities(static):
    from . import lineage
    base = Path(static['path_image'])
    # Include replacement/write observations even when a legacy sidecar's size/mtime still
    # matches. This is cheap per trial and invalidates prediction caches after restored mtimes.
    observed = observations(static)
    return {str(path.relative_to(base)): {'identity': lineage.file_identity(str(path)),
                                        'observation': observed[str(path)]} for path in paths(static)}
