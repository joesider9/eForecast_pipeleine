# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Admitted conversion target; source weather files are never opened for writing."""
import os
from pathlib import Path

import joblib
import pandas as pd

from eforecast.compute_worker import current_worker
from eforecast.nwp_extraction.nwp_pickles import PICKLE_FORMAT, compact_pickle
from eforecast.nwp_extraction.convert_nwp_pickles import NAME


def observation(path):
    value = Path(path).lstat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def execute(payload, context):
    if current_worker() != context:
        raise RuntimeError('weather conversion requires confirmed attachment')
    output = Path(context.result_directory) / 'artifacts'
    output.mkdir()
    converted = skipped = before = after = 0
    for name, expected in payload['sources'].items():
        source = Path(payload['folder']) / name
        if source.is_symlink() or observation(source) != expected:
            raise ValueError('weather source changed before conversion')
        data = joblib.load(source)
        if data.get('format') == PICKLE_FORMAT:
            skipped += 1
            continue
        day = pd.to_datetime(NAME.match(name).group('day'), format='%d%m%y')
        compact = compact_pickle(data, day, max_step_hours=payload['max_step_hours'])
        if observation(source) != expected:
            raise ValueError('weather source changed during conversion')
        before += expected[2]
        if not payload['dry_run']:
            target = output / name
            joblib.dump(compact, target, compress=payload['compress'])
            after += target.stat().st_size
            converted += 1
        del data, compact
    return {'counts': [converted, skipped + payload['symlinks'], before, after],
            'artifacts': [p.relative_to(output.parent).as_posix() for p in output.iterdir()]}
