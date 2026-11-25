# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Convert daily NWP pickles of a group folder to the compact format 2.

    python -m eforecast.nwp_extraction.convert_nwp_pickles <path_group_nwp> [--max-step-hours 60] [--dry-run]

Format 2 keeps float32 arrays, the grid once per file and the forecast steps up
to ``max_step_hours`` after each run's analysis time, joblib-compressed.  Files
already in format 2 and symbolic links are left alone.  Each file is written to
a temporary name and swapped in, so an interrupted run leaves no half file.
"""
import argparse
import logging
import os
import re
import sys
import time

import joblib
import pandas as pd

from eforecast.nwp_extraction.nwp_pickles import PICKLE_FORMAT, compact_pickle

log = logging.getLogger('convert_nwp_pickles')
NAME = re.compile(r'^(?P<model>[a-z]+)_(?P<day>\d{6})\.pickle$')


def convert_folder(folder, max_step_hours=60, compress=3, dry_run=False):
    from eforecast.compute_context import active_worker, dataset_backend
    if active_worker.get() is None:
        from eforecast.config.weather_compute import convert
        return convert(folder, max_step_hours, compress, dry_run, workers=dataset_backend.get())
    names = sorted(name for name in os.listdir(folder) if NAME.match(name))
    before = after = 0
    converted = skipped = 0
    start = time.time()
    for name in names:
        file = os.path.join(folder, name)
        if os.path.islink(file):
            skipped += 1
            continue
        size = os.path.getsize(file)
        nwps = joblib.load(file)
        if nwps.get('format') == PICKLE_FORMAT:
            skipped += 1
            continue
        day = pd.to_datetime(NAME.match(name).group('day'), format='%d%m%y')
        compact = compact_pickle(nwps, day, max_step_hours=max_step_hours)
        if dry_run:
            log.info('%s: %.1f MB, would convert', name, size / 1e6)
            before += size
            continue
        joblib.dump(compact, file + '.tmp', compress=compress)
        os.replace(file + '.tmp', file)
        new_size = os.path.getsize(file)
        before += size
        after += new_size
        converted += 1
        log.info('%s: %.1f MB -> %.1f MB', name, size / 1e6, new_size / 1e6)
    log.info('%d files converted, %d skipped (format 2 or symlink), %.1f GB -> %.1f GB in %.0fs',
             converted, skipped, before / 1e9, after / 1e9, time.time() - start)
    return converted, skipped, before, after


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', help='group NWP folder holding ecmwf_<ddmmyy>.pickle files')
    parser.add_argument('--max-step-hours', type=int, default=60)
    parser.add_argument('--compress', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    convert_folder(args.folder, args.max_step_hours, args.compress, args.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
