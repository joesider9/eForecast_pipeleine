# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The whole picture in a terminal: what is training, where each trial is, what is left, what runs.

``bin/eforecast trials`` is one table. This is everything the dashboard *eforecast training*
shows, as text, from the same three readers and with nothing between them and the machine:

* the machine in a line -- processors in use, memory, each card's load, memory and heat;
* the trials training now, with the loop each one is in (``trial_progress``);
* every case's studies: trials complete, pruned, failed, running and left, the studies that are
  not done, and the trials that finished last (``study_progress``);
* the processes of the pipeline and the heaviest others (``process_watch``).

    bin/eforecast status                  once
    bin/eforecast status --watch 10       again every 10 seconds, until Ctrl-C
    bin/eforecast status --case Total_wind
"""
from __future__ import annotations

import os
import sys
import time

from eforecast.config import process_watch, study_progress, trial_progress


def machine_line(before, after, span, cards):
    """Processors in use over the pause, memory, and each card."""
    busy = sum(after['cpu_seconds'][mode] - before['cpu_seconds'][mode] for mode in after['cpu_seconds'] if mode != 'idle')
    memory = after['memory']
    parts = [f"processors {max(0.0, busy) / max(span, 1e-6):.1f} of {after['cpus']} in use",
             f"memory {memory['used'] / 2 ** 30:.0f} of {memory['total'] / 2 ** 30:.0f} GB"]
    for card in cards:
        text = f"card {card['gpu']}"
        if 'load' in card:
            text += f" {card['load']:.0f} %"
        if 'memory_used' in card and card.get('memory_total'):
            text += f", {card['memory_used'] / 1024:.1f} of {card['memory_total'] / 1024:.0f} GB"
        if 'temperature' in card:
            text += f", {card['temperature']:.0f} C"
        if card.get('slowdown'):
            text += ', SLOWING DOWN FOR HEAT'
        parts.append(text)
    return '; '.join(parts)


def text(reader=None, case=None, pause=1.0, every=False):
    """Everything, as one text. ``reader`` is the ``study_progress.Reader`` of a workspace, or None."""
    before, began = process_watch.machine(), time.time()
    rows, cores = process_watch.measured(pause)
    after, span = process_watch.machine(), time.time() - began
    trials = [trial for trial in trial_progress.running_trials() if case in (None, trial['case'])]
    out = [time.strftime('%Y-%m-%d %H:%M:%S') + '   ' + machine_line(before, after, span, process_watch.cards()), '',
           'TRIALS TRAINING NOW', trial_progress.table(trials), '']
    out.append('STUDIES')
    if reader is None:
        out.append('no workspace is set (EFORECAST_WORKSPACE), so the studies are not read')
    else:
        try:
            out.append(study_progress.table(reader.read(), case, every=every))
        except Exception as error:            # noqa: BLE001 -- the rest of the picture is still worth showing
            out.append(f'the studies could not be read ({type(error).__name__}: {error})')
    wanted = [row for row in rows if case is None or row['case'] in ('', case)]
    out += ['', 'PROCESSES', process_watch.table(wanted, cores)]
    return '\n'.join(out)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--workspace', default=os.environ.get('EFORECAST_WORKSPACE'))
    parser.add_argument('--case', help='one case only')
    parser.add_argument('--all', action='store_true', dest='every', help='also the studies that are done')
    parser.add_argument('--watch', type=float, metavar='SECONDS', help='print again every SECONDS until stopped')
    arguments = parser.parse_args(argv)
    reader = study_progress.Reader(arguments.workspace) if arguments.workspace else None
    while True:
        shown = text(reader, arguments.case, every=arguments.every)
        if arguments.watch:
            sys.stdout.write('\033[2J\033[H')
        print(shown)
        if not arguments.watch:
            return 0
        try:
            time.sleep(max(0.0, arguments.watch - 1.0))
        except KeyboardInterrupt:
            return 0


if __name__ == '__main__':
    raise SystemExit(main())
