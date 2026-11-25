# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Which processes of the pipeline run on this machine, what each one is, and what it uses.

``bin/eforecast trials`` answers for the trials. This answers for everything around them: the
training that started them, their data loaders, a combination or a dataset run and its workers,
the pages, the services, the daemons that share a card -- each with the case it works for, its
processor time, memory, threads, reads and writes, and its memory on a card. Beside them the
heaviest other processes of the machine, so that what else is using it is not a guess, and the
machine itself: processors by what they do, load, memory, and whether a card is slowing itself
down for heat.

A process is told by its command line, never by a registry it would have to write to: it is
found whatever started it and whatever version of the code it runs.

The metrics service serves this at ``/processes`` (Prometheus job ``processes``);
``bin/eforecast status`` prints it. Processor time is a counter: a rate over it is the number of
processors in use.
"""
from __future__ import annotations

import os
import subprocess
import time

from eforecast.config import trial_progress

#: The entry points of a run, and what a person calls each.
RUNS = {'run_training.py': 'training', 'run_combine.py': 'combine', 'run_datasets.py': 'datasets',
        'run_site_graph.py': 'site graph', 'run_model_script.py': 'forecast', 'plan_revision.py': 'plan'}
#: Modules started with ``python -m``.
MODULES = {'eforecast.config.serve': 'web pages', 'eforecast.config.run_coordinator': 'run coordinator',
           'eforecast.config.metrics_server': 'metrics service', 'eforecast.config.run_supervisor': 'worker supervisor',
           'eforecast.config.trial_progress': 'watch', 'eforecast.config.process_watch': 'watch',
           'eforecast.config.study_progress': 'watch', 'eforecast.config.status': 'watch'}
LABELS = ('role', 'case', 'cluster', 'method', 'trial', 'name', 'pid')
#: The order a person reads them in: what a run is, then what serves it, then everything else.
ORDER = ('training', 'trial', 'data loader', 'combine', 'datasets', 'site graph', 'forecast', 'plan', 'worker supervisor',
         'worker', 'card sharing', 'pipeline', 'web pages', 'run coordinator', 'metrics service', 'gpu watchdog', 'watch',
         'other')
FAMILIES = (
    ('eforecast_process_cpu_seconds_total', 'counter', 'Processor time the process has used (user and system).'),
    ('eforecast_process_memory_bytes', 'gauge', 'Memory the process holds (resident).'),
    ('eforecast_process_threads', 'gauge', 'Threads of the process.'),
    ('eforecast_process_gpu_memory_bytes', 'gauge', "The process's memory on the cards."),
    ('eforecast_process_read_bytes_total', 'counter', 'Bytes the process has read from storage.'),
    ('eforecast_process_write_bytes_total', 'counter', 'Bytes the process has written to storage.'),
    ('eforecast_process_start_time_seconds', 'gauge', 'When the process started (Unix time).'),
    ('eforecast_machine_cpu_seconds_total', 'counter', 'Processor time of the whole machine, by what it was spent on.'),
    ('eforecast_machine_cpus', 'gauge', 'Processors of the machine.'),
    ('eforecast_machine_load', 'gauge', 'Load average: processes running or waiting to, over the window.'),
    ('eforecast_machine_memory_bytes', 'gauge', 'Memory of the machine: total, used, available, and swap.'),
    ('eforecast_gpu_thermal_slowdown', 'gauge', '1 while the card slows itself down for heat.'),
    ('eforecast_gpu_thermal_slowdown_seconds_total', 'counter', 'How long the card has slowed itself down for heat.'),
)
#: Other processes shown beside the pipeline's: the heaviest by processor and by memory.
OTHERS = 8
_PREVIOUS = {'at': 0.0, 'cpu': {}}
_CARDS = {'at': 0.0, 'rows': []}


def script_of(argv):
    """The script or module a command line runs: ``('run_training.py', path)`` or ``('-m', module)``."""
    for at, argument in enumerate(argv[1:4], start=1):
        if argument == '-m' and at + 1 < len(argv):
            return '-m', argv[at + 1]
        if argument == '-c':
            return '-c', argv[at + 1] if at + 1 < len(argv) else ''
        if not argument.startswith('-'):
            return os.path.basename(argument), argument
    return (os.path.basename(argv[0]), argv[0]) if argv else ('', '')


def classify(name, argv):
    """``(role, what)`` of a pipeline process, or ``(None, name)`` for any other.

    ``what`` is the short name a person would use for it: the script, the module, the program.
    """
    argv = [str(argument) for argument in argv or ()]
    if any(argument.endswith(trial_progress.RUNNER) for argument in argv) and '--path-weights' in argv[:-1]:
        return 'trial', trial_progress.RUNNER
    kind, target = script_of(argv)
    if kind in RUNS:
        return RUNS[kind], kind
    if kind == '-m':
        if target in MODULES:
            return MODULES[target], target.rsplit('.', 1)[-1]
        if target.startswith(('eforecast.', 'eforecast_online')):
            return 'pipeline', target.rsplit('.', 1)[-1]
        return None, name
    if kind == '-c':                               # code given on the command line: a worker when python runs ours
        return ('worker', 'worker') if name.startswith('python') and 'eforecast' in target else (None, name)
    if name.startswith('nvidia-cuda-mps'):
        return 'card sharing', name
    if 'eforecast' in target:                      # a script of the installation: a watchdog, a chain
        return ('gpu watchdog' if 'gpu-lost-reboot' in target else 'pipeline'), kind
    return None, name


def case_of(argv):
    """The case a command line names with ``--config``, or ''."""
    argv = list(argv or ())
    if '--config' in argv[:-1]:
        return os.path.basename(os.path.normpath(str(argv[argv.index('--config') + 1])))
    return ''


def snapshot(others=OTHERS, now=None):
    """Every pipeline process and the heaviest others, as dictionaries with what they use.

    A process that belongs to a run -- a trial, a data loader, a worker, a card daemon -- carries
    the case of the nearest process above it that names one.
    """
    import psutil

    moment = time.time() if now is None else now
    listed = {}
    for process in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'create_time', 'cpu_times', 'memory_info',
                                        'num_threads']):
        info = process.info
        if info.get('cpu_times') is None or info.get('memory_info') is None:
            continue                                # it ended while it was being listed, or is not ours to read
        listed[info['pid']] = (process, info)
    cards = trial_progress.card_memory()
    rows, rest = [], []
    for pid, (process, info) in listed.items():
        argv = info.get('cmdline') or []
        role, what = classify(info.get('name') or '', argv)
        row = {'pid': pid, 'ppid': info.get('ppid'), 'role': role or 'other', 'name': what or info.get('name') or '?',
               'case': case_of(argv), 'cluster': '', 'method': '', 'trial': '',
               'cpu_seconds': info['cpu_times'].user + info['cpu_times'].system, 'memory': info['memory_info'].rss,
               'threads': info.get('num_threads') or 0, 'started': info.get('create_time') or moment,
               'gpu_memory': cards.get(pid, 0)}
        if role == 'trial':
            folder = argv[argv.index('--path-weights') + 1]
            row['cluster'], row['method'], row['trial'] = trial_progress.describe(folder)
            above = listed.get(info.get('ppid'))
            if above and classify(above[1].get('name') or '', above[1].get('cmdline') or [])[0] == 'trial':
                row['role'] = 'data loader'         # a forked copy of the trial, reading its batches
        (rows if role else rest).append(row)
        if role:
            try:
                counters = process.io_counters()
                row['read_bytes'], row['write_bytes'] = counters.read_bytes, counters.write_bytes
            except Exception:                       # noqa: BLE001 -- another user's process, or one that ended
                pass
    by_pid = {row['pid']: row for row in rows}
    for row in rows:                                # the case of a run is the case of what it started
        above, steps = row, 0
        while not row['case'] and above and steps < 8:
            above = by_pid.get(above['ppid']) or _row_of(listed.get(above['ppid']))
            row['case'] = (above or {}).get('case', '')
            steps += 1
    # the heaviest of the rest: by the processors they use now, and by the memory they hold
    span = moment - _PREVIOUS['at']
    for row in rest:
        earlier = _PREVIOUS['cpu'].get(row['pid'])
        row['cores'] = max(0.0, row['cpu_seconds'] - earlier) / span if earlier is not None and span > 0 else 0.0
    chosen = {row['pid']: row for row in sorted(rest, key=lambda r: r['cores'], reverse=True)[:others] if row['cores'] > 0.005}
    chosen.update({row['pid']: row for row in sorted(rest, key=lambda r: r['memory'], reverse=True)[:max(1, others // 2)]})
    _PREVIOUS.update(at=moment, cpu={pid: info['cpu_times'].user + info['cpu_times'].system
                                     for pid, (_, info) in listed.items()})
    def place(row):                                 # the runs of a case first, case by case; then the services
        role = ORDER.index(row['role']) if row['role'] in ORDER else len(ORDER)
        return (not row['case'], row['case'], role, row['cluster'], row['method'], row['trial'], row['pid'])
    return sorted(rows, key=place) + sorted(chosen.values(), key=lambda r: r['memory'], reverse=True)


def _row_of(entry):
    """A process that is not the pipeline's own, as far as its case goes (a shell that named one)."""
    if entry is None:
        return None
    return {'ppid': entry[1].get('ppid'), 'case': case_of(entry[1].get('cmdline') or [])}


def machine():
    """Processors by what they do, load, and memory of the whole machine."""
    import psutil

    times = psutil.cpu_times()
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    busy = {'user': times.user + getattr(times, 'nice', 0.0),
            'system': times.system + getattr(times, 'irq', 0.0) + getattr(times, 'softirq', 0.0),
            'waiting for storage': getattr(times, 'iowait', 0.0), 'idle': times.idle}
    try:
        load = dict(zip(('1m', '5m', '15m'), os.getloadavg()))
    except OSError:
        load = {}
    return {'cpu_seconds': busy, 'cpus': psutil.cpu_count() or 0, 'load': load,
            'memory': {'total': memory.total, 'used': memory.total - memory.available, 'available': memory.available,
                       'swap used': swap.used, 'swap total': swap.total}}


def cards():
    """Each card: its number, and whether and for how long it has slowed itself down for heat.

    The exporter of the cards (DCGM) has their load, memory, temperature and power; it does not
    have this, which is what card 0 of this machine did before it was lost in November 2026.
    Asked every ten seconds at most, and never for longer than eight: a lost card makes the
    driver's own tool hang.
    """
    if time.time() - _CARDS['at'] > 10:
        rows = []
        try:
            answer = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,clocks_event_reasons.hw_thermal_slowdown,'
                 'clocks_event_reasons_counters.hw_thermal_slowdown,utilization.gpu,memory.used,memory.total,'
                 'temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=8).stdout
            for line in answer.splitlines():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 3 and parts[0].isdigit():
                    row = {'gpu': parts[0], 'slowdown': 1 if parts[1].lower() == 'active' else 0,
                           'slowdown_seconds': int(parts[2]) / 1e6 if parts[2].isdigit() else None}
                    # for a terminal; Prometheus has these from the cards' own exporter
                    for name, text in zip(('load', 'memory_used', 'memory_total', 'temperature', 'power'), parts[3:]):
                        try:
                            row[name] = float(text)
                        except ValueError:
                            pass
                    rows.append(row)
        except (OSError, subprocess.SubprocessError):
            rows = []
        _CARDS.update(at=time.time(), rows=rows)
    return _CARDS['rows']


def _label(value):
    return str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


def _number(value):
    return repr(float(value))


def render(rows, totals=None, gpus=None):
    """The processes, the machine and the cards in Prometheus's text format."""
    samples = {name: [] for name, _, _ in FAMILIES}
    for row in rows:
        labels = ','.join(f'{key}="{_label(row[key])}"' for key in LABELS)
        for family, key in (('eforecast_process_cpu_seconds_total', 'cpu_seconds'), ('eforecast_process_memory_bytes', 'memory'),
                            ('eforecast_process_threads', 'threads'), ('eforecast_process_start_time_seconds', 'started'),
                            ('eforecast_process_read_bytes_total', 'read_bytes'),
                            ('eforecast_process_write_bytes_total', 'write_bytes')):
            if row.get(key) is not None:
                samples[family].append(f'{{{labels}}} {_number(row[key])}')
        if row.get('gpu_memory'):
            samples['eforecast_process_gpu_memory_bytes'].append(f'{{{labels}}} {_number(row["gpu_memory"])}')
    if totals:
        samples['eforecast_machine_cpu_seconds_total'] += [f'{{mode="{mode}"}} {_number(value)}'
                                                           for mode, value in totals['cpu_seconds'].items()]
        samples['eforecast_machine_cpus'].append(f' {_number(totals["cpus"])}')
        samples['eforecast_machine_load'] += [f'{{window="{window}"}} {_number(value)}'
                                              for window, value in totals['load'].items()]
        samples['eforecast_machine_memory_bytes'] += [f'{{kind="{kind}"}} {_number(value)}'
                                                      for kind, value in totals['memory'].items()]
    for card in gpus or ():
        samples['eforecast_gpu_thermal_slowdown'].append(f'{{gpu="{_label(card["gpu"])}"}} {card["slowdown"]}')
        if card.get('slowdown_seconds') is not None:
            samples['eforecast_gpu_thermal_slowdown_seconds_total'].append(
                f'{{gpu="{_label(card["gpu"])}"}} {_number(card["slowdown_seconds"])}')
    lines = []
    for name, kind, text in FAMILIES:
        lines += [f'# HELP {name} {text}', f'# TYPE {name} {kind}'] + [name + sample for sample in samples[name]]
    return '\n'.join(lines) + '\n'


def _size(value):
    for unit, size in (('GB', 2 ** 30), ('MB', 2 ** 20)):
        if value >= size:
            return f'{value / size:.1f} {unit}'
    return f'{value / 1024:.0f} kB' if value else ''


def table(rows, cores=None, now=None):
    """The processes as a table for a terminal. ``cores`` is ``{pid: processors in use now}``."""
    moment = time.time() if now is None else now
    if not rows:
        return 'no process of the pipeline is running on this machine'
    header = ('role', 'case', 'what', 'pid', 'cpus', 'memory', 'on card', 'threads', 'running')
    groups = {}                                     # a script and the copies of itself it forked: one line
    for row in rows:
        what = ' '.join(part for part in (row['cluster'], row['method'], row['trial']) if part) or row['name']
        groups.setdefault((row['role'], row['case'], what), []).append(row)
    lines = []
    for (role, case, what), members in groups.items():
        used = [(cores or {}).get(row['pid'], row.get('cores')) for row in members]
        used = [value for value in used if value is not None]
        pid = str(members[0]['pid']) + (f' +{len(members) - 1}' if len(members) > 1 else '')
        lines.append((role, case, what, pid, f'{sum(used):.2f}' if used else '',
                      _size(sum(row['memory'] for row in members)), _size(sum(row['gpu_memory'] for row in members)),
                      str(sum(row['threads'] for row in members)),
                      trial_progress._duration(moment - min(row['started'] for row in members))))
    widths = [max(len(line[column]) for line in (header, *lines)) for column in range(len(header))]
    return '\n'.join('  '.join(cell.ljust(width) if column < 3 else cell.rjust(width)
                               for column, (cell, width) in enumerate(zip(line, widths))).rstrip()
                     for line in (header, *lines))


def measured(pause=1.0):
    """``(rows, {pid: processors in use})``: two looks, ``pause`` seconds apart, for a terminal."""
    first = {row['pid']: row['cpu_seconds'] for row in snapshot()}
    began = time.time()
    time.sleep(pause)
    rows = snapshot()
    span = max(1e-6, time.time() - began)
    return rows, {row['pid']: max(0.0, row['cpu_seconds'] - first[row['pid']]) / span
                  for row in rows if row['pid'] in first}


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--prometheus', action='store_true', help="print Prometheus's text format instead")
    parser.add_argument('--watch', type=float, metavar='SECONDS', help='print again every SECONDS until stopped')
    arguments = parser.parse_args(argv)
    while True:
        if arguments.prometheus:
            text = render(snapshot(), machine(), cards())
        else:
            rows, cores = measured(min(1.0, arguments.watch or 1.0))
            text = table(rows, cores)
        if arguments.watch:
            sys.stdout.write('\033[2J\033[H' + time.strftime('%H:%M:%S') + '\n')
        print(text)
        if not arguments.watch:
            return 0
        try:
            time.sleep(arguments.watch)
        except KeyboardInterrupt:
            return 0


if __name__ == '__main__':
    raise SystemExit(main())
