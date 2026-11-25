# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Which trials are training on this machine now, their error after every epoch, and how fast.

A trial is a ``trial_runner.py`` process. Its command line names its folder (``--path-weights``),
where it writes ``progress.json`` after every epoch -- the epoch, the error of that epoch and the
best so far, on the validation and on the test set -- and its environment names its card
(``CUDA_VISIBLE_DEVICES``). Its output goes to ``train.log`` in the same folder, where every
training pass ends with ``Run training step with … iter/sec``: its loops (training batches) per
second. Read from the processes rather than from the training, this works for any training:
started from the pages, from a terminal, or by a chain pinned to an older version of the code,
which publishes none of it.

Inside an epoch (23 November 2026) the same log says where the trial is: the last progress bar
it drew -- batches done of how many, their rate now, the time that loop still needs -- and
whether that loop is a training pass. ``parameters.pickle`` says how many epochs it may run, and
the process itself what it uses: processor time, memory, and its memory on the card.

``bin/eforecast trials`` prints them; the metrics service serves them to Prometheus at
``/trials`` (job ``trials``), which the dashboard *eforecast training* draws.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RUNNER = 'trial_runner.py'
SETS = ('val', 'test')
LABELS = ('case', 'cluster', 'method', 'trial', 'gpu')
FAMILIES = (
    ('eforecast_trial_running_seconds', 'How long the trial has been running.'),
    ('eforecast_trial_epoch', 'The last epoch the trial finished.'),
    ('eforecast_trial_epochs_total', 'The epochs the trial may run at most (it may stop earlier).'),
    ('eforecast_trial_error', 'The error of that epoch, on the validation or the test set.'),
    ('eforecast_trial_best_error', 'The best error of the trial so far, on the validation or the test set.'),
    ('eforecast_trial_last_epoch_age_seconds', 'How long ago the trial finished its last epoch.'),
    ('eforecast_trial_loops_per_second', "Training batches per second in the trial's last training pass."),
    ('eforecast_trial_epochs_per_second', 'Epochs per second: of the last epoch, or since the trial began.'),
    ('eforecast_trial_loop_done', 'Batches done in the loop the trial is in now.'),
    ('eforecast_trial_loop_total', 'Batches of the loop the trial is in now.'),
    ('eforecast_trial_loop_rate', 'Batches per second of that loop now.'),
    ('eforecast_trial_loop_left_seconds', 'Seconds that loop still needs, by its own estimate.'),
    ('eforecast_trial_loop_training', '1 while that loop is a training pass, 0 while it is another (validation, test).'),
    ('eforecast_trial_log_age_seconds', 'How long ago the trial last wrote to its log: a trial that hangs stops writing.'),
    ('eforecast_trial_cpu_seconds_total', 'Processor time of the trial and its data loaders.'),
    ('eforecast_trial_memory_bytes', 'Memory (resident) of the trial and its data loaders.'),
    ('eforecast_trial_gpu_memory_bytes', "The trial's memory on its card."),
)
#: The families that only grow; the others are read as they are.
COUNTERS = frozenset({'eforecast_trial_cpu_seconds_total'})
_CARDS = {'at': 0.0, 'index': {}}

LOG = 'train.log'
# The line every training pass ends with (eforecast/deep_models/*/trainer.py), in every version of
# the code: the pass's batches over its seconds, or its seconds per batch when that is over one.
RATE = b'Run training step with '
RATE_LINE = re.compile(rb'Run training step with ([0-9.]+(?:[eE][+-]?[0-9]+)?)(iter/sec|sec/iter)')
# After either of these a rate further up the log belongs to an earlier fit or attempt: each fit
# begins with ``Start training of …`` (network.py), each attempt with the orchestrator's header --
# not always at the start of a line, since an attempt that ended in the middle of a bar left none.
FIT_START = b'Start training of '
ATTEMPT = re.compile(rb'# \d{4}-\d\d-\d\d \d\d:\d\d:\d\d gpu \S+ attempt \d+ ')
# A progress bar as tqdm writes it:  `` 84%|████████▍ | 376/446 [05:51<01:07,  1.04it/s]``. The bar
# of a loop over batches is drawn below the bar of the epochs and ends with "cursor up"; the bar of
# the epochs does not. A bar that has just begun has no rate yet (``?it/s``).
BAR = re.compile(rb'(\d+)%\|[^|\r\n]*\| *(\d+)/(\d+) \[([0-9:]+)<([0-9:?]+), +([0-9.?]+)(it/s|s/it)\]((?:\x1b\[A)?)')
_EPOCHS = {}
_CARD_MEMORY = {'at': 0.0, 'bytes': {}}


def read_progress(path):
    """The progress.json a trial writes after every epoch, or None before its first one."""
    try:
        with open(path, encoding='utf-8') as stream:
            progress = json.load(stream)
        progress['epoch'] = int(progress['epoch'])
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return progress


def loops_per_second(folder, window=1 << 20, limit=16 << 20):
    """Training batches per second of the trial's last training pass, from its ``train.log``; None
    before the first pass of its current fit, or without a log.

    Only the end of the log is read, a megabyte first and more while nothing is found there: tqdm's
    bars add up to about a kilobyte a second between two such lines. The line comes from a
    ``print``, which the trial buffers, but tqdm flushes it when the validation bar opens.
    """
    try:
        with open(os.path.join(folder, LOG), 'rb') as stream:
            size = stream.seek(0, os.SEEK_END)
            span = window
            while True:
                begin = max(0, size - span)
                stream.seek(begin)
                text = stream.read(size - begin)
                at = text.rfind(RATE)
                since = max(at, 0)
                if text.find(FIT_START, since) >= 0 or ATTEMPT.search(text, since):
                    return None          # a new fit or attempt has begun, and has not finished a pass yet
                if at >= 0:
                    match = RATE_LINE.match(text, at)
                    return _loop_rate(match.group(1), match.group(2)) if match else None
                if begin == 0 or span >= limit:
                    return None
                span *= 4
    except OSError:
        return None


def _seconds(text):
    """``05:51`` or ``1:05:51`` as seconds; None for the ``?`` of a bar that cannot say yet."""
    try:
        parts = [int(part) for part in text.decode().split(':')]
    except ValueError:
        return None
    return sum(part * 60 ** at for at, part in enumerate(reversed(parts)))


def loop_progress(folder, window=1 << 20):
    """The loop the trial is in now, from the last progress bar of its ``train.log``.

    ``{'done', 'total', 'rate', 'elapsed', 'left', 'training'}``: batches done and in all, batches
    per second now, the seconds the loop has run and still needs, and whether it is a training
    pass -- one with as many batches as the pass that ended with the last ``Run training step``
    line; None while the trial has not finished a training pass to compare with. None before the
    first bar of the current fit, or without a log.
    """
    try:
        with open(os.path.join(folder, LOG), 'rb') as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - window))
            text = stream.read(window)
    except OSError:
        return None
    bars = [found for found in BAR.finditer(text) if found.group(8)] or list(BAR.finditer(text))
    if not bars:
        return None
    last = bars[-1]
    if text.find(FIT_START, last.end()) >= 0 or ATTEMPT.search(text, last.end()):
        return None                           # a new fit or attempt has begun and has drawn no bar yet
    _, done, total, elapsed, left, number, unit, _ = last.groups()
    rate = None if b'?' in number else _loop_rate(number, b'iter/sec' if unit == b'it/s' else b'sec/iter')
    training = None
    ended = text.rfind(RATE)
    if ended >= 0 and ended < last.start():
        before = [found for found in bars if found.end() <= ended]
        if before:
            training = int(before[-1].group(3)) == int(total)
    return {'done': int(done), 'total': int(total), 'rate': rate, 'elapsed': _seconds(elapsed),
            'left': _seconds(left), 'training': training}


def epochs_total(folder):
    """The epochs a trial may run at most: ``max_iterations`` of its ``parameters.pickle``. Read once."""
    if folder not in _EPOCHS:
        try:
            import joblib
            value = joblib.load(os.path.join(folder, 'parameters.pickle')).get('max_iterations')
            _EPOCHS[folder] = int(value) if _measured(value) and value > 0 else None
        except Exception:                # noqa: BLE001 -- another layout, or a file still being written
            return None
        if len(_EPOCHS) > 512:
            _EPOCHS.pop(next(iter(_EPOCHS)))
    return _EPOCHS[folder]


def log_age(folder, now=None):
    """Seconds since the trial last wrote to its log; None without one."""
    try:
        return max(0.0, (time.time() if now is None else now) - os.path.getmtime(os.path.join(folder, LOG)))
    except OSError:
        return None


def card_memory():
    """``{pid: bytes on a card}`` of the processes computing on the cards. Asked every few seconds."""
    if time.time() - _CARD_MEMORY['at'] > 5:
        try:
            answer = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=10).stdout
            found = {}
            for line in answer.splitlines():
                pid, _, used = line.partition(',')
                if pid.strip().isdigit() and used.strip().isdigit():
                    found[int(pid)] = found.get(int(pid), 0) + int(used) * 2 ** 20
            _CARD_MEMORY['bytes'] = found
        except (OSError, subprocess.SubprocessError):
            _CARD_MEMORY['bytes'] = {}
        _CARD_MEMORY['at'] = time.time()
    return _CARD_MEMORY['bytes']


def _loop_rate(number, unit):
    try:
        value = float(number)
    except ValueError:
        return None
    rate = value if unit == b'iter/sec' else (1.0 / value if value > 0 else 0.0)
    return rate if math.isfinite(rate) and rate > 0 else None


def epochs_per_second(progress):
    """1 / the last epoch's seconds, which a trial reports since 21 November 2026; for one started
    by older code, its epochs over the time since it began (setup included)."""
    if not progress:
        return None
    seconds = progress.get('epoch_seconds')
    if _measured(seconds) and seconds > 0:
        return 1.0 / seconds
    elapsed = progress.get('elapsed')
    if _measured(elapsed) and elapsed > 0:
        return (progress['epoch'] + 1) / elapsed     # epochs count from 0
    return None


def describe(folder):
    """``(cluster, method, trial)`` of a trial's folder, e.g. ``('RBF rule_3', 'MLP', 'test_1')``.

    The folder is ``<model tree>/[cluster_organizer/]<clusterer>/<cluster>/<method>/<trial>``; the
    model tree is the nearest folder above it that has ``exe_files``.
    """
    path = Path(folder)
    root = next((parent for parent in path.parents if (parent / 'exe_files').is_dir()), None)
    parts = list(path.relative_to(root).parts) if root is not None else list(path.parts[-4:])
    trial = parts[-1] if parts else path.name
    method = parts[-2] if len(parts) > 1 else ''
    cluster = ' '.join(part for part in parts[:-2] if part != 'cluster_organizer')
    return cluster or 'global', method, trial


def card_numbers():
    """Card UUID -> its number, as nvidia-smi and the GPU panels number them. Asked once a minute."""
    if time.time() - _CARDS['at'] > 60:
        try:
            answer = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10).stdout
            _CARDS['index'] = {uuid.strip(): number.strip() for number, uuid in
                               (line.split(',', 1) for line in answer.splitlines() if ',' in line)}
        except (OSError, subprocess.SubprocessError):
            _CARDS['index'] = {}
        _CARDS['at'] = time.time()
    return _CARDS['index']


def card_of(environment):
    """The card a trial was given: the first of its ``CUDA_VISIBLE_DEVICES``, by number; ``?``
    when its environment cannot be read (a process of another user)."""
    shown = (environment or {}).get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
    if not shown:
        return '?' if environment is None else 'cpu'
    return shown if shown.isdigit() else card_numbers().get(shown, shown)


def case_of(process):
    """The case a trial trains: the ``--config`` of the training above it, a few parents up."""
    try:
        for parent in process.parents()[:4]:
            argv = parent.cmdline()
            if '--config' in argv[:-1]:
                return os.path.basename(os.path.normpath(argv[argv.index('--config') + 1]))
    except Exception:                    # noqa: BLE001 -- a parent that ended meanwhile, or another user's
        pass
    return ''


def running_trials():
    """Every trial process on this machine this user may look at, one per trial.

    A trial's data loaders are forked copies of it with the same command line; only the topmost
    process of a trial is counted.
    """
    import psutil

    found = {}
    for process in psutil.process_iter(['pid', 'ppid', 'cmdline', 'create_time']):
        argv = process.info.get('cmdline') or []
        if not any(str(arg).endswith(RUNNER) for arg in argv) or '--path-weights' not in argv[:-1]:
            continue
        found[process.info['pid']] = (process, argv[argv.index('--path-weights') + 1])
    trials = []
    cards = card_memory() if found else {}
    for pid, (process, folder) in found.items():
        if process.info.get('ppid') in found:
            continue
        try:
            environment = process.environ()
        except Exception:                # noqa: BLE001 -- psutil.AccessDenied, or the trial just ended
            environment = None
        cluster, method, trial = describe(folder)
        # the trial and the copies of itself it forked to load data: what the trial costs
        family = [process] + [other for other, _ in found.values() if _descends(other, pid, found)]
        trials.append({'pid': pid, 'folder': folder, 'case': case_of(process), 'cluster': cluster,
                       'method': method, 'trial': trial, 'gpu': card_of(environment),
                       'started': process.info.get('create_time') or time.time(),
                       'progress': read_progress(os.path.join(folder, 'progress.json')),
                       'loops': loops_per_second(folder), 'loop': loop_progress(folder),
                       'epochs': epochs_total(folder), 'log_age': log_age(folder),
                       **_usage(family, cards)})
    return sorted(trials, key=lambda t: (t['case'], t['cluster'], t['method'], t['trial']))


def _descends(process, pid, found):
    """Whether a trial process is a forked copy below the trial ``pid``."""
    parent = process.info.get('ppid')
    while parent in found:
        if parent == pid:
            return True
        parent = found[parent][0].info.get('ppid')
    return False


def _usage(family, cards):
    """Processor seconds, resident memory and card memory of a trial with its data loaders."""
    cpu = memory = 0.0
    seen = False
    for process in family:
        try:
            times = process.cpu_times()
            cpu += times.user + times.system
            memory += process.memory_info().rss
            seen = True
        except Exception:                # noqa: BLE001 -- it ended between the listing and the question
            continue
    on_card = sum(cards.get(process.info['pid'], 0) for process in family)
    return {'cpu_seconds': cpu if seen else None, 'memory': memory if seen else None,
            'gpu_memory': on_card if on_card else None}


def _label(value):
    return str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


def _number(value):
    value = float(value)
    if math.isnan(value):
        return 'NaN'
    if math.isinf(value):
        return '+Inf' if value > 0 else '-Inf'
    return repr(value)


def _measured(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def render(trials, now=None):
    """The trials in Prometheus's text format; a trial's series end with it."""
    moment = time.time() if now is None else now
    samples = {name: [] for name, _ in FAMILIES}
    for trial in trials:
        labels = ','.join(f'{key}="{_label(trial[key])}"' for key in LABELS)
        samples['eforecast_trial_running_seconds'].append(f'{{{labels}}} {_number(moment - trial["started"])}')
        if trial.get('loops') is not None:
            samples['eforecast_trial_loops_per_second'].append(f'{{{labels}}} {_number(trial["loops"])}')
        loop = trial.get('loop') or {}
        for family, value in (('eforecast_trial_loop_done', loop.get('done')), ('eforecast_trial_loop_total', loop.get('total')),
                              ('eforecast_trial_loop_rate', loop.get('rate')),
                              ('eforecast_trial_loop_left_seconds', loop.get('left')),
                              ('eforecast_trial_loop_training', None if loop.get('training') is None else int(loop['training'])),
                              ('eforecast_trial_epochs_total', trial.get('epochs')),
                              ('eforecast_trial_log_age_seconds', trial.get('log_age')),
                              ('eforecast_trial_cpu_seconds_total', trial.get('cpu_seconds')),
                              ('eforecast_trial_memory_bytes', trial.get('memory')),
                              ('eforecast_trial_gpu_memory_bytes', trial.get('gpu_memory'))):
            if value is not None:
                samples[family].append(f'{{{labels}}} {_number(value)}')
        progress = trial['progress']
        if progress is None:
            continue
        epochs = epochs_per_second(progress)
        if epochs is not None:
            samples['eforecast_trial_epochs_per_second'].append(f'{{{labels}}} {_number(epochs)}')
        samples['eforecast_trial_epoch'].append(f'{{{labels}}} {progress["epoch"]}')
        for family, prefix in (('eforecast_trial_error', ''), ('eforecast_trial_best_error', 'best_')):
            for which in SETS:
                value = progress.get(f'{prefix}mae_{which}')
                if _measured(value):
                    samples[family].append(f'{{{labels},set="{which}"}} {_number(value)}')
        if _measured(progress.get('time')):
            samples['eforecast_trial_last_epoch_age_seconds'].append(
                f'{{{labels}}} {_number(max(0.0, moment - progress["time"]))}')
    lines = []
    for name, text in FAMILIES:
        kind = 'counter' if name in COUNTERS else 'gauge'
        lines += [f'# HELP {name} {text}', f'# TYPE {name} {kind}'] + [name + sample for sample in samples[name]]
    return '\n'.join(lines) + '\n'


def _duration(seconds):
    seconds = int(max(0, seconds))
    if seconds < 3600:
        return f'{seconds // 60}m {seconds % 60:02d}s'
    return f'{seconds // 3600}h {seconds % 3600 // 60:02d}m'


def _rate(value):
    """Three significant digits, and no exponent from 100 up: 12.3, 0.00603, 1234."""
    if value is None:
        return ''
    return f'{value:.0f}' if value >= 100 else f'{value:.3g}'


def loop_text(loop):
    """The loop a trial is in, in a few words: ``train 376/446 84%`` (``eval`` for a pass that is not
    training, ``loop`` while it cannot be told yet)."""
    if not loop:
        return ''
    kind = {True: 'train', False: 'eval'}.get(loop.get('training'), 'loop')
    share = f" {100 * loop['done'] / loop['total']:.0f}%" if loop.get('total') else ''
    return f"{kind} {loop['done']}/{loop['total']}{share}"


def table(trials, now=None):
    """The trials as a table for a terminal."""
    moment = time.time() if now is None else now
    if not trials:
        return 'no trial is training on this machine'
    header = ('case', 'cluster', 'method', 'trial', 'gpu', 'epoch', 'loop now', 'it/s', 'left', 'val', 'test',
              'best val', 'best test', 'loops/s', 'epochs/s', 'running', 'last epoch')
    rows = []
    for trial in trials:
        progress = trial['progress'] or {}
        loop = trial.get('loop') or {}
        error = [f'{progress[key]:.4f}' if _measured(progress.get(key)) else ''
                 for key in ('mae_val', 'mae_test', 'best_mae_val', 'best_mae_test')]
        last = f'{_duration(moment - progress["time"])} ago' if _measured(progress.get('time')) else 'none yet'
        epoch = str(progress.get('epoch', ''))
        if epoch and trial.get('epochs'):
            epoch += f"/{trial['epochs']}"
        rows.append((trial['case'], trial['cluster'], trial['method'], trial['trial'], trial['gpu'], epoch,
                     loop_text(loop), _rate(loop.get('rate')),
                     _duration(loop['left']) if _measured(loop.get('left')) else '', *error, _rate(trial.get('loops')),
                     _rate(epochs_per_second(trial['progress'])), _duration(moment - trial['started']), last))
    widths = [max(len(str(row[column])) for row in (header, *rows)) for column in range(len(header))]
    return '\n'.join('  '.join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip()
                     for row in (header, *rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--prometheus', action='store_true', help="print Prometheus's text format instead")
    parser.add_argument('--watch', type=float, metavar='SECONDS', help='print again every SECONDS until stopped')
    arguments = parser.parse_args(argv)
    while True:
        trials = running_trials()
        text = render(trials) if arguments.prometheus else table(trials)
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
