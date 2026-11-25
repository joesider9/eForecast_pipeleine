# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""What every study of the workspace's cases has done, and what it has left.

A study is the search of one method in one cluster of a case: so many trials wanted, each of
them complete, pruned, failed, running or still to come. The trainings keep each study in a
file beside the models (``study.journal``), so this reads those files, for every case of the
workspace -- whatever started the training and whatever version of the code it runs:

* per study, the trials by how they ended, how many are wanted, how many it will still be given
  and how many it lacks beyond that (counted as the scheduler counts: ``optuna_search.attempts``), the best
  value, when the last one finished and how long one takes;
* the trials that finished most recently, across every case;
* each study whole, for the copier that shows them in the Optuna dashboard
  (``docker/monitoring/optuna_mirror``): the dashboard reads a database, the trainings write
  files, and nothing else connects the two.

A study is read again only when its file has changed. The metrics service keeps this up to date
in the background and serves it at ``/studies`` (Prometheus job ``studies``), ``/studies/index``
and ``/studies/export?name=...``; ``bin/eforecast studies`` prints it.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

STATES = ('complete', 'pruned', 'failed', 'running', 'waiting')
_STATE = {'COMPLETE': 'complete', 'PRUNED': 'pruned', 'FAIL': 'failed', 'RUNNING': 'running', 'WAITING': 'waiting'}
LABELS = ('case', 'cluster', 'method')
FAMILIES = (
    ('eforecast_study_trials', 'Trials of the study, by how each stands: complete, pruned, failed, running, waiting.'),
    ('eforecast_study_trials_wanted', 'Trials the case asks of the study (n_trials of its method).'),
    ('eforecast_study_trials_left', 'Trials the study will still be given: the ones running, and the ones still to '
                                    'be asked for as far as its attempts last (twice the trials wanted; the methods '
                                    'trained on the processors have no such bound).'),
    ('eforecast_study_trials_given_up', 'Trials the study lacks and will not be asked for: its attempts are used up.'),
    ('eforecast_study_failures_not_counted', 'Failed trials of the study that are not its attempts: the run was '
                                             'stopped or died under them, or their card was lost.'),
    ('eforecast_study_best_value', 'The best value of the study so far (the studies minimise).'),
    ('eforecast_study_last_finished_timestamp_seconds', 'When the last trial of the study finished (Unix time).'),
    ('eforecast_study_trial_seconds', 'How long a finished trial of the study took, on average.'),
    ('eforecast_study_activity_timestamp_seconds', 'When the study was last written to (Unix time).'),
    ('eforecast_finished_trial_value', 'The value of a trial that finished recently (NaN when it has none).'),
    ('eforecast_finished_trial_timestamp_seconds', 'When that trial finished (Unix time).'),
    ('eforecast_finished_trial_seconds', 'How long that trial took.'),
    ('eforecast_studies_read_timestamp_seconds', 'When the studies were last read (Unix time).'),
    ('eforecast_studies_read_seconds', 'How long reading them took.'),
    ('eforecast_studies_unreadable', 'Cases or studies that could not be read at the last reading.'),
)
#: Finished trials listed one by one, the newest across every case.
RECENT = 40
#: The scheduler's own notes on a trial it is running (``leases``): who holds it and a heartbeat it
#: refreshes every minute. Not facts about the trial, and a copy of them would change every minute.
BOOKKEEPING = ('lease_owner', 'lease_attempt', 'lease_heartbeat')
EVERY = float(os.environ.get('EFORECAST_STUDIES_EVERY', '60') or 60)


def cluster_label(path_model, cluster_dir):
    """A cluster as the trials name it: ``RBF rule_3`` for ``.../cluster_organizer/RBF/rule_3``, ``global``."""
    try:
        parts = Path(cluster_dir).absolute().relative_to(Path(path_model).absolute()).parts
    except ValueError:
        parts = Path(cluster_dir).parts[-2:]
    return ' '.join(part for part in parts if part != 'cluster_organizer') or 'global'


def _stamp(moment):
    """A trial's time as Unix seconds; the studies record local time without a zone."""
    return None if moment is None else moment.timestamp()


def _plain(value):
    """What JSON and a label can carry."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)


def summarise(study, *, case, cluster, method, wanted, activity, limited=True):
    """``(summary, whole)`` of one study: its numbers, and every trial of it for the copier.

    What is left is counted by the scheduler's own definition (``optuna_search.attempts``), so
    that a page cannot promise a trial the scheduler will not ask for. ``limited`` is whether
    the study's attempts are bounded: they are for a method the cards train.
    """
    from optuna.distributions import distribution_to_json
    from eforecast.training.command_line import optuna_search

    stand = optuna_search.attempts(study, wanted, limited=limited)
    counts = {state: 0 for state in STATES}
    finished, trials = [], []
    best = None
    for trial in study.get_trials(deepcopy=False):
        state = _STATE.get(trial.state.name, 'waiting')
        counts[state] += 1
        began, ended = _stamp(trial.datetime_start), _stamp(trial.datetime_complete)
        value = trial.value if trial.values is not None and len(trial.values) == 1 else None
        if state == 'complete' and value is not None and math.isfinite(value) and (best is None or value < best[0]):
            best = (value, trial.number)
        if state in ('complete', 'pruned', 'failed') and ended is not None:
            finished.append({'trial': f'test_{trial.number}', 'state': state, 'value': _plain(value), 'at': ended,
                             'seconds': ended - began if began is not None else None})
        trials.append({'number': trial.number, 'state': trial.state.name, 'values': trial.values,
                       'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                       'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                       'params': {name: _plain(value_) for name, value_ in trial.params.items()},
                       'distributions': {name: distribution_to_json(distribution)
                                         for name, distribution in trial.distributions.items()},
                       'intermediate_values': {str(step): value_ for step, value_ in trial.intermediate_values.items()},
                       'user_attrs': {name: value_ for name, value_ in trial.user_attrs.items()
                                      if name not in BOOKKEEPING and _jsonable(value_)}})
    done = [row for row in finished if row['state'] != 'failed' and row['seconds'] is not None]
    summary = {'case': case, 'cluster': cluster, 'method': method, 'name': study.study_name, 'counts': counts,
               'wanted': wanted, 'left': stand['left'], 'given_up': stand['given_up'], 'owed': stand['owed'],
               'not_counted': stand['excused'], 'attempts': stand['counted'], 'allowed': stand['allowed'],
               'best': None if best is None else best[0], 'best_trial': None if best is None else f'test_{best[1]}',
               'last_finished': max((row['at'] for row in finished), default=None),
               'seconds': sum(row['seconds'] for row in done) / len(done) if done else None,
               'activity': activity, 'finished': sorted(finished, key=lambda row: row['at'])[-RECENT:]}
    whole = {'name': study.study_name, 'case': case, 'cluster': cluster, 'method': method,
             'direction': study.direction.name, 'trials': trials}
    return summary, whole


def _jsonable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


class Reader:
    """The studies of a workspace, read again only where a file has changed."""

    def __init__(self, workspace):
        self.workspace = workspace
        self._cases = {}                      # case -> (what its configuration was when resolved, static_data)
        self._studies = {}                    # journal -> (its size and time when read, summary, whole as JSON)

    def statics(self):
        """``{case: static_data}`` of every case that resolves; the others are counted, not fatal."""
        from eforecast.config import studies as pages
        from eforecast.config.workspace import Workspace

        workspace = Workspace(self.workspace, admin=True)
        found, unreadable = {}, 0
        for entry in workspace.list():
            name, stamp = entry['name'], entry.get('modified')
            cached = self._cases.get(name)
            if cached is None or cached[0] != stamp:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):          # the initializer prints as it goes
                        static = pages._static(workspace.load(name), str(workspace.case_path(name)))
                except Exception as error:                                   # noqa: BLE001 -- one case, not all
                    log.debug('case %s does not resolve (%r)', name, error)
                    unreadable += 1
                    self._cases.pop(name, None)
                    continue
                cached = self._cases[name] = (stamp, static)
            found[name] = cached[1]
        for name in set(self._cases) - set(found):
            self._cases.pop(name, None)
        return found, unreadable

    def journals(self, static):
        """``(cluster name, cluster folder, method, study file)`` of every study a case's model tree holds."""
        from eforecast.common_utils import trial_layout
        from eforecast.config import studies as pages

        if not os.path.isdir(static['path_model']):
            return
        for cluster_name, cluster_dir in pages.clusters_of(static).items():
            if not os.path.isdir(cluster_dir):
                continue
            for method in sorted(os.listdir(cluster_dir)):
                try:
                    journal = Path(trial_layout.method_directory(static, cluster_dir, method)) / 'study.journal'
                except (OSError, ValueError):
                    continue
                if journal.is_file():
                    yield cluster_name, cluster_dir, method, journal

    def read(self):
        """Everything, as one snapshot: ``{'studies', 'finished', 'unreadable', 'at', 'seconds'}``."""
        import optuna
        from eforecast.config.capabilities import CPU_METHODS
        from eforecast.training.command_line import optuna_search

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        began = time.time()
        statics, unreadable = self.statics()
        seen, summaries = set(), []
        for case, static in statics.items():
            try:
                studies = list(self.journals(static))
            except Exception as error:                                       # noqa: BLE001 -- a tree without its clusters file
                log.debug('the studies of %s cannot be listed (%r)', case, error)
                unreadable += 1
                continue
            for cluster_name, cluster_dir, method, journal in studies:
                key = str(journal)
                try:
                    status = journal.stat()
                    stamp = (status.st_size, status.st_mtime_ns)
                    cached = self._studies.get(key)
                    if cached is None or cached[0] != stamp:
                        study = optuna_search.read_study(static, cluster_name, cluster_dir, method)
                        if study is None:
                            raise LookupError('no study of this case in the file')
                        wanted = (static.get(method) or {}).get('n_trials')
                        wanted = int(wanted) if isinstance(wanted, (int, float)) and not isinstance(wanted, bool) else None
                        summary, whole = summarise(study, case=case, cluster=cluster_label(static['path_model'], cluster_dir),
                                                   method=method, wanted=wanted, activity=status.st_mtime,
                                                   limited=method not in CPU_METHODS)
                        whole['version'] = f'{stamp[0]}-{stamp[1]}'
                        cached = self._studies[key] = (stamp, summary, json.dumps(whole))
                except Exception as error:                                   # noqa: BLE001 -- one study, not all
                    log.debug('study %s cannot be read (%r)', journal, error)
                    unreadable += 1
                    continue
                seen.add(key)
                summaries.append(cached[1])
        for key in set(self._studies) - seen:                                # a study that was deleted or moved
            self._studies.pop(key, None)
        finished = sorted(({**row, 'case': s['case'], 'cluster': s['cluster'], 'method': s['method']}
                           for s in summaries for row in s['finished']), key=lambda row: row['at'])[-RECENT:]
        return {'studies': sorted(summaries, key=lambda s: (s['case'], s['cluster'], s['method'])), 'finished': finished,
                'unreadable': unreadable, 'at': time.time(), 'seconds': time.time() - began}

    def index(self):
        """What the copier asks first: every study's name and the version of its file."""
        return [{'name': summary['name'], 'version': f'{stamp[0]}-{stamp[1]}', 'case': summary['case'],
                 'cluster': summary['cluster'], 'method': summary['method'],
                 'trials': sum(summary['counts'].values())}
                for stamp, summary, _ in list(self._studies.values())]

    def whole(self, name):
        """One study with every trial, as JSON text; None when no study has this name."""
        for _, summary, whole in list(self._studies.values()):
            if summary['name'] == name:
                return whole
        return None


class Watcher(threading.Thread):
    """Keeps a snapshot of the studies fresh, so that a scrape never waits for a file."""

    def __init__(self, workspace, every=EVERY):
        super().__init__(daemon=True, name='eforecast-studies')
        self.reader, self.every = Reader(workspace), every
        self.snapshot, self.error = None, None
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                self.snapshot, self.error = self.reader.read(), None
            except Exception as error:                                       # noqa: BLE001 -- the next round may read
                self.error = f'{type(error).__name__}: {error}'
                log.warning('the studies could not be read (%s)', self.error)
            self._halt.wait(self.every)

    def stop(self):
        self._halt.set()


# ----------------------------------------------------------------------------------- rendering

def _label(value):
    return str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


def _number(value):
    if value is None:
        return 'NaN'
    value = float(value)
    return 'NaN' if math.isnan(value) else ('+Inf' if value > 0 else '-Inf') if math.isinf(value) else repr(value)


def render(snapshot):
    """A snapshot in Prometheus's text format."""
    samples = {name: [] for name, _ in FAMILIES}
    for study in snapshot['studies']:
        labels = ','.join(f'{key}="{_label(study[key])}"' for key in LABELS)
        samples['eforecast_study_trials'] += [f'{{{labels},state="{state}"}} {study["counts"][state]}' for state in STATES]
        for family, key in (('eforecast_study_trials_wanted', 'wanted'), ('eforecast_study_trials_left', 'left'),
                            ('eforecast_study_trials_given_up', 'given_up'),
                            ('eforecast_study_failures_not_counted', 'not_counted'),
                            ('eforecast_study_best_value', 'best'),
                            ('eforecast_study_last_finished_timestamp_seconds', 'last_finished'),
                            ('eforecast_study_trial_seconds', 'seconds'),
                            ('eforecast_study_activity_timestamp_seconds', 'activity')):
            if study.get(key) is not None:
                samples[family].append(f'{{{labels}}} {_number(study[key])}')
    for row in snapshot['finished']:
        labels = ','.join(f'{key}="{_label(row[key])}"' for key in (*LABELS, 'trial', 'state'))
        samples['eforecast_finished_trial_value'].append(f'{{{labels}}} {_number(row["value"])}')
        samples['eforecast_finished_trial_timestamp_seconds'].append(f'{{{labels}}} {_number(row["at"])}')
        if row.get('seconds') is not None:
            samples['eforecast_finished_trial_seconds'].append(f'{{{labels}}} {_number(row["seconds"])}')
    samples['eforecast_studies_read_timestamp_seconds'].append(f' {_number(snapshot["at"])}')
    samples['eforecast_studies_read_seconds'].append(f' {_number(snapshot["seconds"])}')
    samples['eforecast_studies_unreadable'].append(f' {snapshot["unreadable"]}')
    lines = []
    for name, text in FAMILIES:
        lines += [f'# HELP {name} {text}', f'# TYPE {name} gauge'] + [name + sample for sample in samples[name]]
    return '\n'.join(lines) + '\n'


def _ago(seconds):
    seconds = int(max(0, seconds))
    if seconds < 3600:
        return f'{seconds // 60}m ago'
    if seconds < 86400:
        return f'{seconds // 3600}h {seconds % 3600 // 60:02d}m ago'
    return f'{seconds // 86400}d ago'


def _lines(header, rows, left=1):
    widths = [max(len(str(row[column])) for row in (header, *rows)) for column in range(len(header))]
    return ['  '.join(str(cell).ljust(width) if column < left else str(cell).rjust(width)
                      for column, (cell, width) in enumerate(zip(row, widths))).rstrip() for row in (header, *rows)]


def table(snapshot, case=None, now=None, every=False):
    """The studies as text for a terminal: each case's totals, then its studies that are not done
    (all of them with ``every``), then the trials that finished last."""
    moment = time.time() if now is None else now
    studies = [s for s in snapshot['studies'] if case in (None, s['case'])]
    if not studies:
        return 'no study' + (f' of {case}' if case else '') + ' was found in the workspace'
    out = []
    for name in dict.fromkeys(s['case'] for s in studies):
        mine = [s for s in studies if s['case'] == name]
        total = {state: sum(s['counts'][state] for s in mine) for state in STATES}
        wanted, left = sum(s['wanted'] or 0 for s in mine), sum(s['left'] or 0 for s in mine)
        given_up = sum(s.get('given_up') or 0 for s in mine)
        short = [s for s in mine if s.get('given_up')]
        open_ = [s for s in mine if s['left'] or s['counts']['running']]
        hours = sum((s['left'] or 0) * (s['seconds'] or 0) for s in mine) / 3600
        out.append(f"{name}: {len(mine)} studies, {len(mine) - len(open_) - len([s for s in short if s not in open_])} done, "
                   f"{len(open_)} open; trials "
                   f"{total['complete']} complete, {total['pruned']} pruned, {total['failed']} failed, "
                   f"{total['running']} running, {left} left of {wanted}"
                   + (f'; about {hours:.1f} h of training left, one trial after another' if hours else ''))
        if given_up:
            out.append(f"  {given_up} more trial(s) are lacking and will not be asked for: {len(short)} "
                       f"{'study has' if len(short) == 1 else 'studies have'} used up their attempts "
                       f"(twice the trials wanted). They are listed with what they gave up.")
        shown = mine if every else open_ + [s for s in short if s not in open_]
        rows = []
        for s in shown:
            last = _ago(moment - s['last_finished']) if s['last_finished'] else ''
            rows.append((f"{s['cluster']} {s['method']}", s['counts']['complete'], s['counts']['pruned'], s['counts']['failed'],
                         s['counts']['running'], '' if s['left'] is None else s['left'],
                         *((s.get('given_up') or '',) if given_up else ()),
                         '' if s['wanted'] is None else s['wanted'],
                         '' if s['best'] is None else f"{s['best']:.4f}",
                         '' if s['seconds'] is None else f"{s['seconds'] / 60:.0f}m", last))
        if rows:
            out += ['  ' + line for line in _lines(('study', 'complete', 'pruned', 'failed', 'running', 'left',
                                                    *(('given up',) if given_up else ()), 'wanted',
                                                    'best', 'a trial', 'last finished'), rows)]
        out.append('')
    recent = [row for row in snapshot['finished'] if case in (None, row['case'])][-12:]
    if recent:
        rows = [(time.strftime('%d %b %H:%M', time.localtime(row['at'])), row['case'], f"{row['cluster']} {row['method']}",
                 row['trial'], row['state'], '' if row['value'] is None else f"{row['value']:.4f}",
                 '' if row['seconds'] is None else f"{row['seconds'] / 60:.0f}m") for row in reversed(recent)]
        out += ['finished last'] + ['  ' + line for line in _lines(('when', 'case', 'study', 'trial', 'ended', 'value',
                                                                     'took'), rows, left=5)]
    return '\n'.join(out).rstrip()


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--workspace', default=os.environ.get('EFORECAST_WORKSPACE'))
    parser.add_argument('--case', help='one case only')
    parser.add_argument('--all', action='store_true', dest='every', help='also the studies that are done')
    parser.add_argument('--prometheus', action='store_true', help="print Prometheus's text format instead")
    parser.add_argument('--watch', type=float, metavar='SECONDS', help='print again every SECONDS until stopped')
    arguments = parser.parse_args(argv)
    if not arguments.workspace:
        print('no workspace: set EFORECAST_WORKSPACE or give --workspace', file=sys.stderr)
        return 2
    reader = Reader(arguments.workspace)
    while True:
        snapshot = reader.read()
        text = render(snapshot) if arguments.prometheus else table(snapshot, arguments.case, every=arguments.every)
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
