# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""What the trials' hyper-parameters say; written with every evaluation.

``Results/hyper_parameters.csv``
    one row per finished trial of the case: the hyper-parameters that were searched, how its
    training ended, and its error on the selection and on the holdout dates.
``Results/hyper_parameter_report.txt``
    the same trials grouped by each hyper-parameter, method by method, the value that did best
    first, and how much that says.

Which slice says what (the protocol of ``ensemble``)
    The **selection** slice is the one that may choose, so a hyper-parameter is judged there. The
    **holdout** column beside it only says whether that judgment holds on dates nothing was
    chosen from: a value picked *because* of its holdout number makes the holdout one more
    selection slice, and the honest number of ``summary.csv`` is gone.

How trials are compared
    Clusters are not equally hard, so trials of different studies (cluster x method) are not
    compared by their error but by their **place** inside their own study -- 0 for its best
    trial, 100 for its worst -- and by how far **above the best** of their study they are.

Sources
    ``<trial>/parameters.pickle`` (the networks), the study's results table (the CPU methods,
    which keep no parameter file), ``trial_result.json`` and ``progress.json``. A hyper-parameter
    of a method is a parameter with more than one value inside one of its studies; the rest was
    not searched.

Nothing here can fail an evaluation: a report that cannot be written is said, and left out.
"""
import json
import logging
import math
import os
import textwrap
import time

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TABLE = 'hyper_parameters.csv'
REPORT = 'hyper_parameter_report.txt'
SLICES = ('selection', 'holdout')
#: What a parameter file holds that is no hyper-parameter: names, places and bookkeeping.
NOT_SEARCHED = frozenset({'trial_number', 'number', 'name', 'cluster_dir', 'cluster_name', 'method', 'n_trials',
                          'n_jobs'})
#: Optimizers that find their own step (``deep_models/pytorch_2x/optimizers.py`` builds them with a
#: fixed rate): the learning rate drawn for such a trial is not its step, only the floor of its
#: schedule, so those trials say nothing about the learning rate.
OWN_STEP = ('prodigy',)
#: Trials a value needs before it can be called the best one.
ENOUGH = 3
NAN = float('nan')


# ------------------------------------------------------------------------------------- reading

def _read_json(path):
    try:
        with open(path, encoding='utf-8') as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _number(value):
    """A finite float, or NaN."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return NAN
    return value if math.isfinite(value) else NAN


def _scalar(value):
    """``(True, value)`` for what a hyper-parameter can be, ``(False, None)`` for a structure or a path.

    ``None`` is a choice like any other (no feature selection, no compression) and is written
    ``none``, which a table can group by.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return True, 'none'
    if isinstance(value, bool):
        return True, value
    if isinstance(value, (int, float)):
        return True, value if math.isfinite(value) else 'none'
    if isinstance(value, str):
        return os.sep not in value and len(value) <= 80, value
    return False, None


def layers(experiment):
    """The layers of each branch in a few words: ``nwp: conv 2, Flatten, dense 512 | output: dense 64``."""
    if not isinstance(experiment, dict):
        return ''
    branches = []
    for branch, items in experiment.items():
        words = []
        for item in items if isinstance(items, (list, tuple)) else ():
            kind, size = (item[0], item[1]) if isinstance(item, (list, tuple)) and len(item) > 1 else (item, None)
            sized = isinstance(size, (int, float)) and not isinstance(size, bool)
            words.append(f'{kind} {size:g}' if sized else str(kind))
        branches.append(f'{branch}: ' + ', '.join(words))
    return ' | '.join(branches)


def folder_parameters(path_weights):
    """What a network was trained with (``parameters.pickle``); None when the folder has no such file."""
    file = os.path.join(path_weights, 'parameters.pickle')
    if not os.path.isfile(file):
        return None
    import joblib
    try:
        parameters = joblib.load(file)
    except Exception as error:  # noqa: BLE001 -- written by another version, or no pickle at all
        log.debug('cannot read %s (%r)', file, error)
        return None
    if not isinstance(parameters, dict):
        return None
    found = {}
    for name, value in parameters.items():
        usable, value = _scalar(value)
        if usable and name not in NOT_SEARCHED:
            found[str(name)] = value
    described = layers(parameters.get('experiment'))
    if described:
        found['layers'] = described
    return found


def study_parameters(static_data, spec, tables):
    """A trial's parameters from the results table of its study (``params_<name>`` columns).

    The CPU methods keep no parameter file; their study writes this table when it ends. ``tables``
    remembers each study's table for the trials after the first.
    """
    key = (spec['cluster_name'], spec['method'])
    if key not in tables:
        tables[key] = {}
        try:
            from eforecast.common_utils import trial_layout
            file = trial_layout.results_file(static_data, spec['cluster_dir'], spec['cluster_name'], spec['method'])
            if os.path.isfile(file):
                table = pd.read_csv(file)
                number = next((c for c in ('trial_number', 'number') if c in table.columns), None)
                names = [c for c in table.columns if c.startswith('params_')]
                for _, row in table.iterrows() if number is not None else ():
                    if pd.isna(row[number]):
                        continue
                    values = {}
                    for column in names:
                        usable, value = _scalar(None if pd.isna(row[column]) else row[column])
                        if usable and column[len('params_'):] not in NOT_SEARCHED:
                            values[column[len('params_'):]] = value
                    tables[key][f'test_{int(row[number])}'] = values
        except Exception as error:  # noqa: BLE001 -- a table of another shape: these trials have no parameters here
            log.debug('cannot read the results table of %s %s (%r)', *key, error)
    return tables[key].get(spec['trial'])


def outcome(path_weights):
    """How the training of a trial ended, from its ``trial_result.json`` and ``progress.json``."""
    result = _read_json(os.path.join(path_weights, 'trial_result.json'))
    epoch = _read_json(os.path.join(path_weights, 'progress.json')).get('epoch')
    state = result.get('state')
    return {'state': state if isinstance(state, str) else None,
            'objective': _number(result.get('value')), 'val_error': _number(result.get('mae_val')),
            'test_error': _number(result.get('mae_test')), 'minutes': _number(result.get('duration')) / 60,
            'epochs': epoch + 1 if isinstance(epoch, int) and not isinstance(epoch, bool) else NAN}


def errors(frame, targets, metrics):
    """The MAE of one trial's prediction on each slice, measured as the evaluation tables measure it."""
    frame = targets.inverse(frame)
    found = {}
    for name in SLICES:
        dates = getattr(targets, name)
        measured = metrics(frame.loc[frame.index.intersection(dates)], targets.y, targets.rated,
                           targets.columns).get('all', {})
        found[f'mae_{name}'] = _number(measured.get('mae'))
        found[f'n_{name}'] = int(measured.get('n') or 0)
    return found


# ------------------------------------------------------------------------------------ the table

def searched(studies, parameters):
    """The parameters with more than one value inside one study, in the order they were met."""
    names = list(dict.fromkeys(name for found in parameters for name in found))
    return [name for name in names
            if any(len({repr(parameters[at][name]) for at in members if name in parameters[at]}) > 1
                   for members in studies.values())]


def collect(static_data, targets, specs, trial_predictions, *, metrics, current=None):
    """``(table, searched, constants)``: a row per finished trial; per method the columns of the
    parameters that were searched in it; and per method the parameters every trial shares.

    ``current(spec)`` says whether the cached prediction of a trial was made from the model and the
    inputs of today; a trial without a current prediction keeps its row and has no error.
    """
    rows, parameters, tables = [], [], {}
    for spec in specs:
        row = {'cluster': spec['cluster_name'], 'method': spec['method'], 'trial': spec['trial'],
               'kept': bool(spec.get('selected')), **outcome(spec['path_weights'])}
        row.update({f'mae_{name}': NAN for name in SLICES}, **{f'n_{name}': 0 for name in SLICES})
        frame = (trial_predictions or {}).get(spec['path_weights'])
        if frame is not None and len(frame) and _is_current(current, spec):
            try:
                row.update(errors(frame, targets, metrics))
            except Exception as error:  # noqa: BLE001 -- one unreadable prediction is one trial without an error
                log.debug('no error for %s (%r)', spec['path_weights'], error)
        found = folder_parameters(spec['path_weights'])
        if found is None:
            found = study_parameters(static_data, spec, tables) if spec.get('cluster_dir') else None
        rows.append(row)
        parameters.append(found or {})
    if not rows:
        return pd.DataFrame(), {}, {}
    table = pd.DataFrame(rows)
    studies = {key: list(members) for key, members in table.groupby(['cluster', 'method'], sort=False).indices.items()}
    fixed = set(table.columns)

    def column_of(name):                      # a parameter called like a column of the table keeps both
        return f'param_{name}' if name in fixed else name
    names_of = {method: searched({key: members for key, members in studies.items() if key[1] == method}, parameters)
                for method in dict.fromkeys(table['method'])}
    for name in columns_of(names_of):
        table[column_of(name)] = pd.Series([found.get(name, np.nan) for found in parameters], dtype=object)
    by_method = {method: [column_of(name) for name in names] for method, names in names_of.items()}
    constants = {}
    for method, members in table.groupby('method', sort=False).indices.items():
        shared = {}
        for name in dict.fromkeys(name for at in members for name in parameters[at]):
            values = {repr(parameters[at].get(name)) for at in members}
            if len(values) == 1 and name != 'layers' and parameters[members[0]][name] != 'none':
                shared[name] = parameters[members[0]][name]
        constants[method] = shared
    return table, by_method, constants


def columns_of(by_method):
    """Every searched parameter once, in the order the methods met them."""
    return list(dict.fromkeys(name for names in by_method.values() for name in names))


def _is_current(current, spec):
    if current is None:
        return True
    try:
        return bool(current(spec))
    except Exception as error:  # noqa: BLE001 -- a check that cannot be made is not a reason to say nothing
        log.debug('cannot tell whether the prediction of %s is current (%r)', spec['path_weights'], error)
        return True


def placed(table):
    """Each trial's place in its study (0 best ... 100 worst), how far above its best it is (%), and
    whether it is that best -- on each slice. A study with one scored trial compares nothing."""
    table = table.copy()
    for name in SLICES:
        error = pd.to_numeric(table[f'mae_{name}'], errors='coerce')
        place, above, best = (pd.Series(np.nan, index=table.index), pd.Series(np.nan, index=table.index),
                              pd.Series(False, index=table.index))
        for _, members in table.groupby(['cluster', 'method'], sort=False).groups.items():
            scored = error.loc[members].dropna()
            if len(scored) < 2:
                continue
            place.loc[scored.index] = 100 * (scored.rank(method='average') - 1) / (len(scored) - 1)
            if scored.min() > 0:
                above.loc[scored.index] = 100 * (scored / scored.min() - 1)
            best.loc[scored.idxmin()] = True
        table[f'place_{name}'], table[f'above_best_{name}'], table[f'best_{name}'] = place, above, best
    return table


# ----------------------------------------------------------------------------------- the report

def evidence(best, others):
    """``clear``, ``some`` or ``weak``: how far the best value's places stand from the others',
    against the scatter that so few trials have by themselves."""
    best, others = np.asarray(best, dtype=float), np.asarray(others, dtype=float)
    if len(best) < ENOUGH or len(others) < ENOUGH:
        return 'weak'
    scatter = math.sqrt(best.var(ddof=1) / len(best) + others.var(ddof=1) / len(others))
    distance = others.mean() - best.mean()
    if scatter == 0:
        return 'clear' if distance > 0 else 'weak'
    return 'clear' if distance / scatter >= 2.5 else 'some' if distance / scatter >= 1.5 else 'weak'


def rank_correlation(first, second):
    """Spearman's, or NaN when there is too little to correlate."""
    frame = pd.DataFrame({'a': pd.to_numeric(first, errors='coerce'), 'b': pd.to_numeric(second, errors='coerce')}).dropna()
    if len(frame) < 4 or frame.a.nunique() < 2 or frame.b.nunique() < 2:
        return NAN
    return float(frame.a.rank().corr(frame.b.rank()))


def _is_numeric(values):
    """A number read along its range: many different values, or any that is not whole (a rate)."""
    values = [v for v in values if not (isinstance(v, str) and v == 'none')]
    if not values or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return False
    return len(set(values)) > 5 or any(float(v) != int(v) for v in values)


def _show(value, digits=1):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ''
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def _label(value):
    """A value as a table names it: 100.0 read back from a csv is the 100 that was drawn."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _lines(headers, rows, indent='  ', left=1, last_left=False):
    """A plain table: the first ``left`` columns (and the last, on request) to the left, numbers to the right."""
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [max([len(str(header))] + [len(row[at]) for row in cells]) for at, header in enumerate(headers)]
    final = len(headers) - 1
    def line(row):
        return (indent + '  '.join(str(cell).ljust(width) if at < left or (last_left and at == final) else str(cell).rjust(width)
                                   for at, (cell, width) in enumerate(zip(row, widths)))).rstrip()
    return [line(headers)] + [line(row) for row in cells]


def grouped(present, name):
    """``(groups, used, numeric)``: the trials that have this parameter as ``(label, members)`` pairs
    by its value; the trials that were grouped; and whether it was read as a number, in which case
    the groups are the thirds of its range."""
    values = list(present[name])
    if _is_numeric(values):
        numbers = present[present[name].map(lambda v: not isinstance(v, str))]
        column = numbers[name].astype(float)
        low, high = column.quantile(1 / 3), column.quantile(2 / 3)
        parts = [('low', column <= low), ('middle', (column > low) & (column <= high)), ('high', column > high)]
        groups = [(f'{part:6} {column[mask].min():.4g} .. {column[mask].max():.4g}', numbers[mask])
                  for part, mask in parts if mask.any()]
        return groups, numbers, True
    labels = present[name].map(_label)
    return [(label, present[labels == label]) for label in sorted(labels.unique())], present, False


def parameter_section(sub, name, seen):
    """``(summary row or None, lines)`` for one hyper-parameter of one method."""
    present = sub[sub[name].notna()]
    title = name
    if name == 'learning_rate' and 'optimizer' in sub.columns:
        own = present['optimizer'].isin(OWN_STEP)
        if own.any():
            title = (f"{name} ({int((~own).sum())} trials; the {int(own.sum())} of "
                     f"{', '.join(sorted(set(present.loc[own, 'optimizer'])))} are left out: that optimizer finds its own step)")
            present = present[~own]
    if present[name].map(repr).nunique() < 2:
        return None, []
    if _is_numeric(list(present[name])) and len(present) < 2 * ENOUGH:
        return None, [f'  {title}: too few trials to read a number along its range; they are in {TABLE}', '']
    groups, used, numeric = grouped(present, name)
    if not numeric and len(groups) > max(8, len(present) // 3):
        return None, [f'  {name}: {len(groups)} different values among {len(present)} trials, too many to group; '
                      f'they are in {TABLE}', '']
    signature = frozenset(frozenset(members.index) for _, members in groups)
    if signature in seen:
        return None, [f'  {name}: groups the trials exactly as {seen[signature]} does; see there', '']
    seen[signature] = name
    rows, stats = [], []
    for label, members in groups:
        places = {slice_: members[f'place_{slice_}'].dropna() for slice_ in SLICES}
        stats.append((label, members, places))
        rows.append([label, len(members), _show(places['selection'].mean()), _show(members['above_best_selection'].mean()),
                     int(members['best_selection'].sum()), _show(places['holdout'].mean()),
                     _show(members['above_best_holdout'].mean()), int(members['kept'].sum()), _show(members['minutes'].mean())])
    means = [places['selection'].mean() if len(places['selection']) else NAN for _, _, places in stats]
    order = sorted(range(len(stats)), key=lambda at: (math.isnan(means[at]), means[at]))     # the best first
    if not numeric:
        rows = [rows[at] for at in order]
    lines = [f'  {title}']
    if numeric:
        lines.append('    rank correlation of the value with the place: selection '
                     f"{_show(rank_correlation(used[name], used['place_selection']), 2) or 'n/a'}, holdout "
                     f"{_show(rank_correlation(used[name], used['place_holdout']), 2) or 'n/a'} "
                     '(above 0: a larger value is worse)')
    lines += _lines(['value', 'trials', 'place sel', 'above best %', 'best of study', 'place hold', 'above best %',
                     'kept', 'minutes'], rows, indent='    ') + ['']
    eligible = [at for at in order if len(stats[at][2]['selection']) >= ENOUGH]
    if not eligible:
        return None, lines
    best = eligible[0]
    label, members, places = stats[best]
    others = np.concatenate([stats[at][2]['selection'].to_numpy() for at in range(len(stats)) if at != best] or [[]])
    on_holdout = [at for at in eligible if len(stats[at][2]['holdout']) >= ENOUGH]
    if not on_holdout:
        holds = 'not scored'
    else:
        winner = min(on_holdout, key=lambda at: stats[at][2]['holdout'].mean())
        holds = 'agrees' if winner == best else f'differs: {" ".join(str(stats[winner][0]).split())}'
    summary = [name, ' '.join(str(label).split()), len(places['selection']), _show(places['selection'].mean()),
               _show(float(others.mean())) if len(others) else '', evidence(places['selection'], others), holds]
    return summary, lines


def report(table, searched_columns, constants=None, *, case='', sizes=None):
    """The text of ``hyper_parameter_report.txt`` for a placed table; ``searched_columns`` as ``collect`` gives them."""
    constants = constants or {}
    studies = table.groupby(['cluster', 'method'], sort=False).ngroups
    out = [f"Hyper-parameters of {case or 'the case'}: {len(table)} finished trials of {studies} studies (cluster x method)",
           f"written by the evaluate stage on {time.strftime('%Y-%m-%d %H:%M')}; every trial is in {TABLE}"]
    if sizes:
        out.append(f"evaluated on {sizes.get('selection', 0)} selection and {sizes.get('holdout', 0)} holdout dates")
    out.append('')
    out += ['How to read it',
            '  selection     the dates that choose: a hyper-parameter is judged on them (sel)',
            '  holdout       the dates nothing was chosen from: whether that judgment holds (hold). Picking a value for',
            '                its holdout number makes the holdout one more selection slice.',
            "  place         a trial's place in its own study by its error: 0 its best trial, 100 its worst. Clusters are",
            '                not equally hard, so trials of different studies are compared by place, not by error.',
            "  above best %  how far a trial's error is above the best trial of its study",
            '  best of study how many studies have their best trial (selection dates) at this value',
            '  kept          trials the selection kept for the ensemble',
            '  evidence      how far the best value stands from the others against the scatter of so few trials:',
            '                clear, some or weak. Weak means: do not act on it.', '']
    methods = list(dict.fromkeys(table['method']))
    summaries, details = [], []
    for method in methods:
        sub = table[table['method'] == method]
        details.append(f"{method}: {len(sub)} trials of {sub['cluster'].nunique()} studies, "
                       f"{int(sub['place_selection'].notna().sum())} placed on the selection dates")
        shared = constants.get(method) or {}
        if shared:
            details += textwrap.wrap('the same in every trial: ' + ', '.join(f'{k}={_label(v)}' for k, v in shared.items()),
                                     width=110, initial_indent='  ', subsequent_indent='    ')
        details.append('')
        seen, any_parameter = {}, False
        for name in searched_columns.get(method, ()):
            if name not in sub.columns or sub[name].notna().sum() == 0:
                continue
            summary, lines = parameter_section(sub, name, seen)
            details += lines
            any_parameter = any_parameter or bool(lines)
            if summary:
                summaries.append([method] + summary)
        if not any_parameter:
            details += ['  no parameter has more than one value inside a study: nothing was searched, or the trials',
                        '  keep no record of their parameters', '']
    out.append('1. The best value of each hyper-parameter (judged on the selection dates)')
    if summaries:
        out += _lines(['method', 'hyper-parameter', 'best value', 'trials', 'its place', "the others'", 'evidence',
                       'holdout'], summaries, left=3, last_left=True)
    else:
        out.append('  no hyper-parameter has a value with enough trials to be judged')
    out += ['', '2. Every hyper-parameter, method by method', ''] + details
    out += ['3. The best trial of each study (selection dates)'] + _best_trials(table, searched_columns) + ['']
    out += ['4. Does the training tell the error on unseen dates?'] + _training_check(table, methods) + ['']
    repeated = _repeated_sets(table, searched_columns, methods)
    if repeated:
        out += ['5. The same parameter set trained in several studies'] + repeated + ['']
    missing = int(table['mae_selection'].isna().sum())
    if missing:
        out += [f'{missing} trial(s) have no current prediction on the evaluation period and so no error here. Running',
                'every stage predicts every trial: python run_combine.py --config <case>', '']
    return '\n'.join(out).rstrip() + '\n'


def _best_trials(table, searched_columns):
    best = table[table['best_selection']]
    if best.empty:
        return ['  no study has two trials with an error on the selection dates']
    rows = []
    for _, row in best.sort_values(['method', 'cluster']).iterrows():
        values = ', '.join(f'{name}={_value(row[name])}' for name in searched_columns.get(row['method'], ())
                           if not _missing(row[name]) and name != 'layers')
        rows.append([f"{row['cluster']} {row['method']} {row['trial']}", _show(row['mae_selection'], 3),
                     _show(row['mae_holdout'], 3), _show(row['place_holdout']), values])
    return _lines(['study and trial', 'MAE sel', 'MAE hold', 'place hold', 'parameters'], rows, last_left=True)


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _value(value):
    return f'{value:.3g}' if isinstance(value, float) and not value.is_integer() else _label(value)


def _training_check(table, methods):
    """The objective and the end of each training against the errors on the unseen dates."""
    lines = ['  objective = the value the study minimised (a pruned trial: its value when it was stopped). The rank',
             '  correlations are inside the studies: +1 the same order as on the unseen dates, 0 no relation.']
    rows = []
    for method in methods:
        sub = table[table['method'] == method].copy()
        sub['place_objective'] = NAN
        agree = {name: 0 for name in SLICES}
        compared = 0
        for _, members in sub.groupby('cluster', sort=False).groups.items():
            scored = sub.loc[members, 'objective'].dropna()
            if len(scored) < 2:
                continue
            sub.loc[scored.index, 'place_objective'] = 100 * (scored.rank(method='average') - 1) / (len(scored) - 1)
            compared += 1
            for name in SLICES:
                agree[name] += bool(sub.loc[scored.idxmin(), f'best_{name}'])
        if not compared:
            continue
        rows.append([method, compared, _show(rank_correlation(sub['place_objective'], sub['place_selection']), 2),
                     _show(rank_correlation(sub['place_objective'], sub['place_holdout']), 2),
                     _show(rank_correlation(sub['place_selection'], sub['place_holdout']), 2),
                     f"{agree['selection']} of {compared}", f"{agree['holdout']} of {compared}"])
    if rows:
        lines += _lines(['method', 'studies', 'objective~sel', 'objective~hold', 'sel~hold', 'best by objective is best on sel',
                         'on hold'], rows)
    else:
        lines.append('  no study has two trials with an objective value')
    states = table[table['state'].notna()]
    if len(states):
        rows = [[f'{method} {state}', len(members), _show(members['place_selection'].mean()),
                 _show(members['place_holdout'].mean()), int(members['kept'].sum()), _show(members['minutes'].mean())]
                for (method, state), members in states.groupby(['method', 'state'], sort=True)]
        lines += ['', '  how the trainings ended, and where those trials stand on the unseen dates'] + _lines(
            ['method and end', 'trials', 'place sel', 'place hold', 'kept', 'minutes'], rows)
    return lines


def _repeated_sets(table, searched_columns, methods):
    """Parameter sets that several studies trained: the cleanest comparison there is, cluster by cluster."""
    rows = []
    for method in methods:
        sub = table[table['method'] == method]
        columns = [name for name in searched_columns.get(method, ())
                   if name in sub.columns and sub[name].notna().any() and name != 'layers']
        if not columns:
            continue
        exact = sub[columns].apply(lambda row: '|'.join(repr(row[c]) for c in columns), axis=1)     # not the rounded text
        for _, members in sub.groupby(exact, sort=False):
            if members['cluster'].nunique() >= 2:
                first = members.iloc[0]
                text = ', '.join(f'{c}={_value(first[c])}' for c in columns if not _missing(first[c]))
                rows.append((members['place_selection'].mean(), [method, members['cluster'].nunique(),
                             _show(members['place_selection'].mean()), _show(members['place_holdout'].mean()),
                             int(members['best_selection'].sum()), text]))
    rows.sort(key=lambda pair: (math.isnan(pair[0]), pair[0]))
    if not rows:
        return []
    shown = [row for _, row in rows[:20]]
    lines = _lines(['method', 'studies', 'place sel', 'place hold', 'best of study', 'parameters'], shown, last_left=True)
    if len(rows) > len(shown):
        lines.append(f'  ... and {len(rows) - len(shown)} more')
    return lines


# ------------------------------------------------------------------------------------- writing

def write(static_data, targets, specs, trial_predictions, path, *, metrics, current=None, say=None):
    """Both files into ``path``; the table, or None when there was nothing to say or it could not be said."""
    say = say or log.warning
    try:
        table, by_method, constants = collect(static_data, targets, specs, trial_predictions, metrics=metrics,
                                              current=current)
        if table.empty:
            return None
        table = placed(table)
        names = columns_of(by_method)
        front = ['cluster', 'method', 'trial', 'kept', 'state'] + names
        measured = [f'{kind}_{name}' for kind in ('mae', 'place', 'above_best', 'best', 'n') for name in SLICES]
        table = table[front + measured + ['objective', 'val_error', 'test_error', 'epochs', 'minutes']]
        table = table.sort_values(['method', 'cluster', 'place_selection', 'trial'], na_position='last').reset_index(drop=True)
        os.makedirs(path, exist_ok=True)
        table.to_csv(os.path.join(path, TABLE), index=False, float_format='%.6g')
        sizes = {name: len(getattr(targets, name, ())) for name in SLICES}
        text = report(table, by_method, constants, case=str(static_data.get('_id') or ''), sizes=sizes)
        with open(os.path.join(path, REPORT), 'w', encoding='utf-8') as stream:
            stream.write(text)
        log.info('hyper-parameters: %d trials of %d studies, %d searched parameter(s) -> %s and %s in %s', len(table),
                 table.groupby(['cluster', 'method']).ngroups, len(names), TABLE, REPORT, path)
        return table
    except Exception as error:  # noqa: BLE001 -- a report must not fail the evaluation it describes
        say('the hyper-parameter report was not written (%s: %s); the evaluation itself is complete',
            type(error).__name__, error)
        return None
