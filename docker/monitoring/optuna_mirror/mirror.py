# eForecast project author: George Sideratos and Claude
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Copies the studies of the cases into the database the Optuna dashboard reads, one study there
for each case and cluster.

The trainings keep each study -- one for a case, a cluster and a method -- in a file beside its
models; the Optuna dashboard reads a database, and lists what is in it in one row, without groups.
Copied one for one that was 87 studies on 25 November 2026, and 27 of them were of a method that
asks for a single trial, where the dashboard's charts have nothing to compare and stay empty. So
the copy is grouped: ``Total_wind / RBF_rule_04`` holds the trials of every method of that
cluster. This asks the metrics service which studies there are (``/studies/index``), takes the
ones of every group in which a study changed (``/studies/export``), and writes them:

* a trial says which method it is of in the parameter ``(method)`` -- the name sorts first, so
  the charts that take a parameter open on it and show every trial -- and carries its own
  parameters under the method's name, ``MLP.learning_rate``: a chart of one of those shows that
  method's trials;
* its attributes say where it comes from: ``method``, ``folder`` (``test_12``, the folder the
  training keeps it in) and ``source`` (the training's study and the trial's number there). The
  number a trial has in the dashboard is its place in the history of the cluster, **not** that
  number: several methods cannot all have a trial 12;
* a running trial is copied as running and brought up to date on the next rounds -- its values
  after every epoch, then how it ended;
* only into studies it made itself (it marks them): a study of the same name that something else
  made is left alone and said;
* a group is written anew when it cannot be added to: a training began one of its studies again,
  or a method joined the cluster (the choices of ``(method)`` cannot grow). The trials of a study
  the metrics service does not offer at that moment -- a case whose folder moved -- are kept;
* the copies an earlier version of this made, one for every method, are folded into the groups
  and removed: from the training's study where the metrics service offers it, from the copy
  itself where it does not.

It reads the trainings' work over HTTP and never opens a model folder; it needs Optuna and the
database driver and nothing of the pipeline, so it runs in the dashboard's own image. Settings:

    OPTUNA_MIRROR_STORAGE   the dashboard's database (required)
    EFORECAST_STUDIES_URL   the metrics service, default http://host.docker.internal:9109
                            (file:///folder reads index.json and <name>.json there instead)
    OPTUNA_MIRROR_EVERY     seconds between rounds, default 60

    python mirror.py          copy, round after round
    python mirror.py --once   one round, then say what was done
"""
import datetime
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import optuna
from optuna.distributions import CategoricalDistribution, distribution_to_json, json_to_distribution
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

#: The study attribute that says a study in the database is a copy made here.
MARK = 'eforecast_mirror'
#: What that attribute says of how the copy is laid out: 2 is one study for a case and a cluster.
LAYOUT = 2
#: The parameter that says which method a trial is of. The dashboard lists parameters by name and
#: opens its charts on the first; ``(`` sorts before every letter and digit.
METHOD = '(method)'
#: Where ``seen`` remembers that no copy of the earlier layout is left to fold in.
FOLDED = ('earlier copies are folded in',)


def reader(source):
    """``fetch(path)`` for the metrics service, or for a folder of exports (``file:///...``)."""
    if source.startswith('file://'):
        folder = urllib.parse.urlparse(source).path

        def fetch(path):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
            name = 'index' if path.startswith('/studies/index') else query['name'][0]
            with open(os.path.join(folder, name + '.json'), encoding='utf-8') as stream:
                return json.load(stream)
        return fetch

    def fetch(path):
        with urllib.request.urlopen(source.rstrip('/') + path, timeout=60) as answer:
            return json.load(answer)
    return fetch


def cluster_label(cluster):
    """``RBF rule_4`` as ``RBF_rule_04``: one word, numbered so that the names sort as the clusters count."""
    text = '_'.join(str(cluster or 'global').split())
    stem = text.rstrip('0123456789')
    return stem + text[len(stem):].zfill(2) if len(stem) < len(text) else text


def group_name(case, cluster):
    return f'{case} / {cluster_label(cluster)}'


def _order(key):
    """A case's groups in the order a person reads them: the global one, then the clusters as they count."""
    case, cluster = key
    return str(case), cluster != 'global', cluster_label(cluster)


def _moment(text):
    return None if text is None else datetime.datetime.fromisoformat(text)


def _text(moment):
    return None if moment is None else moment.isoformat()


def source_of(member, number):
    """What a trial's ``source`` attribute says: the training's study, and the trial's number in it."""
    return f'{member}#{number}'


def frozen(trial, method, member, methods):
    """A trial of an export as the study of its group holds it."""
    distributions = {METHOD: CategoricalDistribution(methods)}
    distributions.update({f'{method}.{name}': json_to_distribution(text) for name, text in trial['distributions'].items()})
    params = {METHOD: method}
    params.update({f'{method}.{name}': value for name, value in trial['params'].items()
                   if f'{method}.{name}' in distributions})
    attrs = {**trial['user_attrs'], 'method': method, 'folder': f"test_{trial['number']}",
             'source': source_of(member, trial['number'])}
    return FrozenTrial(number=0, trial_id=0, state=TrialState[trial['state']], value=None, values=trial['values'],
                       datetime_start=_moment(trial['datetime_start']), datetime_complete=_moment(trial['datetime_complete']),
                       params=params, distributions=distributions, user_attrs=attrs, system_attrs={},
                       intermediate_values={int(step): value for step, value in trial['intermediate_values'].items()})


def carried(have, methods):
    """A trial the group already holds, for the group written anew: itself, among these methods."""
    return FrozenTrial(number=0, trial_id=0, state=have.state, value=None, values=have.values,
                       datetime_start=have.datetime_start, datetime_complete=have.datetime_complete,
                       params=dict(have.params), distributions={**have.distributions, METHOD: CategoricalDistribution(methods)},
                       user_attrs=dict(have.user_attrs), system_attrs={},
                       intermediate_values=dict(have.intermediate_values))


def exported(storage, study):
    """A copy of the earlier layout -- one method's study -- as the export it was made from."""
    mark = study.user_attrs[MARK]
    trials = [{'number': trial.number, 'state': trial.state.name, 'values': trial.values,
               'datetime_start': _text(trial.datetime_start), 'datetime_complete': _text(trial.datetime_complete),
               'params': dict(trial.params),
               'distributions': {name: distribution_to_json(found) for name, found in trial.distributions.items()},
               'intermediate_values': {str(step): value for step, value in trial.intermediate_values.items()},
               'user_attrs': dict(trial.user_attrs)}
              for trial in storage.get_all_trials(study._study_id, deepcopy=False)]
    return {'name': study.study_name, 'case': mark.get('case'), 'cluster': mark.get('cluster'), 'method': mark.get('method'),
            'direction': study.direction.name, 'trials': trials, 'version': mark.get('version')}


def earlier_copies(storage):
    """``{(case, cluster): [study, ...]}`` of the copies made one for every method, before the groups."""
    found = {}
    for study in storage.get_all_studies():
        mark = study.user_attrs.get(MARK)
        if isinstance(mark, dict) and mark.get('layout') is None and mark.get('case') and mark.get('method'):
            found.setdefault((mark['case'], mark.get('cluster') or 'global'), []).append(study)
    return found


def bring_up_to_date(storage, have, wanted):
    """A trial that was copied while it ran: what it has learnt since, and how it ended.

    ``ended``, ``updated``, or ``same`` when the training has said nothing new about it."""
    written = 0
    if have.state == TrialState.WAITING and wanted.state != TrialState.WAITING:
        storage.set_trial_state_values(have._trial_id, TrialState.RUNNING)
        written += 1
    for name, value in wanted.params.items():
        if name not in have.params:
            distribution = wanted.distributions[name]
            storage.set_trial_param(have._trial_id, name, distribution.to_internal_repr(value), distribution)
            written += 1
    for step, value in wanted.intermediate_values.items():
        if have.intermediate_values.get(step) != value:
            storage.set_trial_intermediate_value(have._trial_id, step, value)
            written += 1
    for name, value in wanted.user_attrs.items():
        if have.user_attrs.get(name) != value:
            storage.set_trial_user_attr(have._trial_id, name, value)
            written += 1
    if wanted.state.is_finished():
        storage.set_trial_state_values(have._trial_id, wanted.state, wanted.values)
        return 'ended'
    return 'updated' if written else 'same'


def _first(whole):
    return next((trial['datetime_start'] for trial in whole['trials'] if trial['datetime_start']), None)


def _when(trial):
    """Where a trial stands in the history of its cluster: by when it began, the unbegun ones last."""
    member, _, number = str(trial.user_attrs.get('source', '')).rpartition('#')
    return (trial.datetime_start is None, trial.datetime_start or datetime.datetime.min,
            str(trial.user_attrs.get('method', '')), member, int(number) if number.isdigit() else 0)


def copy(storage, key, members):
    """One group into the database: ``{'new': n, 'updated': n, 'ended': n, 'same': n}`` or ``{'foreign': 1}``.

    ``members`` is ``{method: export}`` of the group's studies as the trainings hold them now.
    ``again`` in the answer says why the group was written anew, when it was.
    """
    case, cluster = key
    name = group_name(case, cluster)
    try:
        study_id = storage.get_study_id_from_name(name)
    except KeyError:
        study_id = None
    held, before = [], {}
    if study_id is not None:
        mark = storage.get_study_user_attrs(study_id).get(MARK)
        if not isinstance(mark, dict):
            return {'foreign': 1}                  # made by something else: not ours to write into
        held, before = storage.get_all_trials(study_id, deepcopy=False), mark.get('members') or {}
    of = {}                                        # method -> the trials of it the group holds
    for trial in held:
        of.setdefault(trial.user_attrs.get('method'), []).append(trial)
    methods = sorted({*members, *(method for method in of if method)})
    again = None
    for method, whole in members.items():
        ours, first = of.get(method, []), _first(whole)
        was = before.get(method) or {}
        if ours and (was.get('name') != whole['name'] or len(ours) > len(whole['trials'])
                     or (first and was.get('first') and was['first'] != first)):
            again = f'{method} had begun again'
            of[method] = []                        # what the group held of it is of the study that was
    recorded = list(mark.get('methods') or []) if held else []
    if held and again is None and not set(methods) <= set(recorded):
        again = 'a method joined'
    # what is added to a study names the methods as its trials already do; a study written anew, as they are now
    choices = recorded if held and again is None else methods
    wanted = {source_of(whole['name'], trial['number']): frozen(trial, method, whole['name'], choices)
              for method, whole in members.items() for trial in whole['trials']}
    did = {'new': 0, 'updated': 0, 'ended': 0, 'same': 0}
    if again is not None:
        # Nothing in Optuna removes a trial from a study or widens a parameter's choices: the study goes
        # and comes back with what the trainings hold, and with what it held of the studies not offered now.
        kept = [carried(trial, choices) for method, trials in of.items() if method not in members for trial in trials]
        storage.delete_study(study_id)
        study_id, held, did['again'] = None, [], again
        writes = sorted([*wanted.values(), *kept], key=_when)
    if study_id is None:
        direction = next((whole.get('direction') for whole in members.values() if whole.get('direction')), 'MINIMIZE')
        study_id = storage.create_new_study([StudyDirection[direction]], name)
    if again is None:
        have = {trial.user_attrs.get('source'): trial for trial in held}
        writes = []
        for source, trial in wanted.items():
            if source not in have:
                writes.append(trial)
            elif not have[source].state.is_finished():       # a finished trial does not change any more
                did[bring_up_to_date(storage, have[source], trial)] += 1
        writes.sort(key=_when)
    for trial in writes:
        storage.create_new_trial(study_id, template_trial=trial)
    did['new'] = len(writes) - (len(kept) if again is not None else 0)
    storage.set_study_user_attr(study_id, MARK, {
        'layout': LAYOUT, 'case': case, 'cluster': cluster, 'methods': choices,
        'members': {**{method: was for method, was in before.items() if method not in members and of.get(method)},
                    **{method: {'name': whole['name'], 'first': _first(whole), 'version': whole.get('version')}
                       for method, whole in members.items()}}})
    return did


def once(storage, fetch, seen, say=print):
    """One round over every group; ``seen`` remembers the versions that are copied."""
    index = fetch('/studies/index')
    totals = {'studies': 0, 'new': 0, 'updated': 0, 'ended': 0, 'same': 0, 'foreign': 0, 'failed': 0, 'folded': 0}
    groups, where = {}, seen.setdefault(('where',), {})
    for entry in index['studies']:
        if entry.get('case') is None or entry.get('method') is None:
            # an index that names the studies and no more: each is asked once what it is of
            if entry['name'] not in where:
                try:
                    whole = fetch('/studies/export?name=' + urllib.parse.quote(entry['name']))
                except Exception as error:         # noqa: BLE001 -- one study; the others are still copied
                    totals['failed'] += 1
                    say(f"{entry['name']}: not copied ({type(error).__name__}: {error})")
                    continue
                where[entry['name']] = (whole.get('case'), whole.get('cluster'), whole.get('method'))
            case, cluster, method = where[entry['name']]
            entry = {**entry, 'case': case, 'cluster': cluster, 'method': method}
        groups.setdefault((entry['case'], entry.get('cluster') or 'global'), {})[entry['method']] = entry
    earlier = {} if seen.get(FOLDED) else earlier_copies(storage)
    left = 0
    for key in sorted({*groups, *earlier}, key=_order):
        offered, copies, name = groups.get(key, {}), earlier.get(key, []), group_name(*key)
        version = tuple(sorted((entry['name'], entry['version']) for entry in offered.values()))
        if seen.get(name) == version and not copies:
            continue
        try:
            members = {method: fetch('/studies/export?name=' + urllib.parse.quote(entry['name']))
                       for method, entry in offered.items()}
            folded = [study for study in copies if study.user_attrs[MARK]['method'] not in members]
            for study in folded:                   # nothing offers it any more: the copy is what there is of it
                members[study.user_attrs[MARK]['method']] = exported(storage, study)
            did = copy(storage, key, members)
            if did.get('foreign'):
                left += len(copies)
            else:
                for study in copies:               # the group holds what they held
                    storage.delete_study(study._study_id)
                did['folded'] = len(copies)
        except Exception as error:                 # noqa: BLE001 -- one group; the others are still copied
            totals['failed'] += 1
            left += len(copies)
            say(f'{name}: not copied ({type(error).__name__}: {error})')
            continue
        seen[name] = version
        totals['studies'] += 1
        for word, value in did.items():
            if word != 'again':
                totals[word] = totals.get(word, 0) + value
        if did.get('foreign'):
            say(f'{name}: a study of this name is in the database and was not made by this copier; it is left alone')
        elif did['new'] or did['updated'] or did['ended'] or did.get('again') or did.get('folded'):
            say(f"{name}: {did['new']} new trial(s), {did['updated']} brought up to date, {did['ended']} ended"
                + (f" (written anew: {did['again']})" if did.get('again') else '')
                + (f"; {did['folded']} earlier copy(ies), one for each method, folded in and removed" if did.get('folded') else ''))
    if not left:
        seen[FOLDED] = True
    return totals


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    source = os.environ.get('EFORECAST_STUDIES_URL', 'http://host.docker.internal:9109')
    every = float(os.environ.get('OPTUNA_MIRROR_EVERY', '60') or 60)
    storage = optuna.storages.RDBStorage(os.environ['OPTUNA_MIRROR_STORAGE'])
    fetch, seen = reader(source), {}

    def say(text):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), text, flush=True)
    say(f'copying the studies of {source} into the Optuna dashboard\'s database, one for each case and cluster, '
        f'every {every:.0f} s')
    while True:
        try:
            totals = once(storage, fetch, seen, say)
            if '--once' in argv:
                say(f"one round: {totals['studies']} groups looked at, {totals['new']} new trials, "
                    f"{totals['updated'] + totals['ended']} brought up to date, {totals['foreign']} left alone, "
                    f"{totals['folded']} earlier copies folded in, {totals['failed']} failed")
                return 1 if totals['failed'] else 0
        except Exception as error:                 # noqa: BLE001 -- the source or the database is away: try again
            say(f'this round did not finish ({type(error).__name__}: {error})')
            if '--once' in argv:
                return 2
        time.sleep(every)


if __name__ == '__main__':
    sys.exit(main())
