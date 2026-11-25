# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A14-02 -- what a colleague needs in order to decide whether a result can be used.

The numbers already exist. `Results/*.csv` holds the evaluation the combine stage wrote, A15-02
holds the approval pointer and A15-03 holds the published forecast. What was missing is the
**labels**: which score answers which question, over what period and in what units, what is
absent, and whether the forecast on the screen is the one colleagues should be using.

Three rules this module exists to enforce:

* a score that chose something is not an estimate of accuracy, and is never shown as one;
* a completed run is not an approval -- "finished" and "fit for routine use" are different
  facts, and A15-02 owns the second;
* a failed attempt never makes the older valid forecast look newly generated.

Read-only: bounded CSV/payload reads and hashes of the named evaluation receipt outputs. No dataset, no model, no refit, and
**nothing is recomputed** -- a metric that is not in an artifact is reported missing rather
than derived, because a derived number in a trust summary is the one thing that must not
happen.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

# The order the serving ensemble itself prefers. Imported rather than repeated: a page that
# names a different predictor than the service serves is the defect this constant exists for.
from eforecast.bundle_serving import SERVED_PREFERENCE as DEFAULT_PREDICTORS, served_path
from eforecast.config.service import ServiceError
# The studies page builds the same paths from the same document; one way of asking where a
# case keeps its model is better than two that can disagree.
from eforecast.config.studies import _static as case_static

ROWS = 12                                           # what a summary shows of any one table
READ = 4000                                         # what it reads before giving up on a table

TUNING, SELECTION, FINAL, BANDS = 'tuning', 'selection', 'final', 'bands'

DEFAULT_PATH = ('models', DEFAULT_PREDICTORS[0])


def likely_served(final_rows):
    """Which predictor this case would serve, when nothing is approved yet.

    An approved bundle states the path it qualified, and that is the authority -- `summary()`
    uses it. Without one there is still an answer, and it is not a constant: a single-method
    case's models level is `single`, so a page that asks for `kmeans` regardless shows a
    colleague **no headline number at all**. Measured end to end on a one-method case, which
    is the state every case is in between `run_combine` and an approval.
    """
    return served_path({str(row.get('predictor')) for row in final_rows
                        if str(row.get('level')) == 'models'})

# Only these levels answer for the whole period. A cluster's rows cover the dates that cluster
# is active on, so its MAE is over a different -- and usually easier -- sample; `ensemble.py`
# draws the same line for its own "whole period" report. Comparing across the line is how a
# cluster predictor comes to look twice as accurate as the model that serves every hour.
WHOLE_PERIOD_LEVELS = ('models', 'cluster_average', 'global')

# Zones where the stored local time is ambiguous twice a year. A13-04 found this in both real
# cases; the decision about the data is the owner's, so this module labels and does not fix.
AMBIGUOUS = ('Europe/', 'US/', 'America/', 'Canada/', 'Australia/', 'NZ', 'Pacific/Auckland')


def _stat(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
    except OSError:
        return {'path': str(path), 'exists': False}
    return {'path': str(path), 'exists': True, 'modified': info.st_mtime,
            'size_bytes': info.st_size}


def _table(path: Path, sources: dict[str, Any], name: str) -> list[dict[str, str]]:
    """One artifact, read whole but bounded. Its absence is recorded, not raised."""
    sources[name] = _stat(path)
    if not sources[name]['exists']:
        return []
    rows: list[dict[str, str]] = []
    try:
        with open(path, newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
                if len(rows) >= READ:
                    sources[name]['truncated'] = True
                    break
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        sources[name]['error'] = f'{type(error).__name__}: {error}'
        return []
    sources[name]['rows'] = len(rows)
    return rows


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None       # NaN is not a score


def _int(value: Any) -> int | None:
    number = _float(value)
    return None if number is None else int(number)


def _metrics(row: dict[str, str]) -> dict[str, Any]:
    return {'predictor': row.get('predictor'), 'level': row.get('level'),
            'cluster': row.get('cluster'), 'mae': _float(row.get('mae')),
            'rmse': _float(row.get('rmse')), 'bias': _float(row.get('bias')),
            'n': _int(row.get('n'))}


# ------------------------------------------------------------------- which score, which question


def roles(final_rows, evaluation_rows, selection_rows, interval_rows, sources) -> list[dict]:
    """The scores a case has, each labelled with the question it actually answers.

    The order is deliberate: the honest estimate first, the scores that chose something after
    it. A colleague reading top to bottom reads the number that is true about unseen data
    before the numbers that are true about the data that picked the model.
    """
    out = [{
        'role': FINAL, 'label': 'Final evaluation (holdout)',
        'period': 'the holdout slice, read by this report and by nothing else',
        'estimates_accuracy': True,
        'caption': 'The only numbers here that estimate accuracy on data the models never '
                   'influenced. MAE, RMSE and bias are in percent of rated power.',
        'source': sources.get('summary', {}).get('path'),
        'rows': [_metrics(row) for row in final_rows[:ROWS]],
        'missing': not final_rows and 'no final evaluation has been recorded for this case' or None,
    }, {
        'role': SELECTION, 'label': 'Selection score (selection slice)',
        'period': 'the selection slice, which chose these trials and fitted the combiners',
        'estimates_accuracy': False,
        'caption': 'These numbers chose what was kept. The rows that chose a model flatter it, '
                   'so this is a ranking, not an estimate of how it will do next week.',
        'source': sources.get('selection', {}).get('path'),
        'rows': [{'cluster': row.get('cluster'), 'method': row.get('method'),
                  'trial': row.get('trial'), 'mae': _float(row.get('mae_selection')),
                  'n': _int(row.get('n')), 'kept': str(row.get('kept', '')).lower() == 'true'}
                 for row in selection_rows[:ROWS]],
        'missing': not selection_rows and 'no selection table was written' or None,
    }]
    if interval_rows:
        out.append({
            'role': BANDS, 'label': 'Interval coverage',
            'period': 'the slice the bands were scored on',
            'estimates_accuracy': False,
            'caption': 'What the bands promised against what they delivered. A coverage below '
                       'its target means the interval is narrower than it claims.',
            'source': sources.get('intervals', {}).get('path'),
            'rows': [{'level': row.get('level'), 'predictor': row.get('predictor'),
                      'method': row.get('method'),
                      'target': _float(row.get('coverage_target')),
                      'achieved': _float(row.get('coverage')),
                      'width': _float(row.get('width')), 'n': _int(row.get('n'))}
                     for row in interval_rows[:ROWS]],
            'missing': None,
        })
    # The tuning scores live in the Optuna studies, which the studies page already loads. A
    # trust summary should not open a study database to repeat a number that chose a trial.
    out.append({
        'role': TUNING, 'label': 'Tuning scores (per-trial objective)',
        'period': 'whatever each study optimised',
        'estimates_accuracy': False,
        'caption': 'On the "Studies & results" tab. A trial objective chose one trial out of '
                   'many; comparing it with the numbers above compares different questions.',
        'source': None, 'rows': [], 'missing': None,
    })
    return out


def headline(final_rows, served_path=DEFAULT_PATH) -> dict[str, Any] | None:
    """The result a colleague actually receives, and the better one they must not simply take.

    `served_path` is the approved bundle's qualified path when there is an approval, and
    otherwise what this case's own ensemble would choose. A different predictor scoring better on the holdout is worth seeing -- and worth
    a warning, because choosing it *because* it won here turns the holdout into another
    selection slice and the honest number stops being honest.
    """
    if not final_rows:
        return None
    rows = [_metrics(row) for row in final_rows]
    comparable = [row for row in rows
                  if row['mae'] is not None and row['level'] in WHOLE_PERIOD_LEVELS]
    served = next((row for row in rows if (row['level'], row['predictor']) == tuple(served_path)), None)
    best = min(comparable, key=lambda row: row['mae']) if comparable else None
    out = {'served': served, 'best': best, 'is_best': bool(served and best and served is best)}
    out['served_path'] = list(served_path)
    if served is None:
        out['note'] = (f'the predictor this case serves ({served_path[0]} / {served_path[1]}) '
                       'is not in this evaluation; the figures below are other predictors')
    elif best and served is not best and served['mae'] is not None and best['mae'] is not None:
        out['note'] = (f'{best["level"]}/{best["predictor"]} scores better here, over the same '
                       'period. Switching to it because it won on the holdout would make the '
                       'holdout a selection slice: decide on evidence from outside this table.')
    out['compared'] = ('predictors that answer for the whole period; a cluster scores only on '
                       'the hours its cluster is active, so it is not compared here')
    return out


# ------------------------------------------------------------------- period, units, coverage


def coverage(rows, sources) -> dict[str, Any]:
    """How much of each slice the predictors actually answered.

    `n` in the evaluation table counts finite target cells, which is a different question: a
    predictor that answers nine of ten issue times with sixteen horizons each looks full. This
    table is the one that shows a gap as a gap.
    """
    if not rows:
        return {'recorded': False,
                'detail': 'no coverage table was written, so missing timestamps cannot be '
                          'ruled out; these results predate the coverage report or the '
                          'evaluation did not finish',
                'slices': [], 'worst': []}
    slices: dict[str, dict[str, int]] = {}
    for row in rows:
        name = row.get('slice') or 'unknown'
        entry = slices.setdefault(name, {'requested': 0, 'covered': 0, 'missing': 0,
                                         'predictors': 0})
        entry['requested'] = max(entry['requested'], _int(row.get('requested')) or 0)
        entry['covered'] = max(entry['covered'], _int(row.get('covered')) or 0)
        entry['missing'] = max(entry['missing'], _int(row.get('missing')) or 0)
        entry['predictors'] += 1
    worst = sorted((row for row in rows if (_int(row.get('missing')) or 0) > 0),
                   key=lambda row: -(_int(row.get('missing')) or 0))[:ROWS]
    return {'recorded': True, 'source': sources.get('coverage', {}).get('path'),
            'slices': [{'slice': name, **values} for name, values in sorted(slices.items())],
            'worst': [{'level': row.get('level'), 'cluster': row.get('cluster'),
                       'predictor': row.get('predictor'), 'slice': row.get('slice'),
                       'requested': _int(row.get('requested')),
                       'covered': _int(row.get('covered')),
                       'missing': _int(row.get('missing'))} for row in worst]}


def period(static_data, evaluation_rows, coverage_block, sources) -> dict[str, Any]:
    """The period these numbers describe, as the artifacts record it.

    The tables carry slice names and row counts, not dates, so nothing here invents a date
    range. What it can do is compare the protocol the case declares with the one the artifacts
    were written under: a case that asks for a calibration slice and results that have none
    were produced by different rules, and the numbers are not comparable across that line.
    """
    # The initializer flattens these; `Evaluation_start` keeps its capital from the case file.
    declared = {'evaluation_start': static_data.get('Evaluation_start'),
                'selection_fraction': static_data.get('selection_fraction', 0.5),
                'calibration_fraction': static_data.get('calibration_fraction', 0.0)}
    present = sorted({row.get('slice') for row in evaluation_rows if row.get('slice')})
    slices = {entry['slice']: entry for entry in coverage_block.get('slices') or []}
    known = set(present) | set(slices)
    out = {'declared': declared, 'slices_present': present,
           'rows_per_slice': {name: entry['requested'] for name, entry in slices.items()},
           'written_at': (sources.get('summary') or {}).get('modified'),
           'protocol': ('three slices: selection chose the trials, calibration fitted the '
                        'bands, holdout was kept back for this report' if 'calibration' in known
                        else 'two slices in these artifacts: selection and holdout')}
    if float(declared['calibration_fraction'] or 0) > 0 and 'calibration' in known:
        out['note'] = None
    elif float(declared['calibration_fraction'] or 0) > 0 and known:
        out['note'] = ('the case asks for a separate calibration slice and these artifacts have '
                       'none, so they were produced under the earlier two-slice protocol; the '
                       'bands were fitted on the rows that chose the predictor')
    return out


def produced_by(document, runs, static=None) -> dict[str, Any]:
    """Which run wrote these numbers, and whether the case still looks like it did then.

    A run records the digest of the document it accepted, so this is an exact answer where
    file times are a guess: if that digest is not the case as it stands now, these numbers
    describe a configuration that no longer exists. Saying which run also answers the "what
    version am I looking at" half of the question, which a timestamp cannot.
    """
    from eforecast.config import case_revision
    unknown = {'state': 'unknown', 'detail': 'no verified receipt binds this evaluation to a producing run'}
    if runs is None or static is None:
        return unknown
    try:
        from eforecast.common_utils import lineage
        receipt = json.loads((Path(static['path_model']) / '.stage-artifacts/receipts/evaluate.json').read_text())
        if receipt.get('state') != 'complete' or receipt.get('id') != lineage.dependency_id(
                {k:v for k,v in receipt.items() if k != 'id'}):
            return unknown
        summary = Path(static['path_model']) / 'Results/summary.csv'
        records = [r for rows in receipt.get('outputs', {}).values() for r in rows]
        bound = next((r for r in records if Path(r['path']).resolve() == summary.resolve()), None)
        if not bound or bound.get('sha256') != lineage.digest_file(str(summary)):
            return unknown
        producer = receipt.get('producer', {})
        finished = [run for run in runs() if run.get('id') == producer.get('run_id')
                    and run.get('status') == 'finished'
                    and not (run.get('options') or {}).get('dry_run')
                    and '--dry-run' not in (run.get('argv') or [])]
        if len(finished) != 1:
            return unknown
        latest = finished[0]
    except (OSError, ValueError, TypeError, KeyError):
        return unknown
    accepted = latest.get('accepted_case_revision')
    out = {'run_id': latest.get('id'), 'finished': latest.get('finished'),
           'accepted_case_revision': accepted}
    if not accepted:
        out.update(state='unknown',
                   detail='the run that produced these results did not record which revision '
                          'of the case it accepted')
        return out
    out['case_revision'] = case_revision.revision(document)
    if accepted == out['case_revision']:
        out.update(state='current',
                   detail='these results were produced from the case exactly as it stands now')
    else:
        out.update(state='case_changed',
                   detail='the case has been edited since the run that produced these results, '
                          'so they describe an earlier configuration',
                   remedy='rerun the combine and evaluation stage, or compare the review tab '
                          'before trusting these numbers')
    return out


def freshness(run_block, sources) -> dict[str, Any]:
    """Do these numbers describe the case as it is now?

    The run's accepted revision answers it exactly when there is one. Otherwise the file times
    answer it roughly, and the reply says which of the two it is -- a rough answer presented as
    an exact one is how a trust summary stops being trustworthy.
    """
    written = (sources.get('summary') or {}).get('modified')
    out = {'written_at': written, 'run_id': run_block.get('run_id')}
    if written is None:
        return {**out, 'state': 'none', 'evidence': 'none',
                'detail': 'no evaluation artifacts exist for this case yet'}
    if run_block.get('state') in {'current', 'case_changed'}:
        return {**out, 'state': run_block['state'], 'evidence': 'the run\'s accepted revision',
                'detail': run_block['detail'],
                **({'remedy': run_block['remedy']} if run_block.get('remedy') else {})}
    edited = (run_block.get('document_modified')
              if run_block.get('document_modified') else None)
    if edited and edited > written:
        return {**out, 'state': 'case_changed', 'evidence': 'file times', 'edited_at': edited,
                'detail': 'the case file was edited after these results were written, so they '
                          'may describe an earlier configuration',
                'remedy': 'rerun the combine and evaluation stage, or compare the review tab '
                          'before trusting these numbers'}
    return {**out, 'state': 'unknown', 'evidence': 'file times only', 'edited_at': edited,
            'detail': 'evaluation exists, but its producer and case revision are not verified'}


def units(static_data, calibration) -> dict[str, Any]:
    """What the numbers mean. A metric without its unit is a number without a meaning."""
    rated = static_data.get('rated')
    recorded = (calibration or {}).get('rated')
    out = {'rated': rated, 'rated_recorded': recorded,
           'target': static_data.get('type'),
           'metrics': 'percent of rated power' if rated else 'percent of the target',
           'steps': len((calibration or {}).get('columns') or []) or static_data.get('horizon'),
           'resolution': static_data.get('ts_resolution'),
           'horizon_type': static_data.get('horizon_type')}
    if rated and recorded and abs(float(rated) - float(recorded)) > 1e-6:
        out['warning'] = (f'the evaluation was scored against a rated power of {recorded}, and '
                          f'the case now declares {rated}; the percentages are in the old unit')
    return out


def caveats(static_data) -> list[dict[str, str]]:
    """What is known to be wrong with the inputs, said plainly, without pretending it is fixed."""
    found = []
    zone = str(static_data.get('local_timezone') or '')
    step = str(static_data.get('ts_resolution') or '')
    if zone.startswith(AMBIGUOUS) and step not in {'D', 'd'}:
        found.append({
            'code': 'dst_ambiguous',
            'detail': f'the series are stored in local time in {zone}, where one hour occurs '
                      'twice each autumn and one does not exist each spring',
            'consequence': 'those hours are shifted or dropped on import, and a model can learn '
                           'the shift as if it were weather',
            'remedy': 'an owner decision: store the series '
                      'in UTC and re-derive the datasets, or accept two hours a year knowingly'})
    return found


# ------------------------------------------------------------------- approved, or experimental


def model(store, case, name=None) -> dict[str, Any]:
    """Which version this case uses for routine forecasts -- A15-02's pointer, not a run badge.

    A finished run produces a model. It does not produce permission to use it, and the two are
    shown apart on purpose: "experimental" here is not a criticism of the numbers, it is the
    plain fact that nobody approved them.
    """
    if store is None:
        return {'state': 'unknown',
                'detail': 'the durable store is not reachable from here, so the approved '
                          'version cannot be read'}
    from eforecast.config import model_approval
    try:
        current = model_approval.active(store, case)
    except Exception as error:                          # noqa: BLE001 -- a page, not a job
        return {'state': 'unknown', 'detail': f'the approval could not be read ({error})'}
    if current is None:
        return {'state': 'experimental',
                'detail': 'no version is approved for this case, so any result here is '
                          'experimental: useful to look at, not something to publish from',
                # The durable identity is opaque on purpose; the command takes the case
                # a maintainer can see, and `approve` resolves it to the same identity.
                'remedy': 'a maintainer approves one with: eforecast approve '
                          f'{name or case} <bundle folder>',
                'case_id': case}
    return {'state': 'approved' if current.get('qualified') else 'unknown',
            'qualified': bool(current.get('qualified')), 'bundle': current.get('bundle'),
            'revision': current.get('revision'), 'approved_at': current.get('updated_at'),
            'approved_by': current.get('by'), 'note': current.get('note'),
            'runtimes': current.get('runtimes'), 'components': current.get('components'),
            'root': current.get('root'),
            'detail': ('this qualified version is selected for routine forecasts' if current.get('qualified') else
                       'Legacy approval retained; capture and qualify its serving reference before routine use.')}


def forecast(store, static_data, case, now) -> dict[str, Any]:
    """The two answers A15-03 keeps apart, plus what the last cycle left on disk.

    The published forecast and the latest attempt are separate fields here for the same reason
    they are separate rows in the store: a failed attempt five minutes ago must not make a
    two-day-old forecast look fresh, and a good forecast must not hide that every attempt
    since has failed.
    """
    out: dict[str, Any] = {}
    if store is None:
        out['state'] = 'unknown'
        out['detail'] = 'the durable store is not reachable from here'
    else:
        from eforecast.config import forecast_publication as publication
        try:
            state = publication.status(store, case, now=now)
            out.update(state)
            valid = publication.last_valid(store, case)
            if valid is not None:
                recorded = (valid.get('summary') or {})
                out['covered'] = {'first': recorded.get('first'), 'last': recorded.get('last'),
                                  'rows': recorded.get('rows'),
                                  'columns': recorded.get('columns')}
                out['bundle'] = valid.get('bundle')
                from .model_approval import history
                payload = valid.get('payload_ref') or {}
                out['verified'] = False
                if valid.get('verified') and payload.get('path'):
                    from eforecast.common_utils.lineage import digest_file
                    info = _stat(Path(payload['path']))
                    if info.get('exists') and info.get('size_bytes', 0) <= 32 * 1024**2:
                        out['verified'] = digest_file(payload['path']) == payload.get('sha256')
                out['approved_history'] = any(h['bundle'] == valid.get('bundle') and h.get('qualified')
                                              for h in history(store, case))
        except Exception as error:                      # noqa: BLE001
            out['state'] = 'unknown'
            out['detail'] = f'the publication status could not be read ({error})'
    folder = Path(static_data.get('path_group') or '') / 'predictions'
    out['files'] = {'latest': _stat(folder / 'latest.csv'),
                    'status': _stat(folder / 'status.json')}
    if out['files']['status'].get('exists'):
        try:
            record = json.loads((folder / 'status.json').read_text(encoding='utf-8')[:200_000])
        except (OSError, ValueError) as error:
            out['last_cycle'] = {'error': f'{type(error).__name__}: {error}'}
        else:
            out['last_cycle'] = {
                'issue_time': record.get('issue_time'), 'state': record.get('state'),
                'rows': record.get('rows'), 'days': record.get('days'),
                'warnings': (record.get('warnings') or [])[:ROWS],
                'publication': record.get('publication'),
                'duration_s': record.get('duration_s')}
    return out


# ------------------------------------------------------------------- one sentence, one action


def verdict(model_block, forecast_block, headline_block, coverage_block, fresh) -> dict[str, Any]:
    """One sentence a colleague can act on, and the reason behind it.

    Deliberately conservative. "Usable" requires an approved version *and* a published forecast
    that is not stale; everything else is named for what is missing rather than failed.
    """
    actions: list[dict[str, str]] = []
    published = (forecast_block.get('forecast') or {}).get('state')
    attempt = (forecast_block.get('attempt') or {}).get('state')
    approved = model_block.get('state') == 'approved'

    if headline_block is None:
        actions.append({'what': 'run the combine and evaluation stage',
                        'why': 'there is no recorded evaluation for this case yet'})
    if not approved and model_block.get('state') != 'unknown':
        actions.append({'what': 'ask a maintainer to approve a version',
                        'why': 'nothing is approved, so these results are experimental'})
    if published == 'stale':
        actions.append({'what': 'see docs/troubleshooting.md',
                        'why': 'the published forecast is older than it should be'})
    if published == 'none':
        actions.append({'what': 'run a forecast',
                        'why': 'nothing has ever passed the publication checks for this case'})
    if attempt == 'rejected':
        actions.append({'what': 'read the reason on the forecast line',
                        'why': 'the last attempt did not publish; the previous forecast stands'})
    if fresh.get('state') == 'case_changed':
        actions.append({'what': 'rerun the evaluation',
                        'why': 'the case was edited after these numbers were written'})
    if coverage_block.get('recorded') and any(
            entry['missing'] for entry in coverage_block.get('slices') or []):
        actions.append({'what': 'check the coverage table',
                        'why': 'some predictors did not answer every requested timestamp'})

    if headline_block is None:
        state, sentence = 'none', 'No evaluation has been recorded for this case yet.'
    elif model_block.get('state') == 'unknown':
        # "Nobody approved this" and "we could not read the approval" are different facts, and
        # the second one must not be reported as the first.
        state = 'attention'
        sentence = ('There are recorded results, but whether a version is approved could not '
                    'be read here, so nothing on this page should be treated as routine.')
    elif approved and (forecast_block.get('bundle') != model_block.get('bundle')
                       or not forecast_block.get('verified') or not forecast_block.get('approved_history')
                       or not model_block.get('evaluation_matches') or fresh.get('state') != 'current'):
        state, sentence = 'attention', ('The publication, approved version and displayed evaluation '
                                        'are not all verified as the same current result.')
    elif approved and published == 'ok' and attempt == 'ok':
        state, sentence = 'usable', ('An approved version is published and the last forecast '
                                     'attempt worked.')
    elif approved and published in {'ok', 'stale'}:
        state = 'attention'
        sentence = ('An approved version is published, but ' +
                    ('the forecast is stale' if published == 'stale' else
                     'the last attempt did not publish') + '.')
    elif approved:
        state, sentence = 'attention', ('A version is approved and nothing is published for '
                                        'this case yet.')
    else:
        state = 'experimental'
        sentence = ('These results are experimental: nobody has approved a version for routine '
                    'use.')
    return {'state': state, 'sentence': sentence, 'actions': actions[:6]}


# ------------------------------------------------------------------- the page


def summary(document: dict[str, Any], path: str | None = None, *, store=None, runs=None,
            now: float | None = None, case_id: str | None = None,
            name: str | None = None) -> dict[str, Any]:
    """Everything the results page shows, read from recorded artifacts and durable state.

    Raises `ServiceError` only when the case itself cannot be resolved -- a missing artifact is
    content of the page, not a failure of it.
    """
    static_data = case_static(document, path)
    label = str(static_data['_id'])
    if store is not None and case_id is None:
        from .serving_scope import for_config
        case_id = for_config(store, path, static_data)['case_id']
    case = case_id or label
    results = Path(static_data['path_model']) / 'Results'
    sources: dict[str, Any] = {}

    evaluation_rows = _table(results / 'evaluation.csv', sources, 'evaluation')
    final_rows = _table(results / 'summary.csv', sources, 'summary')
    selection_rows = _table(results / 'selection.csv', sources, 'selection')
    coverage_rows = _table(results / 'coverage.csv', sources, 'coverage')
    interval_rows = _table(results / 'intervals_summary.csv', sources, 'intervals')
    unsupported = _table(results / 'unsupported_clusters.csv', sources, 'unsupported_clusters')
    unfinished = _table(results / 'unfinished_trials.csv', sources, 'unfinished_trials')
    sources['calibration'] = _stat(results / 'calibration.json')
    calibration = None
    if sources['calibration'].get('exists'):
        try:
            loaded = json.loads((results / 'calibration.json').read_text(encoding='utf-8'))
            calibration = {'rated': loaded.get('rated'), 'columns': loaded.get('columns'),
                           'coverages': loaded.get('coverages'),
                           'predictors': len(loaded.get('predictors') or {})}
        except (OSError, ValueError) as error:
            sources['calibration']['error'] = f'{type(error).__name__}: {error}'

    coverage_block = coverage(coverage_rows, sources)
    model_block = model(store, case, name)
    served_path = likely_served(final_rows)
    if model_block.get('root'):
        try:
            from eforecast import model_bundle
            serving = (model_bundle.read_fixture(model_block['root']) or {}).get('serving') or {}
            served_path = serving.get('path') or served_path
        except (OSError, ValueError):
            pass
    headline_block = headline(final_rows, served_path)
    forecast_block = forecast(store, static_data, case, now)
    model_block['evaluation_matches'] = False
    if model_block.get('root') and sources['summary'].get('exists'):
        try:
            from eforecast import model_bundle
            from eforecast.common_utils.lineage import digest_file
            manifest = model_bundle.read(model_block['root'])
            item = next((c for c in manifest['components'] if c['path'] == 'Results/summary.csv'), None)
            model_block['evaluation_matches'] = bool(item and item['identity'].get('sha256') ==
                                                       digest_file(str(results / 'summary.csv')))
        except (OSError, ValueError, KeyError):
            pass
    forecast_block['matches_active'] = bool(forecast_block.get('verified') and
        forecast_block.get('approved_history') and forecast_block.get('bundle') == model_block.get('bundle'))

    produced = produced_by(document, runs, static_data)
    if path:
        produced['document_modified'] = _stat(Path(path)).get('modified')
    fresh = freshness(produced, sources)
    return {
        'case': {'name': label, 'id': case_id, 'type': static_data.get('type'),
                 'horizon_type': static_data.get('horizon_type'),
                 'path_model': static_data.get('path_model'),
                 'model_exists': os.path.isdir(static_data.get('path_model') or '')},
        'verdict': verdict(model_block, forecast_block, headline_block, coverage_block, fresh),
        'model': model_block,
        'headline': headline_block,
        'scores': roles(final_rows, evaluation_rows, selection_rows, interval_rows, sources),
        'coverage': coverage_block,
        'period': period(static_data, evaluation_rows, coverage_block, sources),
        'units': units(static_data, calibration),
        'freshness': fresh,
        'produced_by': produced,
        'forecast': forecast_block,
        'incomplete': {
            'unsupported_clusters': [{'cluster': row.get('cluster'), 'method': row.get('method'),
                                      'reason': row.get('reason')} for row in unsupported[:ROWS]],
            'unfinished_trials': [{'cluster': row.get('cluster'), 'method': row.get('method'),
                                   'trial': row.get('trial'), 'reason': row.get('reason')}
                                  for row in unfinished[:ROWS]],
            'missing_artifacts': sorted(name for name, info in sources.items()
                                        if not info.get('exists'))},
        'caveats': caveats(static_data),
        'sources': sources,
    }
