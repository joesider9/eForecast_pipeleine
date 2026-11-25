# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A15-03: publish a forecast only when it is one, and never lose the last good one.

Today `eforecast_online/service.py` writes `latest.csv` from whatever the attempt produced. An
empty forecast appends a warning, writes an empty `latest.csv` over a good one, and reports
`state: ok`. That is the failure this module exists to remove: **a failed attempt must not be
able to destroy a valid forecast.**

The shape:

* every attempt is recorded as an **immutable issue**, identified by case, UTC issue time,
  pinned model bundle, input version and attempt -- so two attempts are never confused and a
  repeat is recognisable rather than merely tolerated;
* the **last-valid pointer** moves only for an eligible issue, atomically, with a revision
  check. An empty, partial, non-finite, failed, cancelled, stale-fenced, superseded or
  backfilled attempt leaves it exactly where it was;
* **latest-attempt status and last-valid freshness are different questions** and are answered
  separately. A colleague asking "is my forecast current?" must not be reassured by an attempt
  that ran five minutes ago and produced nothing.

`latest.csv` stays, because consumers read it -- but it is written **from the last valid
issue**, never from the attempt, and it is a convenience export rather than the authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

VERSION = 1

VALID, REJECTED = 'valid', 'rejected'

#: Why an attempt did not become the published forecast. Each is a different conversation with
#: whoever has to act, so they are not collapsed into "failed".
EMPTY = 'empty'                  # it produced no rows at all
PARTIAL = 'partial'              # fewer rows than the horizon asks for
NONFINITE = 'nonfinite'          # a NaN or an infinity reached the output
UNITS = 'units'                  # values implausible for the site's rated capacity
MISSING_QUANTILES = 'quantiles'  # a band the case expects was not produced
FAILED = 'failed'                # the attempt raised
CANCELLED = 'cancelled'
STALE_FENCE = 'stale_fence'      # an older fence reporting after a newer one
SUPERSEDED = 'superseded'        # an older issue time arriving after a newer one
BACKFILL = 'backfill'            # deliberately filling the past; never the latest
UNDECLARED_CORRECTION = 'undeclared_correction'

SCHEMA = (
    # Immutable: an issue row is written once and never updated. A correction is a new row.
    '''CREATE TABLE IF NOT EXISTS eforecast_forecast_issues (
       namespace TEXT NOT NULL, case_id TEXT NOT NULL, issue_id TEXT NOT NULL,
       issued_at DOUBLE PRECISION NOT NULL, state TEXT NOT NULL, bundle TEXT NOT NULL,
       input_version TEXT NOT NULL, attempt TEXT NOT NULL, payload TEXT NOT NULL,
       recorded_at DOUBLE PRECISION NOT NULL,
       PRIMARY KEY (namespace, case_id, issue_id))''',
    '''CREATE INDEX IF NOT EXISTS eforecast_forecast_issues_time
       ON eforecast_forecast_issues (namespace, case_id, issued_at, recorded_at)''',
    # The pointer. One row per case, moved by compare-and-set and by nothing else.
    '''CREATE TABLE IF NOT EXISTS eforecast_forecast_valid (
       namespace TEXT NOT NULL, case_id TEXT NOT NULL, revision BIGINT NOT NULL,
       issue_id TEXT NOT NULL, issued_at DOUBLE PRECISION NOT NULL, payload TEXT NOT NULL,
       updated_at DOUBLE PRECISION NOT NULL,
       PRIMARY KEY (namespace, case_id))''',
)


class PublicationError(ValueError):
    """The forecast cannot be published, and the last valid one is untouched."""


def _in(store, cursor):
    from contextlib import nullcontext
    return nullcontext(cursor) if cursor is not None else store._transaction()


def lock_case(store, cursor, case):
    """Serialize pointer and export decisions, including the first row, on both SQL stores."""
    if store.placeholder == '%s':
        key = int.from_bytes(hashlib.sha256((store.namespace + ':' + case).encode()).digest()[:8],
                             'big', signed=True)
        store._execute(cursor, 'SELECT pg_advisory_xact_lock(?)', (key,))


# ---------------------------------------------------------------------- identity


def issue_identity(case, issued_at, bundle, input_version, attempt):
    """What makes this issue *this* issue.

    All five, because each one changes what the numbers mean: the same issue time from a
    different model, or from a different input cycle, is a different forecast and must not
    quietly replace the first.
    """
    for name, value in (('case', case), ('bundle', bundle),
                        ('input_version', input_version), ('attempt', attempt)):
        if not isinstance(value, str) or not value.strip():
            raise PublicationError(f'{name} must be named for an issue to be identified')
    if not isinstance(issued_at, (int, float)) or not math.isfinite(issued_at) or issued_at <= 0:
        raise PublicationError('issued_at must be a positive UTC timestamp')
    seed = json.dumps([case, round(float(issued_at), 6), bundle, input_version, attempt],
                      sort_keys=True)
    return 'i-' + hashlib.sha256(seed.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------- the validity gate


def check(summary, *, expected_rows=None, rated=None, quantiles=(), geometry=None):
    """Everything checked before a forecast may be published. Returns a list of reasons.

    An empty list means publishable. The reasons are deliberately specific: "partial" and
    "empty" send a reader to different places, and reporting a partial horizon as complete is
    the thing this exists to stop.
    """
    reasons = []
    rows = summary.get('rows')
    if not isinstance(rows, int) or rows < 0:
        return [{'reason': FAILED, 'detail': 'the attempt reported no row count'}]
    if rows == 0:
        reasons.append({'reason': EMPTY, 'detail': 'the forecast produced no rows'})
    elif expected_rows is not None and rows < expected_rows:
        reasons.append({'reason': PARTIAL,
                        'detail': f'{rows} of {expected_rows} rows; a partial horizon is not a '
                                  f'complete forecast'})
    values = summary.get('values') or []
    if rows and not values:
        reasons.append({'reason': NONFINITE, 'detail': 'the forecast contains no numeric values'})
    if any(isinstance(v, bool) or v is None or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        reasons.append({'reason': NONFINITE,
                        'detail': 'a non-finite value reached the output'})
    elif rated and values:
        out = [v for v in values if not (-0.01 * rated <= v <= 1.5 * rated)]
        if out:
            reasons.append({'reason': UNITS,
                            'detail': f'{len(out)} value(s) implausible for a {rated} rated '
                                      f'site (for example {out[0]:.3f})'})
    produced = set(summary.get('quantiles') or ())
    missing = [q for q in quantiles if q not in produced]
    if missing:
        reasons.append({'reason': MISSING_QUANTILES,
                        'detail': f'the case expects bands {missing} and they were not produced'})
    if geometry is not None:
        expected_index, columns = geometry['index'], geometry['columns']
        index = summary.get('index') or []
        actual_columns = summary.get('columns') or []
        if index != expected_index or len(index) != len(set(index)):
            reasons.append({'reason': PARTIAL, 'detail': 'timestamps differ from the requested ordered coverage'})
        if actual_columns != columns or len(actual_columns) != len(set(actual_columns)):
            reasons.append({'reason': PARTIAL, 'detail': 'target or interval columns differ from the requested horizon'})
        if len(values) != rows * len(columns):
            reasons.append({'reason': NONFINITE, 'detail': 'every forecast cell must be numeric and finite'})
        if summary.get('interval_order_valid') is not True:
            reasons.append({'reason': MISSING_QUANTILES, 'detail': 'interval bounds are missing or crossed'})
    return reasons


def compact(summary):
    """What is worth keeping about an attempt: its shape and its extremes, not the forecast.

    The values are needed to judge the attempt and for nothing afterwards. Storing them would
    put a copy of every forecast in the run store -- a size problem (a quarter-hourly case with
    fifty columns is ~80 kB an attempt, several times a day, forever) and a second place the
    numbers live, which is worse. The files hold the forecast; this holds the evidence.
    """
    values = summary.get('values') or []
    kept = {key: value for key, value in summary.items() if key != 'values'}
    finite = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)
              and math.isfinite(v)]
    kept['values_checked'] = len(values)
    if finite:
        kept['min'], kept['max'] = min(finite), max(finite)
    if len(finite) != len(values):
        kept['nonfinite'] = len(values) - len(finite)
    return kept


# ---------------------------------------------------------------------- reading


def last_valid(store, case, *, cursor=None):
    """The forecast colleagues should be using, or None when there has never been one."""
    with _in(store, cursor) as handle:
        from .serving_scope import read_keys
        row = None
        for key in read_keys(store, case, handle):
            store._execute(handle, '''SELECT revision, issue_id, issued_at, payload, updated_at
                FROM eforecast_forecast_valid WHERE namespace=? AND case_id=?''',
                (store.namespace, key))
            row = handle.fetchone()
            if row is not None:
                break
    if row is None:
        return None
    revision, issue_id, issued_at, payload, updated = row
    return {'case': case, 'revision': int(revision), 'issue_id': issue_id,
            'issued_at': float(issued_at), 'updated_at': float(updated), **json.loads(payload), 'record_key': key}


def latest_attempt(store, case, *, cursor=None):
    """The most recent attempt, valid or not. **Not** the same question as `last_valid`."""
    with _in(store, cursor) as handle:
        from .serving_scope import read_keys
        rows = []
        for key in read_keys(store, case, handle):
            store._execute(handle, """SELECT issue_id, issued_at, state, payload, recorded_at
                FROM eforecast_forecast_issues WHERE namespace=? AND case_id=?
                ORDER BY recorded_at DESC LIMIT 1""", (store.namespace, key))
            row = handle.fetchone()
            if row:
                rows.append(row)
        row = max(rows, key=lambda r:r[4]) if rows else None
    if row is None:
        return None
    issue_id, issued_at, state, payload, recorded = row
    return {'case': case, 'issue_id': issue_id, 'issued_at': float(issued_at), 'state': state,
            'recorded_at': float(recorded), **json.loads(payload)}


def issue(store, case, issue_id, *, cursor=None):
    with _in(store, cursor) as handle:
        from .serving_scope import read_keys
        row = None
        for key in read_keys(store, case, handle):
            store._execute(handle, """SELECT issued_at,state,bundle,input_version,attempt,payload,recorded_at
                FROM eforecast_forecast_issues WHERE namespace=? AND case_id=? AND issue_id=?""",
                (store.namespace,key,issue_id))
            row = handle.fetchone()
            if row:
                break
    if row is None:
        return None
    issued_at, state, bundle, input_version, attempt, payload, recorded = row
    return {'case': case, 'issue_id': issue_id, 'issued_at': float(issued_at), 'state': state,
            'bundle': bundle, 'input_version': input_version, 'attempt': attempt,
            'recorded_at': float(recorded), **json.loads(payload)}


# ---------------------------------------------------------------------- publishing


def publish(store, case, *, issued_at, bundle, input_version, attempt, summary,
            expected_rows=None, rated=None, quantiles=(), backfill=False,
            correction_of=None, outcome='complete', cursor=None, now=None,
            geometry=None, acceptance=None, hold=None, payload_ref=None):
    """Record the attempt and move the pointer only if it deserves to move.

    Returns ``{'issue_id', 'state', 'published', 'reasons', 'last_valid'}``. `published` is
    whether the pointer moved; an attempt can be perfectly valid and still not move it, because
    something newer is already there.

    Every path through this function records the attempt. A forecast nobody can see having
    failed is how a stale number survives for a week.
    """
    moment = time.time() if now is None else now
    identity = issue_identity(case, issued_at, bundle, input_version, attempt)

    reasons = []
    if outcome == 'cancelled':
        reasons.append({'reason': CANCELLED, 'detail': 'the attempt was cancelled'})
    elif outcome != 'complete':
        reasons.append({'reason': FAILED, 'detail': f'the attempt ended {outcome}'})
    else:
        reasons = check(summary, expected_rows=expected_rows, rated=rated, quantiles=quantiles, geometry=geometry)
    if outcome == 'complete' and (acceptance is None or hold is None or payload_ref is None):
        reasons.append({'reason': STALE_FENCE, 'detail': 'verified acceptance, attempt hold and retained payload are required'})
    if backfill:
        reasons.append({'reason': BACKFILL,
                        'detail': 'a backfill fills the past; it never becomes the latest '
                                  'forecast'})

    record = {'version': VERSION, 'summary': compact(summary), 'outcome': outcome,
              'backfill': bool(backfill), 'correction_of': correction_of,
              'bundle': bundle, 'input_version': input_version, 'attempt': attempt,
              'verified': False, 'payload_ref': payload_ref,
              'approval_revision': (hold or {}).get('approval_revision')}

    with _in(store, cursor) as handle:
        lock_case(store, handle, case)
        existing = issue(store, case, identity, cursor=handle)
        if existing is not None:
            # A settled delivery remains idempotent after its hold has been released.
            return {'issue_id': identity, 'state': existing['state'], 'published': False,
                    'reasons': [{'reason': 'duplicate',
                                 'detail': 'this exact issue was already recorded; a repeat '
                                           'does not refresh anything'}],
                    'last_valid': last_valid(store, case, cursor=handle)}
        if not reasons:
            if not acceptance():
                raise PublicationError('completion/fence acceptance is no longer valid')
            store._execute(handle, """SELECT bundle,payload FROM eforecast_model_holds
                WHERE namespace=? AND case_id=? AND attempt_id=? AND token=? AND released_at IS NULL""",
                (store.namespace,case,hold['attempt_id'],hold['hold_token']))
            row = handle.fetchone()
            if row is None or row[0] != bundle or not json.loads(row[1]).get('qualified'):
                raise PublicationError('the attempt no longer holds this qualified bundle')
            from eforecast.common_utils.lineage import digest_file
            if digest_file(payload_ref['path']) != payload_ref['sha256']:
                raise PublicationError('retained forecast payload changed before acceptance')
            record['verified'] = True
        current = last_valid(store, case, cursor=handle)
        if not reasons and current is not None:
            if issued_at < current['issued_at']:
                reasons.append({'reason': SUPERSEDED,
                                'detail': f'this issue is older than the published one; the '
                                          f'newer forecast stays'})
            elif issued_at == current['issued_at']:
                # Same issue time: only a declared correction may replace it, and only when it
                # says what it is correcting. Silently replacing results from a different
                # input or model version is the thing this forbids.
                if correction_of != current['issue_id']:
                    reasons.append({
                        'reason': UNDECLARED_CORRECTION,
                        'detail': f'an issue already exists for this time from bundle '
                                  f'{str(current.get("bundle"))[:12]} / inputs '
                                  f'{current.get("input_version")}; a replacement must declare '
                                  f'correction_of={current["issue_id"]}'})

        state = VALID if not reasons else REJECTED
        store._execute(handle, '''INSERT INTO eforecast_forecast_issues
            (namespace,case_id,issue_id,issued_at,state,bundle,input_version,attempt,payload,
             recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (store.namespace, case, identity, float(issued_at), state, bundle, input_version,
             attempt, json.dumps(dict(record, reasons=reasons), sort_keys=True), moment))

        published = False
        if state == VALID:
            payload = json.dumps(record, sort_keys=True)
            if current is None or current.get('record_key', case) != case:
                store._execute(handle, '''INSERT INTO eforecast_forecast_valid
                    (namespace,case_id,revision,issue_id,issued_at,payload,updated_at)
                    VALUES (?,?,?,?,?,?,?)''',
                    (store.namespace, case, (current or {}).get('revision', 0) + 1, identity, float(issued_at), payload, moment))
                published = True
            else:
                store._execute(handle, '''UPDATE eforecast_forecast_valid
                    SET revision=?, issue_id=?, issued_at=?, payload=?, updated_at=?
                    WHERE namespace=? AND case_id=? AND revision=?''',
                    (current['revision'] + 1, identity, float(issued_at), payload, moment,
                     store.namespace, case, current['revision']))
                published = handle.rowcount == 1
                if not published:
                    raise PublicationError('publication pointer changed outside its case lock; retry')
        if published:
            # A16-06's cumulative counter, incremented in the transaction that accepted this
            # issue: it commits with the publication or not at all, a redelivery of the same
            # issue adds nothing, and deleting history cannot change it (R27-15).
            from .verified_completions import record as count_verified
            count_verified(store, SERVING_RUNTIME, identity, cursor=handle, now=moment)
        final = last_valid(store, case, cursor=handle)
    return {'issue_id': identity, 'state': state, 'published': published,
            'reasons': reasons, 'last_valid': final}


# ---------------------------------------------------------------------- status, two questions


#: The runtime a published forecast was served by. One venue produces accepted forecasts
#: today; a stage completion counted by the same counter would carry its worker's runtime.
SERVING_RUNTIME = 'online'

STALE_AFTER_SECONDS = 6 * 3600


def status(store, case, *, now=None, stale_after=STALE_AFTER_SECONDS):
    """What a colleague needs: is there a good forecast, how old is it, and did the last try work.

    Deliberately two answers. An attempt that ran five minutes ago and produced nothing must not
    make a two-day-old forecast look current, and a valid forecast must not hide that every
    attempt since has failed.
    """
    moment = time.time() if now is None else now
    valid = last_valid(store, case)
    attempt = latest_attempt(store, case)
    out = {'case': case}
    if valid is None:
        out['forecast'] = {'state': 'none', 'detail': 'no valid forecast has ever been published',
                           'remedy': 'run a forecast; until one passes the checks there is '
                                     'nothing for colleagues to use'}
    else:
        age = moment - valid['issued_at']
        out['forecast'] = {
            'state': 'stale' if age > stale_after else 'ok',
            'issue_id': valid['issue_id'], 'issued_at': valid['issued_at'],
            'age_seconds': round(age, 1),
            'detail': f'issued {_when(valid["issued_at"])} UTC ({age / 3600:.1f} h ago)'
                      + (' -- STALE' if age > stale_after else ''),
            'remedy': 'this forecast is old; check the latest attempt below'
                      if age > stale_after else 'Nothing to do.'}
    if attempt is None:
        out['attempt'] = {'state': 'none', 'detail': 'no attempt has been recorded'}
    elif attempt['state'] == VALID:
        out['attempt'] = {'state': 'ok',
                          'detail': f'the last attempt published {attempt["issue_id"]}'}
    else:
        reasons = ', '.join(sorted({item['reason'] for item in attempt.get('reasons') or []}))
        out['attempt'] = {
            'state': 'rejected', 'issue_id': attempt['issue_id'],
            'detail': f'the last attempt was not published ({reasons or "unknown"})',
            'remedy': 'the published forecast above is unchanged; fix the attempt and retry'}
    return out


def _when(stamp):
    return time.strftime('%Y-%m-%d %H:%M', time.gmtime(stamp))


# ---------------------------------------------------------------------- the legacy export


def export_latest(store, case, folder, *, writer=None):
    """Idempotent projection of the durable pointer, serialized with every pointer change.

    The retained payload makes this repairable after a process or filesystem failure. Unique
    temporary names prevent one exporter from consuming another exporter's incomplete file.
    """
    from pathlib import Path
    import os
    import shutil
    import uuid
    from eforecast.common_utils.lineage import digest_file
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'latest.csv'
    temporary = folder / ('.latest-' + uuid.uuid4().hex + '.partial')
    try:
        with store._transaction() as handle:
            lock_case(store, handle, case)
            valid = last_valid(store, case, cursor=handle)
            if valid is None:
                return {'written': False, 'path': str(target), 'why': 'no valid forecast'}
            if writer is not None:
                writer(temporary, valid)
            else:
                payload = valid.get('payload_ref')
                if not payload:
                    return {'written': False, 'path': str(target), 'why': 'legacy issue has no retained payload'}
                shutil.copyfile(payload['path'], temporary)
                if digest_file(str(temporary)) != payload['sha256']:
                    raise PublicationError('retained forecast payload failed verification')
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            fd = os.open(folder, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            return {'written': True, 'path': str(target), 'issue_id': valid['issue_id']}
    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------- the approved model


def store(workspace=None):
    """The durable state a forecasting machine needs: approvals and published forecasts.

    The host passes its own run store to the functions here and never calls this. A machine
    that only forecasts has no run model and no training packages, so it opens the same
    database with only these tables -- every statement is ``IF NOT EXISTS``, and the host
    creates the rest when it opens it.
    """
    import os
    from eforecast.config import model_approval
    from eforecast.config.sql_store import configured
    from eforecast.config.verified_completions import SCHEMA as counted
    if workspace is None:
        workspace = os.environ.get('EFORECAST_WORKSPACE')
    # ``publish`` counts each accepted issue in the tables of ``verified_completions``. They were
    # there only when something else had opened the run store on this file first, which every
    # run once did; a forecast published on a store nothing else had opened failed on them.
    return configured(workspace, schema=tuple(model_approval.SCHEMA) + tuple(SCHEMA) + tuple(counted))


def begin(store, case, forecast_id, *, attempt_id=None, cursor=None, now=None):
    """Claim the approved bundle for a forecast about to run, and return what it must use.

    This is the other half of A15-02's pin. A forecast takes the approved version **once**, at
    the start, and keeps it: an approval landing while an ensemble is half done must change the
    next forecast, not this one. It also stops cleanup removing the bundle underneath a running
    forecast.

    Returns the bundle identity to pass to `publish`, so a forecast cannot accidentally publish
    under a different model than it ran with.
    """
    from eforecast.config import model_approval
    pinned = model_approval.pin(store, case, forecast_id, attempt_id=attempt_id, cursor=cursor, now=now)
    return {**pinned, 'approval_revision': pinned['revision']}


def finish(store, case, forecast_id, *, hold_token=None, attempt_id=None, cursor=None, now=None):
    """The forecast is over; its hold on the bundle ends. Cleanup may consider it again."""
    from eforecast.config import model_approval
    return model_approval.release(store, case, forecast_id, hold_token=hold_token,
                                  attempt_id=attempt_id, cursor=cursor, now=now)


def render(state):
    """The two answers, in the order a colleague needs them."""
    lines = []
    forecast = state['forecast']
    if forecast['state'] == 'none':
        lines.append('forecast  none published yet')
    else:
        lines.append(f'forecast  {forecast["detail"]}')
    lines.append(f'attempt   {state["attempt"]["detail"]}')
    for part in (forecast, state['attempt']):
        if part.get('remedy') and part['remedy'] != 'Nothing to do.':
            lines.append(f'          {part["remedy"]}')
    return '\n'.join(lines)


def main(argv=None):
    """`forecast-status <case>` -- what is published, and whether the last try worked."""
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(prog='eforecast forecast-status',
                                     description=main.__doc__)
    parser.add_argument('case')
    parser.add_argument('--repair-export', metavar='OUTPUT_FOLDER',
                        help='explicitly restore latest.csv from the verified durable pointer; does not retrain')
    arguments = parser.parse_args(argv)
    durable = store(os.environ.get('EFORECAST_WORKSPACE'))
    from .serving_scope import for_cli
    from .service import ServiceError
    try:
        case = for_cli(durable, arguments.case)['case_id']
    except ServiceError as error:
        print(str(error), file=sys.stderr)
        return 2
    if arguments.repair_export:
        repaired = export_latest(durable, case, arguments.repair_export)
        if not repaired['written']:
            print(f'export    not repaired: {repaired["why"]}')
            return 2
        print(f'export    restored {repaired["issue_id"]} to {repaired["path"]}')
    state = status(durable, case)
    print(render(state))
    if state['forecast']['state'] == 'none':
        return 2
    return 1 if state['forecast']['state'] == 'stale' or \
        state['attempt']['state'] == 'rejected' else 0


if __name__ == '__main__':
    raise SystemExit(main())
