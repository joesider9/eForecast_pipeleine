# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A15-02: which bundle a case uses for routine forecasts, and how to go back.

Approval is **not** "training finished". A trial that completed is a candidate; a bundle that
validated is a candidate; neither is what colleagues' forecasts run on until a maintainer says
so. Keeping those apart is the whole point of this module -- it is why a pipeline can finish a
hundred trials overnight without changing anyone's forecast the next morning.

Three properties, each of which is a way the obvious implementation goes wrong:

* **The pointer moves atomically, or not at all.** Two maintainers approving at once must not
  leave a case pointing at half of each bundle. The move is one `UPDATE ... WHERE revision=?`
  in the run store's own transaction, so the loser is told to reload rather than silently
  overwriting.
* **The previous version is retained, and going back is a pointer operation.** Not retraining,
  not copying weights over the active model. Rollback re-validates the target first, because a
  pointer to a bundle whose components have since been deleted is worse than no rollback.
* **An accepted forecast pins the bundle it used.** Approval during an in-flight ensemble must
  not switch half of it, so a forecast records the bundle identity it started with and a
  pinned bundle cannot be deleted.

Case permissions are not re-implemented here. This module takes a case name and a `by`; the
API reaches it through `writable_case`, which is the only way a route reaches a case at all.
"""
from __future__ import annotations

import json
import time
import uuid

from .serving_scope import SCHEMA as _SCOPE_SCHEMA

SCHEMA = _SCOPE_SCHEMA + (
    '''CREATE TABLE IF NOT EXISTS eforecast_model_approvals (
       namespace TEXT NOT NULL, case_id TEXT NOT NULL, revision BIGINT NOT NULL,
       bundle TEXT NOT NULL, payload TEXT NOT NULL, updated_at DOUBLE PRECISION NOT NULL,
       PRIMARY KEY (namespace, case_id))''',
    # Every approval that was ever active, so "the previous working version" is a fact rather
    # than a hope. Rows are never deleted by approving: that is what retention means here.
    '''CREATE TABLE IF NOT EXISTS eforecast_model_approval_history (
       namespace TEXT NOT NULL, case_id TEXT NOT NULL, revision BIGINT NOT NULL,
       bundle TEXT NOT NULL, payload TEXT NOT NULL, recorded_at DOUBLE PRECISION NOT NULL,
       PRIMARY KEY (namespace, case_id, revision))''',
    # What an accepted forecast pinned. An approval that lands mid-ensemble changes what the
    # *next* forecast uses, never what a running one is already using.
    '''CREATE TABLE IF NOT EXISTS eforecast_model_pins (
       namespace TEXT NOT NULL, case_id TEXT NOT NULL, forecast_id TEXT NOT NULL,
       bundle TEXT NOT NULL, revision BIGINT NOT NULL, payload TEXT NOT NULL,
       pinned_at DOUBLE PRECISION NOT NULL, released_at DOUBLE PRECISION,
       PRIMARY KEY (namespace, case_id, forecast_id))''',
    '''CREATE INDEX IF NOT EXISTS eforecast_model_pins_bundle
       ON eforecast_model_pins (namespace, bundle, released_at)''',
)

SCHEMA += ('''CREATE TABLE IF NOT EXISTS eforecast_model_holds (
    namespace TEXT NOT NULL, case_id TEXT NOT NULL, attempt_id TEXT NOT NULL,
    token TEXT NOT NULL, bundle TEXT NOT NULL, revision BIGINT NOT NULL, payload TEXT NOT NULL,
    pinned_at DOUBLE PRECISION NOT NULL, released_at DOUBLE PRECISION,
    PRIMARY KEY(namespace,case_id,attempt_id))''',)

VERSION = 2


class ApprovalError(ValueError):
    """The approval cannot be made, and the active version is untouched."""


class ApprovalConflict(ApprovalError):
    """Someone else moved the pointer first. Reload and decide again."""


def _in(store, cursor):
    """Join the caller's transaction, or open one. Approving is a single atomic act."""
    from contextlib import nullcontext
    return nullcontext(cursor) if cursor is not None else store._transaction()


def _identity(manifest):
    from eforecast.bundle_serving import identity
    return identity(manifest)


def candidate(root, *, images=None, fixture=None):
    """Read a bundle and check it is fit to be approved. Returns ``(manifest, identity)``.

    Validation is not optional and not cached: a bundle that validated last week may name a
    runtime this deployment can no longer provide, and approving it would make that nobody's
    discovery until a forecast failed.
    """
    from eforecast import model_bundle, model_bundle_validate
    manifest = model_bundle.read(root)
    report = model_bundle_validate.validate(root, images=images, fixture=fixture)
    if report['verdict'] != model_bundle_validate.OK:
        failures = [item['detail'] for item in report['findings'] if item['state'] != 'ok']
        raise ApprovalError(
            'this bundle does not validate here, so it cannot be approved: '
            + '; '.join(failures[:3]))
    return manifest, _identity(manifest)


def active(store, case, *, cursor=None):
    from .serving_scope import read_keys
    with _in(store, cursor) as handle:
        for key in read_keys(store, case, handle):
            store._execute(handle, """SELECT revision,bundle,payload,updated_at
                FROM eforecast_model_approvals WHERE namespace=? AND case_id=?""",
                (store.namespace, key))
            row = handle.fetchone()
            if row:
                revision, bundle, payload, updated = row
                return {**json.loads(payload), 'case': case, 'revision': int(revision),
                        'bundle': bundle, 'updated_at': float(updated), 'record_key': key}
    return None


def history(store, case, *, limit=32, cursor=None):
    from .serving_scope import read_keys
    if type(limit) is not int or not 1 <= limit <= 512:
        raise ApprovalError(f'bounded history expected 1..512, got {limit!r}')
    with _in(store, cursor) as handle:
        rows = []
        for key in read_keys(store, case, handle):
            store._execute(handle, """SELECT revision,bundle,payload,recorded_at
                FROM eforecast_model_approval_history WHERE namespace=? AND case_id=?
                ORDER BY revision DESC LIMIT ?""", (store.namespace, key, limit))
            rows.extend(handle.fetchall())
    return [{**json.loads(p), 'case': case, 'revision': int(r), 'bundle': b,
             'recorded_at': float(t)} for r,b,p,t in sorted(rows, key=lambda r:r[0], reverse=True)[:limit]]


def approve(store, case, root, *, by, images=None, fixture=None, expected_revision=None,
            note='', cursor=None, now=None):
    """Make a validated bundle the one this case's routine forecasts use.

    ``expected_revision`` is the revision the caller believes is active. Passing it turns two
    simultaneous approvals into one winner and one `ApprovalConflict`, instead of two writes
    where the second silently wins.
    """
    from eforecast.bundle_serving import snapshot
    from eforecast.model_bundle import BundleError
    try:
        root, _, _ = snapshot(root, fixture)
    except BundleError as error:
        raise ApprovalError(f'this bundle does not validate here: {error}') from error
    manifest, identity = candidate(root, images=images, fixture=fixture)
    moment = time.time() if now is None else now
    record = {'version': VERSION, 'identity_version': 2, 'qualified': True, 'identity': identity, 'root': str(root), 'by': by,
              'note': note, 'runtimes': sorted({c['runtime'] for c in manifest['components']}),
              'components': len(manifest['components']),
              'case_declared': manifest.get('case')}
    payload = json.dumps(record, sort_keys=True)

    with _in(store, cursor) as handle:
        from .forecast_publication import lock_case
        lock_case(store, handle, case)
        current = active(store, case, cursor=handle)
        if expected_revision is not None and (current or {}).get('revision', 0) != expected_revision:
            raise ApprovalConflict(
                f'this case is at revision {(current or {}).get("revision", 0)}, not '
                f'{expected_revision}; someone approved in between. Reload and decide again.')
        revision = ((current or {}).get('revision') or 0) + 1
        if current is not None and current.get('record_key', case) == case:
            # Retain what was active before this moment, so going back is a fact.
            store._execute(handle, '''INSERT INTO eforecast_model_approval_history
                (namespace,case_id,revision,bundle,payload,recorded_at) VALUES (?,?,?,?,?,?)
                ON CONFLICT (namespace,case_id,revision) DO NOTHING''',
                (store.namespace, case, current['revision'], current['bundle'],
                 json.dumps({k: v for k, v in current.items()
                             if k not in {'case', 'revision', 'bundle', 'updated_at'}},
                            sort_keys=True), current['updated_at']))
            store._execute(handle, '''UPDATE eforecast_model_approvals
                SET revision=?, bundle=?, payload=?, updated_at=?
                WHERE namespace=? AND case_id=? AND revision=?''',
                (revision, identity, payload, moment, store.namespace, case,
                 current['revision']))
            if handle.rowcount != 1:
                raise ApprovalConflict('the active version changed while approving; nothing '
                                       'was moved. Reload and decide again.')
        else:
            store._execute(handle, '''INSERT INTO eforecast_model_approvals
                (namespace,case_id,revision,bundle,payload,updated_at) VALUES (?,?,?,?,?,?)''',
                (store.namespace, case, revision, identity, payload, moment))
        # The new version is recorded in history as it becomes active, so history is complete
        # whether or not anything replaces it later.
        store._execute(handle, '''INSERT INTO eforecast_model_approval_history
            (namespace,case_id,revision,bundle,payload,recorded_at) VALUES (?,?,?,?,?,?)
            ON CONFLICT (namespace,case_id,revision) DO NOTHING''',
            (store.namespace, case, revision, identity, payload, moment))
    return active(store, case)


def rollback(store, case, *, by, to=None, images=None, fixture=None, note='',
             cursor=None, now=None):
    """Return a case to a previously validated version. A pointer operation, nothing else.

    The target is re-validated first. A rollback to a bundle whose components have since been
    deleted would replace a working model with a broken pointer, which is worse than refusing.
    """
    past = history(store, case, cursor=cursor)
    if not past:
        raise ApprovalError(f'{case} has no approval history, so there is nothing to return to')
    current = active(store, case, cursor=cursor)
    if to is None:
        earlier = [item for item in past
                   if item['revision'] != (current or {}).get('revision')]
        if not earlier:
            raise ApprovalError(
                f'{case} has only ever had one approved version, so there is no previous one '
                f'to return to')
        target = earlier[0]
    else:
        target = next((item for item in past if item['revision'] == to), None)
        if target is None:
            raise ApprovalError(f'{case} has no approved revision {to}')
    root = target.get('root')
    if not root:
        raise ApprovalError(f'revision {target["revision"]} did not record where its bundle is')
    manifest, identity = candidate(root, images=images, fixture=fixture)
    if identity != target['bundle']:
        raise ApprovalError(
            f'the bundle at {root} is no longer the one revision {target["revision"]} approved; '
            f'rolling back would point at something nobody validated')
    return approve(store, case, root, by=by, images=images, fixture=fixture,
                   expected_revision=(current or {}).get('revision', 0),
                   note=note or f'rollback to revision {target["revision"]}',
                   cursor=cursor, now=now)


# ---------------------------------------------------------------------- pinning


def pin(store, case, forecast_id, *, attempt_id=None, cursor=None, now=None):
    """One attempt owns one conditional hold; retries never revive an old release token."""
    attempt_id = attempt_id or forecast_id
    moment = time.time() if now is None else now
    with _in(store, cursor) as handle:
        from .forecast_publication import lock_case
        lock_case(store, handle, case)
        store._execute(handle, """SELECT token,bundle,revision,payload,released_at
            FROM eforecast_model_holds WHERE namespace=? AND case_id=? AND attempt_id=?""",
            (store.namespace, case, attempt_id))
        old = handle.fetchone()
        if old is not None and old[4] is None:
            return {**json.loads(old[3]), 'case':case, 'attempt_id':attempt_id,
                    'hold_token':old[0], 'bundle':old[1], 'revision':int(old[2])}
        current = active(store, case, cursor=handle)
        if current is None:
            raise ApprovalError(f'{case} has no approved bundle; compute a labelled preview or approve a bundle')
        token = uuid.uuid4().hex
        payload = {'root': current.get('root'), 'forecast_id': forecast_id,
                   'identity_version': current.get('identity_version', 1),
                   'qualified': current.get('qualified', False)}
        if old:
            store._execute(handle, """UPDATE eforecast_model_holds
                SET token=?,bundle=?,revision=?,payload=?,pinned_at=?,released_at=NULL
                WHERE namespace=? AND case_id=? AND attempt_id=? AND token=? AND released_at IS NOT NULL""",
                (token,current['bundle'],current['revision'],json.dumps(payload),moment,
                 store.namespace,case,attempt_id,old[0]))
            if handle.rowcount != 1:
                raise ApprovalConflict('attempt hold changed while reacquiring it')
        else:
            store._execute(handle, """INSERT INTO eforecast_model_holds
                (namespace,case_id,attempt_id,token,bundle,revision,payload,pinned_at,released_at)
                VALUES (?,?,?,?,?,?,?,?,NULL)""", (store.namespace,case,attempt_id,token,
                current['bundle'],current['revision'],json.dumps(payload),moment))
    return {**payload,'case':case,'attempt_id':attempt_id,'hold_token':token,
            'bundle':current['bundle'],'revision':current['revision']}


def release(store, case, forecast_id, *, hold_token=None, attempt_id=None, cursor=None, now=None):
    """A late release cannot release a successor. Tokenless calls only release legacy rows."""
    moment = time.time() if now is None else now
    with _in(store, cursor) as handle:
        from .forecast_publication import lock_case
        lock_case(store, handle, case)
        if hold_token is None:
            store._execute(handle, """UPDATE eforecast_model_pins SET released_at=?
                WHERE namespace=? AND case_id=? AND forecast_id=? AND released_at IS NULL""",
                (moment,store.namespace,case,forecast_id))
        else:
            store._execute(handle, """UPDATE eforecast_model_holds SET released_at=?
                WHERE namespace=? AND case_id=? AND attempt_id=? AND token=? AND released_at IS NULL""",
                (moment,store.namespace,case,attempt_id or forecast_id,hold_token))
        return bool(handle.rowcount)


def in_use(store, bundle, *, cursor=None):
    """Forecasts still holding this bundle, and whether it is the active one anywhere.

    Cleanup asks this. A version that is active, or pinned by a forecast that has not finished,
    is not a candidate for deletion however old it looks.
    """
    with _in(store, cursor) as handle:
        store._execute(handle, '''SELECT case_id, forecast_id FROM eforecast_model_pins
            WHERE namespace=? AND bundle=? AND released_at IS NULL''',
            (store.namespace, bundle))
        holders = [{'case': c, 'forecast_id': f} for c, f in handle.fetchall()]
        store._execute(handle, '''SELECT case_id,attempt_id FROM eforecast_model_holds
            WHERE namespace=? AND bundle=? AND released_at IS NULL''', (store.namespace,bundle))
        holders += [{'case': c, 'forecast_id': a} for c,a in handle.fetchall()]
        store._execute(handle, '''SELECT case_id FROM eforecast_model_approvals
            WHERE namespace=? AND bundle=?''', (store.namespace, bundle))
        cases = [row[0] for row in handle.fetchall()]
    return {'bundle': bundle, 'active_for': cases, 'pinned_by': holders}


def deletable(store, bundle, *, cursor=None):
    """``(True, '')`` when cleanup may remove this bundle, else ``(False, why not)``."""
    usage = in_use(store, bundle, cursor=cursor)
    if usage['active_for']:
        return False, (f'it is the approved version for '
                       f'{", ".join(sorted(usage["active_for"]))}')
    if usage['pinned_by']:
        holders = ', '.join(sorted(item['forecast_id'] for item in usage['pinned_by']))
        return False, f'{len(usage["pinned_by"])} unfinished forecast(s) are using it: {holders}'
    return True, ''


# ---------------------------------------------------------------------- the maintainer action


def render(entry, *, past=()):
    if entry is None:
        return ('no approved version: routine forecasts for this case have no model.\n'
                'Approve a validated bundle with: bin/eforecast approve <case> <bundle>')
    lines = [f'active   revision {entry["revision"]}  {entry["bundle"][:16]}',
             f'         {entry.get("root", "")}',
             f'         approved by {entry.get("by", "?")}'
             + (f' -- {entry["note"]}' if entry.get('note') else ''),
             f'         runtimes: {", ".join(entry.get("runtimes") or [])}']
    if not entry.get('qualified'):
        lines.append('         legacy record retained; capture and qualify a serving reference before routine publication')
    for item in past:
        if item['revision'] == entry['revision']:
            continue
        lines.append(f'previous revision {item["revision"]}  {item["bundle"][:16]}  '
                     f'{item.get("root", "")}')
    return '\n'.join(lines)


def main(argv=None):
    """`active`, `approve` and `rollback` for one case.

        bin/eforecast approve  <case> <bundle-directory> [--note ...]
        bin/eforecast rollback <case> [--to REVISION]
        bin/eforecast approved <case>

    Approval is a maintainer action and is deliberately separate from training: finishing a
    trial changes nothing about what colleagues' forecasts use.
    """
    import argparse
    import getpass
    import os
    import sys

    parser = argparse.ArgumentParser(prog='eforecast approve',
                                     description=main.__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    show = commands.add_parser('active', help='what this case uses for routine forecasts')
    show.add_argument('case')
    make = commands.add_parser('approve', help='make a validated bundle the active version')
    make.add_argument('case')
    make.add_argument('bundle')
    make.add_argument('--note', default='')
    make.add_argument('--expect-revision', type=int, default=None,
                      help='the revision you believe is active; refuses if it moved')
    back = commands.add_parser('rollback', help='return to a previously validated version')
    back.add_argument('case')
    back.add_argument('--to', type=int, default=None)
    back.add_argument('--note', default='')
    bind = commands.add_parser('bind-legacy', help='bind an existing project-name record to exactly one case')
    bind.add_argument('case')
    bind.add_argument('legacy_key')
    arguments = parser.parse_args(argv)

    # The approvals of a forecast-only machine, without the host's run model behind them.
    from eforecast.config.forecast_publication import store as durable_state
    store = durable_state(os.environ.get('EFORECAST_WORKSPACE'))
    who = os.environ.get('EFORECAST_USER') or getpass.getuser()
    from .serving_scope import for_cli, bind_legacy
    from .service import ServiceError
    try:
        case_id = for_cli(store, arguments.case)['case_id']
    except ServiceError as error:
        print(str(error), file=sys.stderr)
        return 2
    if arguments.command == 'bind-legacy':
        bind_legacy(store, arguments.legacy_key, case_id)
        print('Existing records are now bound to this case; files were not rewritten.')
        return 0
    try:
        if arguments.command == 'active':
            entry = active(store, case_id)
            print(render(entry, past=history(store, case_id) if entry else ()))
            return 0 if entry else 1
        if arguments.command == 'approve':
            entry = approve(store, case_id, arguments.bundle, by=who,
                            note=arguments.note, expected_revision=arguments.expect_revision)
        else:
            entry = rollback(store, case_id, by=who, to=arguments.to,
                             note=arguments.note)
    except ApprovalConflict as error:
        print(f'nothing was changed: {error}', file=sys.stderr)
        return 3
    except ApprovalError as error:
        print(f'refused: {error}', file=sys.stderr)
        return 2
    print(render(entry, past=history(store, case_id)))
    return 0



def attach(store, hold, execution):
    """Remember the allocation before launching; only this hold token may bind its worker."""
    with store._transaction() as cursor:
        from .forecast_publication import lock_case
        lock_case(store, cursor, hold['case'])
        store._execute(cursor, '''SELECT payload FROM eforecast_model_holds
            WHERE namespace=? AND case_id=? AND attempt_id=? AND token=? AND released_at IS NULL''',
            (store.namespace,hold['case'],hold['attempt_id'],hold['hold_token']))
        row=cursor.fetchone()
        if row is None:
            raise ApprovalConflict('forecast hold changed before worker attachment')
        payload=json.loads(row[0])
        if payload.get('execution') and payload['execution'] != execution:
            raise ApprovalConflict('forecast hold is already attached to another worker')
        payload['execution']=execution
        store._execute(cursor, '''UPDATE eforecast_model_holds SET payload=?
            WHERE namespace=? AND case_id=? AND attempt_id=? AND token=? AND released_at IS NULL''',
            (json.dumps(payload,sort_keys=True),store.namespace,hold['case'],hold['attempt_id'],hold['hold_token']))
        if cursor.rowcount != 1:
            raise ApprovalConflict('forecast hold changed during attachment')


def reconcile_holds(store, case, settled):
    """Release only positively settled worker attempts. Unknown evidence retains protection.

    Probe outside the SQL transaction: publication takes the allocator lock before SQL, so
    probing an allocator while holding SQL would introduce the reverse lock order.
    """
    with store._transaction() as cursor:
        store._execute(cursor, '''SELECT attempt_id,token,payload FROM eforecast_model_holds
            WHERE namespace=? AND case_id=? AND released_at IS NULL''', (store.namespace,case))
        rows=cursor.fetchall()
    released, unknown=[],[]
    for attempt,token,raw in rows:
        payload=json.loads(raw)
        try:
            proven=bool(payload.get('execution') and settled(payload['execution']))
        except Exception:
            proven=False
        if proven and release(store,case,payload['forecast_id'],attempt_id=attempt,hold_token=token):
            released.append(attempt)
        else:
            unknown.append(attempt)
    return {'released':released,'retained':unknown}


if __name__ == '__main__':
    raise SystemExit(main())
