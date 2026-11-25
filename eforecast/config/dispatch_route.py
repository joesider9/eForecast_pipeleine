# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A16-08: which venue a scope's work is dispatched to, and how that decision changes.

A cutover is not a configuration file. It is a decision with four properties, and this module
exists because a route that is only an environment variable has none of them:

* **durable and attributable.** The venue of a scope is a row, with who decided it, when, and
  why. "Why is this running on the cluster?" has an answer that outlives a shell.
* **recorded per accepted run.** `venue_of()` reads the venue the run was *accepted* under,
  from the run itself. Changing a route cannot retarget work that is already in flight.
* **drained before it changes.** A scope with outstanding dispatches cannot be rerouted: an
  attempt must not be claimable by two venues, and old messages must not come back to a
  venue nobody is listening on. The store checks the drain in the decision transaction.
* **never implicit.** The default is `local`, the tested path. A broker that is unreachable
  does not become a local run, and a local installation does not become a cluster one because
  somebody exported a URL -- the only way to change the venue is to decide it here.

The managed broker executor is not integrated yet. Its historical decisions remain readable,
but new broker cutovers and launches are refused, with no local fallback.
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
import hashlib

from .service import ServiceError

#: Historical venue vocabulary; availability is checked separately by require_executable.
LOCAL, BROKER = 'local', 'broker'
VENUES = (LOCAL, BROKER)

#: What a scope is: the whole installation, or one case. A cutover starts with one small scope,
#: which is why this is not a single global switch.
DEFAULT_SCOPE = 'installation'

SCHEMA = (
    '''CREATE TABLE IF NOT EXISTS eforecast_dispatch_routes (
       namespace TEXT NOT NULL, scope TEXT NOT NULL, venue TEXT NOT NULL,
       revision INTEGER NOT NULL, decided_at REAL NOT NULL, decided_by TEXT NOT NULL,
       note TEXT NOT NULL, PRIMARY KEY (namespace, scope))''',
    '''CREATE TABLE IF NOT EXISTS eforecast_dispatch_route_history (
       namespace TEXT NOT NULL, scope TEXT NOT NULL, revision INTEGER NOT NULL,
       venue TEXT NOT NULL, decided_at REAL NOT NULL, decided_by TEXT NOT NULL,
       note TEXT NOT NULL, PRIMARY KEY (namespace, scope, revision))''',
)


class RouteError(ServiceError):
    """The route was not changed, and the message says what would have to be true first."""


class RouteBusy(RouteError):
    """Work is outstanding for this scope; a reroute would strand or double-claim it."""


def _in(store, cursor):
    return nullcontext(cursor) if cursor is not None else store._transaction()


def store(workspace=None):
    """Open the run store: draining must include accepted runs, not just the outbox."""
    from .run_store import configured_store
    return configured_store(workspace)


def lock(store, cursor):
    """Serialize route decisions with accepted submissions, including an absent first row."""
    if store.placeholder == '%s':
        key = int.from_bytes(hashlib.sha256(('dispatch-route:' + store.namespace).encode()).digest()[:8],
                             'big', signed=True)
        store._execute(cursor, 'SELECT pg_advisory_xact_lock(?)', (key,))


def case_scope(owner_id, case_id):
    return 'case:' + hashlib.sha256((owner_id + '/' + case_id).encode()).hexdigest()


def execution_problem(venue):
    if venue == LOCAL:
        return None
    from .kubernetes_runs import configured_target
    try:
        if configured_target() is not None:
            return None
    except ServiceError as error:
        return str(error)
    return ('Broker/Kubernetes execution is not connected to the managed stage lifecycle. '
            'Keep this scope local until the durable launch, completion and cleanup adapter '
            'has passed its integrated qualification; a broker URL is not that adapter.')


def require_executable(venue):
    problem = execution_problem(venue)
    if problem:
        raise RouteError(problem)


def current(store, scope=DEFAULT_SCOPE, *, cursor=None):
    """The decision for this scope, or the default one nobody has changed.

    A scope with no row is `local` at revision 0 -- not "unknown": the local container path is
    what this installation runs when nobody has decided otherwise, and saying so is the point.
    """
    with _in(store, cursor) as handle:
        store._execute(handle, '''SELECT venue,revision,decided_at,decided_by,note
            FROM eforecast_dispatch_routes WHERE namespace=? AND scope=?''',
            (store.namespace, scope))
        row = handle.fetchone()
    if row is None:
        return {'scope': scope, 'venue': LOCAL, 'revision': 0, 'decided_at': None,
                'decided_by': None, 'note': 'the default: no cutover has been decided',
                'default': True}
    venue, revision, decided_at, decided_by, note = row
    return {'scope': scope, 'venue': venue, 'revision': int(revision),
            'decided_at': float(decided_at), 'decided_by': decided_by, 'note': note,
            'default': False}


def effective(store, case=None, *, cursor=None):
    """The venue a new run of this case would be accepted under: the case, else the installation.

    One narrow scope is how a cutover starts, so a case decision wins over the installation's.
    """
    with _in(store, cursor) as handle:
        if case:
            case_route = current(store, case, cursor=handle)
            if not case_route['default']:
                return case_route
        return current(store, DEFAULT_SCOPE, cursor=handle)


def _matches(run, scope):
    return scope == DEFAULT_SCOPE or scope in {
        json.loads(run.metadata_json).get('case'), case_scope(run.owner_id, run.case_id)}


def _drained(run):
    if run.state != 'terminal' or run.cleanup not in {'complete', 'not_required'}:
        return False
    return not any(attempt.state == 'unknown' and attempt.cleanup != 'complete'
                   for task in run.tasks for attempt in task.attempts)


def _retire_finished(store, scope, cursor, now):
    """Invalidate unclaimed messages for completed, cleaned-up runs in this decision.

    A claimed intent may represent a launcher not yet attached to the run. It must be
    reconciled by that executor; route changes cannot invent its cleanup evidence.
    """
    from .dispatch_outbox import FAILED, OBSOLETE, PENDING, PUBLISHED, canonical
    from .run_model import Run
    store._execute(cursor, """SELECT d.dispatch_id,r.payload FROM eforecast_dispatch_outbox d
        JOIN eforecast_runs r ON r.namespace=d.namespace AND r.id=d.run_id
        WHERE d.namespace=? AND d.state IN (?,?)""", (store.namespace, PENDING, PUBLISHED))
    for dispatch_id, payload in cursor.fetchall():
        run = Run.from_json(payload)
        if _matches(run, scope) and _drained(run):
            failure = canonical({'classification': OBSOLETE, 'observed_at': now,
                                 'detail': 'run completed and cleanup resolved before route decision'})
            store._execute(cursor, """UPDATE eforecast_dispatch_outbox SET state=?,failure=?,updated_at=?
                WHERE namespace=? AND dispatch_id=? AND state IN (?,?)""",
                (FAILED, failure, now, store.namespace, dispatch_id, PENDING, PUBLISHED))


def outstanding(store, scope, *, cursor=None):
    """Dispatch rows for this scope that a reroute would strand. Empty means drained.

    Read from the outbox, which is where eligible work is durable: `pending` has not been
    published, `published` is in a queue, `claimed` is being launched. Any of the three means
    an attempt exists that a second venue must not be able to claim.
    """
    from .dispatch_outbox import CLAIMED, PENDING, PUBLISHED
    from .run_model import Run
    result, seen = [], set()
    with _in(store, cursor) as handle:
        # A run can be accepted before it has stage/outbox rows. Unknown execution and
        # unresolved cleanup still own the case, even after a terminal outcome.
        store._execute(handle, """SELECT payload FROM eforecast_runs WHERE namespace=?""",
                       (store.namespace,))
        runs = {}
        for (payload,) in handle.fetchall():
            run = Run.from_json(payload)
            runs[run.id] = run
            if _matches(run, scope) and not _drained(run):
                result.append({'dispatch_id': 'run:' + run.id, 'state': run.state,
                               'run_id': run.id, 'task_id': None})
                seen.add(run.id)
        states = (PENDING, PUBLISHED, CLAIMED)
        store._execute(handle, """SELECT dispatch_id,state,run_id,task_id FROM eforecast_dispatch_outbox
            WHERE namespace=? AND state IN (?,?,?)""", (store.namespace, *states))
        for dispatch, state, run_id, task_id in handle.fetchall():
            run = runs.get(run_id)
            # An orphan has no trustworthy scope or cleanup evidence. Retain it visibly.
            if run is None:
                result.append(dict(dispatch_id=dispatch, state=state, run_id=run_id, task_id=task_id))
            elif run_id not in seen:
                if _matches(run, scope):
                    result.append(dict(dispatch_id=dispatch, state=state, run_id=run_id, task_id=task_id))
    return result


def decide(store, scope, venue, *, by, note='', expected_revision=None, drained=None,
           cursor=None, now=None):
    """Change a scope's venue, or refuse and say what has to be true first.

    `drained` can add a blocker for compatibility with callers that also inspect a queue.
    It cannot override the authoritative run/outbox check inside this transaction.
    """
    if venue not in VENUES:
        raise RouteError(f'{venue!r} is not a recognised venue; this lab has {", ".join(VENUES)}')
    if not by:
        raise RouteError('a route decision is attributable: name who made it')
    moment = time.time() if now is None else now
    with _in(store, cursor) as handle:
        lock(store, handle)
        existing = current(store, scope, cursor=handle)
        if expected_revision is not None and existing['revision'] != expected_revision:
            raise RouteError(f'this scope is at revision {existing["revision"]}, not '
                             f'{expected_revision}; somebody decided in between. Read it again.')
        if existing['venue'] == venue and (not existing['default'] or scope == DEFAULT_SCOPE):
            require_executable(venue)
            return {**existing, 'changed': False}
        _retire_finished(store, scope, handle, moment)
        # Caller evidence may add a blocker, never suppress authoritative durable state.
        waiting = outstanding(store, scope, cursor=handle) + list(drained or [])
        if waiting:
            raise RouteBusy(
                f'{len(waiting)} dispatch(es) are outstanding for {scope}: a reroute would let '
                f'two venues claim the same attempt, or strand a message where nobody is '
                f'listening. Drain or durably cancel them first '
                f'(for example {waiting[0]["dispatch_id"]} is {waiting[0]["state"]}).')
        require_executable(venue)
        revision = existing['revision'] + 1
        record = (store.namespace, scope, revision, venue, moment, by, note or '')
        store._execute(handle, '''INSERT INTO eforecast_dispatch_route_history
            (namespace,scope,revision,venue,decided_at,decided_by,note) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT (namespace,scope,revision) DO NOTHING''', record)
        store._execute(handle, '''INSERT INTO eforecast_dispatch_routes
            (namespace,scope,venue,revision,decided_at,decided_by,note) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT (namespace,scope) DO UPDATE SET venue=excluded.venue,
            revision=excluded.revision, decided_at=excluded.decided_at,
            decided_by=excluded.decided_by, note=excluded.note''',
            (store.namespace, scope, venue, revision, moment, by, note or ''))
        return {**current(store, scope, cursor=handle), 'changed': True}


def unavailable_routes(store, *, cursor=None, limit=16):
    """Bounded read-only inventory so doctor cannot hide a blocked case behind local default."""
    with _in(store, cursor) as handle:
        store._execute(handle, """SELECT scope,venue FROM eforecast_dispatch_routes
            WHERE namespace=? AND venue<>? ORDER BY scope LIMIT ?""", (store.namespace, LOCAL, limit))
        return [{'scope': scope, 'venue': venue} for scope, venue in handle.fetchall()
                if execution_problem(venue)]


def history(store, scope, *, limit=32, cursor=None):
    """Every decision for this scope, newest first. A rollback is a decision, not an erasure."""
    if type(limit) is not int or not 1 <= limit <= 512:
        raise RouteError(f'bounded history expected 1..512, got {limit!r}')
    with _in(store, cursor) as handle:
        store._execute(handle, '''SELECT revision,venue,decided_at,decided_by,note
            FROM eforecast_dispatch_route_history WHERE namespace=? AND scope=?
            ORDER BY revision DESC LIMIT ?''', (store.namespace, scope, limit))
        rows = handle.fetchall()
    return [{'revision': int(revision), 'venue': venue, 'decided_at': float(at),
             'decided_by': by, 'note': note} for revision, venue, at, by, note in rows]


def venue_of(run) -> dict:
    """The venue a run was **accepted** under, read from the run itself.

    Not from today's configuration: a cutover decided while this run is in flight must not
    retarget it, and a rollback must not make a finished run look as though it ran somewhere
    else. A run accepted before this field existed says so rather than guessing.
    """
    metadata = run if isinstance(run, dict) else {}
    recorded = metadata.get('dispatch_route')
    if isinstance(recorded, dict) and recorded.get('venue') in VENUES:
        return {'venue': recorded['venue'], 'scope': recorded.get('scope'),
                'revision': recorded.get('revision'), 'recorded': True}
    if recorded is not None:
        raise RouteError('The accepted dispatch route is malformed; repair its provenance before launch.')
    return {'venue': LOCAL, 'scope': None, 'revision': None, 'recorded': False,
            'note': 'this run was accepted before its venue was recorded; it ran locally'}


def stamp(store, case, *, owner_id=None, case_id=None, cursor=None) -> dict:
    """What `submit` records on a run at acceptance, so `venue_of` can answer later."""
    canonical = case_scope(owner_id, case_id) if owner_id and case_id else case
    decided = effective(store, canonical, cursor=cursor)
    # Historical label-scoped decisions remain readable, with their origin recorded. New
    # CLI decisions resolve a real case to its owner/case identity.
    if case and canonical != case and current(store, canonical, cursor=cursor)['default']:
        decided = effective(store, case, cursor=cursor)
    result = {'venue': decided['venue'], 'scope': decided['scope'],
              'revision': decided['revision']}
    if decided['venue'] == BROKER:
        from .kubernetes_runs import configured_target
        target = configured_target()
        if target is not None:
            result['execution'] = target
    return result


def main(argv=None):
    """`eforecast route` -- read, decide or roll back where a scope's work is dispatched.

        bin/eforecast route                      what this installation does today
        bin/eforecast route <scope>              what one case or the installation does
        bin/eforecast route <scope> <venue> --note "why"      decide it
        bin/eforecast route <scope> --history    every decision, newest first

    A decision is refused while work is outstanding for that scope, because two venues must
    never be able to claim one attempt. The refusal names a dispatch to look at.
    """
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(prog='eforecast route', description=main.__doc__.splitlines()[0])
    parser.add_argument('scope', nargs='?', default=DEFAULT_SCOPE)
    parser.add_argument('venue', nargs='?', choices=VENUES)
    parser.add_argument('--note', default='')
    parser.add_argument('--expect-revision', type=int, default=None,
                        help='the revision you believe is current; refuses if it moved')
    parser.add_argument('--history', action='store_true')
    arguments = parser.parse_args(argv)

    try:
        return _command(arguments, store(os.environ.get('EFORECAST_WORKSPACE')))
    except RouteBusy as error:
        print(f'nothing was changed: {error}', file=sys.stderr)
        return 3
    except (ServiceError, OSError) as error:
        print(f'nothing was changed: {error}', file=sys.stderr)
        return 2


def _command(arguments, durable):
    import getpass
    import os
    import sys
    legacy_scope = None
    if arguments.scope != DEFAULT_SCOPE and not arguments.scope.startswith('case:'):
        legacy_scope = arguments.scope
        from .service import Workspace
        from .run_scope import resolve
        workspace = Workspace()
        if not workspace.exists(arguments.scope):
            raise RouteError(f'case {arguments.scope!r} does not exist in this workspace/tenant')
        owner, case_id = resolve(workspace, durable, arguments.scope)
        arguments.scope = case_scope(owner, case_id)
    if arguments.history:
        for item in history(durable, arguments.scope):
            when = time.strftime('%Y-%m-%d %H:%M', time.gmtime(item['decided_at']))
            print(f"r{item['revision']:<3d} {item['venue']:<8s} {when} UTC  "
                  f"{item['decided_by']}  {item['note']}")
        return 0
    if arguments.venue is None:
        decided = effective(durable, arguments.scope)
        if legacy_scope and current(durable, arguments.scope)['default']:
            decided = effective(durable, legacy_scope)
        print(f"scope    {decided['scope']}")
        print(f"venue    {decided['venue']}" + ('  (default: nobody has decided)' if decided['default'] else ''))
        if not decided['default']:
            when = time.strftime('%Y-%m-%d %H:%M', time.gmtime(decided['decided_at']))
            print(f"decided  revision {decided['revision']} by {decided['decided_by']} at {when} UTC")
            if decided['note']:
                print(f"note     {decided['note']}")
        problem = execution_problem(decided['venue'])
        if problem:
            print(f'blocked  {problem}')
        waiting = outstanding(durable, arguments.scope)
        print(f"waiting  {len(waiting)} outstanding dispatch(es)"
              + ('; a change is refused until these are drained' if waiting else ''))
        return 0
    try:
        decided = decide(durable, arguments.scope, arguments.venue,
                         by=os.environ.get('EFORECAST_USER') or getpass.getuser(),
                         note=arguments.note, expected_revision=arguments.expect_revision)
    except RouteBusy as error:
        print(f'nothing was changed: {error}', file=sys.stderr)
        return 3
    except RouteError as error:
        print(f'nothing was changed: {error}', file=sys.stderr)
        return 2
    if not decided['changed']:
        print(f"{arguments.scope} already dispatches to {decided['venue']}")
        return 0
    print(f"{arguments.scope} now dispatches to {decided['venue']} (revision {decided['revision']}).")
    print('Runs already accepted keep the venue they were accepted under.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
