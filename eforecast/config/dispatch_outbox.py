# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A durable outbox for task dispatch -- A16-04.

The run store is already the authority on what work exists and what state it is in. An
outbox row is written **in the same transaction** that makes a task eligible, so there is
never a moment where a task is eligible and nothing records that it must be dispatched. That
is the whole reason for the pattern: a publish is not a transaction, so publishing inside one
would either lose the message on rollback or lose the row on a crash between the two.

What the broker carries is a small, validated **reference**: identities, a runtime id, an
input digest, a target and bounded requirements. Never a dataset, a pickle, a credential or a
command to run -- a consumer that trusted a command in a message would be executing whatever
reached the queue, and a message is not a trust boundary.

Four states, and the transitions are deliberately one-way:

    pending --publish--> published --claim--> claimed
       \\                    \\
        `--------failed-------`

`claimed` is the handoff. The consumer records it **before** acknowledging delivery, so after
that commit A09 recovery owns the launch even if the consumer dies before starting a process
-- which is what keeps the acknowledgement from staying open for hours of fitting.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time

#: Added to the run store's own schema, so one migration creates everything.
SCHEMA = (
    '''CREATE TABLE IF NOT EXISTS eforecast_dispatch_outbox (
       namespace TEXT NOT NULL, dispatch_id TEXT NOT NULL, run_id TEXT NOT NULL,
       task_id TEXT NOT NULL, attempt_id TEXT NOT NULL, generation BIGINT NOT NULL,
       state TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL,
       created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL,
       attempts INTEGER NOT NULL DEFAULT 0, claimed_by TEXT, confirmation TEXT,
       failure TEXT,
       PRIMARY KEY (namespace, dispatch_id))''',
    '''CREATE INDEX IF NOT EXISTS eforecast_dispatch_outbox_pending
       ON eforecast_dispatch_outbox (namespace, state, created_at, dispatch_id)''',
    '''CREATE INDEX IF NOT EXISTS eforecast_dispatch_outbox_run
       ON eforecast_dispatch_outbox (namespace, run_id, task_id, attempt_id)''',
    # A delivery that is not a dispatch has no identity we can trust, so it is recorded under a
    # digest of its own bytes. Without this the only trace of a malformed message would be a
    # dead-letter queue, which the task forbids as the sole record of unfinished work.
    '''CREATE TABLE IF NOT EXISTS eforecast_dispatch_undeliverable (
       namespace TEXT NOT NULL, digest TEXT NOT NULL, reason TEXT NOT NULL,
       bytes INTEGER NOT NULL, deliveries INTEGER NOT NULL,
       first_seen DOUBLE PRECISION NOT NULL, last_seen DOUBLE PRECISION NOT NULL,
       PRIMARY KEY (namespace, digest))''',
)

VERSION = 1
PENDING, PUBLISHED, CLAIMED, FAILED = 'pending', 'published', 'claimed', 'failed'
COMPLETE = 'complete'

#: How a delivery can be wrong, and who owns it. A classification exists so a maintainer is
#: told which of these it is rather than reading a traceback.
MALFORMED = 'malformed'          # not a message this version understands
UNSUPPORTED = 'unsupported'      # understood, but names a runtime or target we cannot serve
UNROUTABLE = 'unroutable'        # the broker had nowhere to put it
OBSOLETE = 'obsolete'            # understood and valid, but the task moved on: a recorded no-op
EXHAUSTED = 'exhausted'          # redelivered past the bound

#: A message is a reference. Anything that looks like a payload or a secret is refused.
FORBIDDEN_KEYS = frozenset({'command', 'argv', 'script', 'env', 'environment', 'secret',
                            'secrets', 'password', 'token', 'credential', 'credentials',
                            'data', 'dataset', 'payload_bytes', 'pickle', 'image_tar'})
MAX_MESSAGE_BYTES = 8 * 1024
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z')


class OutboxError(RuntimeError):
    """The outbox refused something, with the reason a maintainer needs."""


class Obsolete(OutboxError):
    """The message is valid but the task has moved on. A recorded no-op, not a failure."""


# ----------------------------------------------------------------- identity

def dispatch_id(run_id, task_id, attempt_id, generation):
    """The identity of one dispatch, derived rather than invented.

    Deriving it is what makes republication safe: a publisher that crashed after the broker
    accepted a message, but before recording the confirmation, republishes **the same
    identity**, and the consumer's claim is idempotent on it. A random id would make that
    second publish a second piece of work.

    The generation is in the digest, so a new attempt of the same task is a different
    dispatch and an old message cannot claim it.
    """
    for name, value in (('run_id', run_id), ('task_id', task_id), ('attempt_id', attempt_id)):
        if not isinstance(value, str) or not _ID.match(value):
            raise OutboxError(f'{name}: {value!r} is not an identifier')
    if type(generation) is not int or generation < 0:
        raise OutboxError(f'generation: {generation!r} is not a nonnegative integer')
    body = f'{run_id}\x1f{task_id}\x1f{attempt_id}\x1f{generation}'
    return 'd-' + hashlib.sha256(body.encode()).hexdigest()[:32]


def canonical(value):
    """Stable JSON, so a digest of a message means the same thing on both sides."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


# ----------------------------------------------------------------- the message

def message(*, run_id, task_id, attempt_id, generation, runtime, input_digest, target,
            requirements, deadline=None):
    """A validated dispatch reference, small enough to be obviously not a payload."""
    if not isinstance(runtime, str) or not _ID.match(runtime):
        raise OutboxError(f'runtime: {runtime!r} is not a runtime id')
    if not isinstance(input_digest, str) or not re.fullmatch(r'[0-9a-f]{64}', input_digest):
        raise OutboxError('input_digest: expected a sha256 hex digest of the accepted inputs')
    if not isinstance(target, str) or not _ID.match(target):
        raise OutboxError(f'target: {target!r} is not an execution target')
    if not isinstance(requirements, dict) or not requirements:
        raise OutboxError('requirements: expected a bounded mapping')
    for key, value in requirements.items():
        if key in FORBIDDEN_KEYS:
            raise OutboxError(f'requirements.{key}: a dispatch message carries references, '
                              f'not {key}')
        if isinstance(value, (bool, int, str)) or (isinstance(value, float) and math.isfinite(value)):
            continue
        raise OutboxError(f'requirements.{key}: expected one finite scalar')
    if deadline is not None and not (isinstance(deadline, (int, float)) and deadline > 0):
        raise OutboxError('deadline: expected a positive number of seconds, or None')

    document = {
        'version': VERSION,
        'dispatch_id': dispatch_id(run_id, task_id, attempt_id, generation),
        'identity': {'run_id': run_id, 'task_id': task_id, 'attempt_id': attempt_id,
                     'generation': generation},
        'runtime': runtime,
        'input_digest': input_digest,
        'target': target,
        'requirements': dict(sorted(requirements.items())),
    }
    if deadline is not None:
        document['deadline_seconds'] = float(deadline)
    validate(document)
    return document


def validate(document):
    """Everything a consumer checks before it believes a delivery.

    Raises `OutboxError` classified as `MALFORMED`; the caller records that rather than
    retrying something that will never become valid.
    """
    if not isinstance(document, dict) or type(document.get('version')) is not int or document.get('version') != VERSION:
        raise OutboxError(f'{MALFORMED}: not a version {VERSION} dispatch message')
    expected = {'version', 'dispatch_id', 'identity', 'runtime', 'input_digest', 'target',
                'requirements'}
    extra = set(document) - expected - {'deadline_seconds'}
    if extra:
        raise OutboxError(f'{MALFORMED}: unexpected field(s) {", ".join(sorted(extra))}')
    if expected - set(document):
        raise OutboxError(f'{MALFORMED}: missing {", ".join(sorted(expected - set(document)))}')
    identity = document['identity']
    if not isinstance(identity, dict) or set(identity) != {'run_id', 'task_id', 'attempt_id',
                                                           'generation'}:
        raise OutboxError(f'{MALFORMED}: identity must name the run, task, attempt and generation')
    if document['dispatch_id'] != dispatch_id(identity['run_id'], identity['task_id'],
                                              identity['attempt_id'], identity['generation']):
        raise OutboxError(f'{MALFORMED}: dispatch_id does not derive from the identity it carries')
    for key in ('runtime', 'target'):
        if not isinstance(document[key], str) or not _ID.fullmatch(document[key]):
            raise OutboxError(f'{MALFORMED}: {key} must be an identifier')
    if not isinstance(document['input_digest'], str) or not re.fullmatch(r'[0-9a-f]{64}', document['input_digest']):
        raise OutboxError(f'{MALFORMED}: input_digest must be a sha256 digest')
    requirements = document['requirements']
    if not isinstance(requirements, dict) or not requirements:
        raise OutboxError(f'{MALFORMED}: requirements must be a nonempty mapping')
    for key, value in requirements.items():
        if not isinstance(key, str) or not isinstance(value, (bool, int, str, float)) or (
                isinstance(value, float) and not math.isfinite(value)):
            raise OutboxError(f'{MALFORMED}: requirements must contain finite scalars')
    deadline = document.get('deadline_seconds')
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0):
        raise OutboxError(f'{MALFORMED}: invalid deadline_seconds')
    try:
        encoded = canonical(document)
    except (ValueError, TypeError) as error:
        raise OutboxError(f'{MALFORMED}: invalid JSON') from error
    if len(encoded.encode()) > MAX_MESSAGE_BYTES:
        raise OutboxError(f'{MALFORMED}: {len(encoded)} bytes exceeds the {MAX_MESSAGE_BYTES} '
                          f'byte bound; a dispatch message carries references, not payloads')
    forbidden = FORBIDDEN_KEYS & _keys(document)
    if forbidden:
        raise OutboxError(f'{MALFORMED}: {", ".join(sorted(forbidden))} has no place in a '
                          f'dispatch message')
    # Reuses A16-10's detector rather than a second idea of what a secret looks like.
    from eforecast.config import secrets_source
    leaked = secrets_source.leaks(encoded)
    if leaked:
        raise OutboxError(f'{MALFORMED}: the message contains {", ".join(leaked)}')
    return document


def _keys(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        found.update(value)
        for item in value.values():
            _keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _keys(item, found)
    return found


# ----------------------------------------------------------------- the store side

from contextlib import contextmanager     # noqa: E402  (kept beside the code that uses it)

COLUMNS = ('dispatch_id', 'run_id', 'task_id', 'attempt_id', 'generation', 'state', 'version',
           'payload', 'created_at', 'updated_at', 'attempts', 'claimed_by', 'confirmation',
           'failure')


@contextmanager
def _in(store, cursor):
    """Either join the caller's transaction or open one.

    Joining is the point of an outbox: the row and the state change that made it necessary
    commit together or not at all. Opening one is for a reconciliation, where the state
    change already happened and committed.
    """
    if cursor is not None:
        yield cursor
        return
    with store._transaction() as owned:
        yield owned


def _row(values):
    record = dict(zip(COLUMNS, values))
    record['payload'] = json.loads(record['payload'])
    return record


def enqueue(store, document, *, cursor=None, now=None):
    """Record that a task must be dispatched. Idempotent on the dispatch identity.

    A second call with the same identity and the same message is the normal case -- a
    republish after a crash -- and changes nothing. A second call with the same identity and
    a *different* message is a bug worth raising on, because the two cannot both be true.
    """
    validate(document)
    moment = time.time() if now is None else now
    identity = document['identity']
    encoded = canonical(document)
    with _in(store, cursor) as handle:
        store._execute(handle, '''INSERT INTO eforecast_dispatch_outbox
            (namespace,dispatch_id,run_id,task_id,attempt_id,generation,state,version,payload,
             created_at,updated_at,attempts)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,0)
            ON CONFLICT (namespace,dispatch_id) DO NOTHING''',
            (store.namespace, document['dispatch_id'], identity['run_id'], identity['task_id'],
             identity['attempt_id'], identity['generation'], PENDING, VERSION, encoded,
             moment, moment))
        if handle.rowcount == 0:
            store._execute(handle, '''SELECT payload FROM eforecast_dispatch_outbox
                WHERE namespace=? AND dispatch_id=?''', (store.namespace, document['dispatch_id']))
            existing = handle.fetchone()
            if existing and existing[0] != encoded:
                raise OutboxError('dispatch identity reused with different contents; the same '
                                  'dispatch cannot describe two pieces of work')
            return False
        return True


def pending(store, *, limit=32, states=(PENDING,), cursor=None, target=None):
    """The oldest rows awaiting publication, bounded.

    Bounded because a publisher that drained an unbounded backlog in one pass would hold a
    transaction open for as long as that took, and because A09/A12 fairness decides what runs
    -- the publisher's job is transport, not policy.
    """
    if not isinstance(limit, int) or not 0 < limit <= 1024:
        raise OutboxError('limit: expected 1..1024')
    marks = ','.join('?' for _ in states)
    # Filter before LIMIT so unrelated backlog cannot starve a dedicated consumer.
    clause = '' if target is None else (" AND payload::jsonb->>'target'=?" if store.placeholder == '%s'
                                      else " AND json_extract(payload,'$.target')=?")
    with _in(store, cursor) as handle:
        store._execute(handle, f'''SELECT {",".join(COLUMNS)} FROM eforecast_dispatch_outbox
            WHERE namespace=? AND state IN ({marks}){clause}
            ORDER BY created_at, dispatch_id LIMIT {int(limit)}''',
            (store.namespace, *states, *((target,) if target is not None else ())))
        return [_row(values) for values in handle.fetchall()]


def get(store, dispatch_identity, *, cursor=None):
    with _in(store, cursor) as handle:
        store._execute(handle, f'''SELECT {",".join(COLUMNS)} FROM eforecast_dispatch_outbox
            WHERE namespace=? AND dispatch_id=?''', (store.namespace, dispatch_identity))
        values = handle.fetchone()
    return _row(values) if values else None


def mark_published(store, dispatch_identity, confirmation, *, cursor=None, now=None):
    """Record the broker's confirmation. Never moves a claimed row backwards.

    A confirmation arriving after the consumer already claimed the work is late, not wrong:
    the publisher crashed before recording it and republished, and the second confirmation
    refers to work already handed off. Overwriting `claimed` with `published` would make
    recovery re-dispatch finished work.
    """
    moment = time.time() if now is None else now
    with _in(store, cursor) as handle:
        store._execute(handle, '''UPDATE eforecast_dispatch_outbox
            SET state=?, confirmation=?, updated_at=?, attempts=attempts+1
            WHERE namespace=? AND dispatch_id=? AND state=?''',
            (PUBLISHED, str(confirmation), moment, store.namespace, dispatch_identity, PENDING))
        if handle.rowcount:
            return PUBLISHED
        store._execute(handle, '''SELECT state FROM eforecast_dispatch_outbox
            WHERE namespace=? AND dispatch_id=?''', (store.namespace, dispatch_identity))
        row = handle.fetchone()
        if row is None:
            raise OutboxError(f'no dispatch {dispatch_identity}: a confirmation arrived for '
                              f'work this store never recorded')
        return row[0]


def claim(store, dispatch_identity, consumer, *, launch=None, document=None,
          cursor=None, now=None):
    """Take durable ownership of a launch, before the delivery is acknowledged.

    This is the handoff. After it commits, A09 recovery owns an unfinished launch even if
    this consumer dies before creating a process -- which is why the acknowledgement does not
    have to stay open while a model fits.

    Returns ('claimed', row) the first time and ('already', row) on a redelivery, so a
    duplicate is a recorded no-op rather than a second launch. Never claims a failed row.
    """
    if not isinstance(consumer, str) or not _ID.match(consumer):
        raise OutboxError(f'consumer: {consumer!r} is not an identifier')
    moment = time.time() if now is None else now
    with _in(store, cursor) as handle:
        store._execute(handle, '''UPDATE eforecast_dispatch_outbox
            SET state=?, claimed_by=?, updated_at=?
            WHERE namespace=? AND dispatch_id=? AND state IN (?,?)''',
            (CLAIMED, consumer, moment, store.namespace, dispatch_identity, PENDING, PUBLISHED))
        taken = handle.rowcount
        store._execute(handle, f'''SELECT {",".join(COLUMNS)} FROM eforecast_dispatch_outbox
            WHERE namespace=? AND dispatch_id=?''', (store.namespace, dispatch_identity))
        values = handle.fetchone()
        if document is not None and values is not None and canonical(_row(values)['payload']) != canonical(document):
            raise OutboxError('delivered dispatch differs from the durable record')
        if taken and launch is not None:
            # Inside the claim, so the two commit together: a launch intent without a claim would
            # be dispatched twice, and a claim without the intent would be work nobody owns. If
            # this raises, the transaction rolls back and the dispatch stays deliverable.
            launch(store, document, cursor=handle)
        store._execute(handle, f'''SELECT {",".join(COLUMNS)} FROM eforecast_dispatch_outbox
            WHERE namespace=? AND dispatch_id=?''', (store.namespace, dispatch_identity))
        values = handle.fetchone()
    if values is None:
        raise Obsolete(f'{OBSOLETE}: no dispatch {dispatch_identity} is recorded here')
    row = _row(values)
    if taken:
        return CLAIMED, row
    if row['state'] == CLAIMED:
        return 'already', row
    raise Obsolete(f'{OBSOLETE}: dispatch {dispatch_identity} is {row["state"]}, '
                   f'so this delivery is a no-op')


def record_failure(store, dispatch_identity, classification, detail, *, cursor=None, now=None):
    """A persistent record of a delivery that cannot be made to work.

    Persistent on purpose: a dead-letter queue is the broker's memory, and the task says not
    to rely on it as the only record of unfinished work. This row outlives the queue.
    """
    if classification not in (MALFORMED, UNSUPPORTED, UNROUTABLE, OBSOLETE, EXHAUSTED):
        raise OutboxError(f'classification: {classification!r} is not one this store records')
    moment = time.time() if now is None else now
    failure = canonical({'classification': classification, 'detail': str(detail)[:1000],
                         'observed_at': moment})
    with _in(store, cursor) as handle:
        store._execute(handle, '''UPDATE eforecast_dispatch_outbox
            SET state=?, failure=?, updated_at=? WHERE namespace=? AND dispatch_id=? AND state NOT IN (?,?)''',
            (FAILED, failure, moment, store.namespace, dispatch_identity, CLAIMED, COMPLETE))
        return bool(handle.rowcount)


def reconcile(store, eligible, *, cursor=None, now=None):
    """Re-record dispatches the store still shows as eligible but the outbox has lost.

    After a queue is deleted or a delivery expires, the broker is not evidence of anything.
    The run store is, so eligibility is read from it and any dispatch it implies is enqueued
    again -- under the same derived identity, so this is idempotent and a message already in
    flight stays valid.
    """
    restored = []
    with _in(store, cursor) as handle:
        for document in eligible:
            validate(document)
            existing = get(store, document['dispatch_id'], cursor=handle)
            if existing is None:
                enqueue(store, document, cursor=handle, now=now)
                restored.append(document['dispatch_id'])
            elif existing['state'] == PUBLISHED:
                # published but never claimed: the message may be gone with the queue, so it
                # becomes publishable again under the same identity
                store._execute(handle, '''UPDATE eforecast_dispatch_outbox
                    SET state=?, updated_at=? WHERE namespace=? AND dispatch_id=? AND state=?''',
                    (PENDING, time.time() if now is None else now, store.namespace,
                     document['dispatch_id'], PUBLISHED))
                if handle.rowcount:
                    restored.append(document['dispatch_id'])
    return restored


def record_undeliverable(store, digest, reason, size, *, cursor=None, now=None):
    """Remember a delivery that was not a dispatch, by the digest of its own bytes.

    Counted rather than accumulated: a broker redelivering the same rubbish raises `deliveries`
    instead of filling the table, and an operator can see both how often and what it was.
    """
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise OutboxError('an undeliverable record is keyed by a sha256 digest of the delivery')
    moment = time.time() if now is None else now
    with _in(store, cursor) as handle:
        store._execute(handle, '''INSERT INTO eforecast_dispatch_undeliverable
            (namespace,digest,reason,bytes,deliveries,first_seen,last_seen) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT (namespace,digest) DO UPDATE SET deliveries=eforecast_dispatch_undeliverable.deliveries+1,
            last_seen=excluded.last_seen, reason=excluded.reason''',
            (store.namespace, digest, str(reason)[:1024], int(size), 1, moment, moment))


def undeliverable(store, *, limit=32, cursor=None):
    """The persistent failed-delivery record a maintainer reads, oldest first."""
    if type(limit) is not int or not 1 <= limit <= 1024:
        raise OutboxError(f'bounded listing expected 1..1024, got {limit!r}')
    with _in(store, cursor) as handle:
        store._execute(handle, '''SELECT digest,reason,bytes,deliveries,first_seen,last_seen
            FROM eforecast_dispatch_undeliverable WHERE namespace=?
            ORDER BY first_seen, digest LIMIT ?''', (store.namespace, limit))
        return [dict(zip(('digest', 'reason', 'bytes', 'deliveries', 'first_seen', 'last_seen'), row))
                for row in handle.fetchall()]


# ---------------------------------------------------------------------- eligibility

#: What the eligibility hook may copy out of a stage contract's options. Everything else stays
#: in the run store, which the consumer re-reads before it claims anything: a message exists to
#: say *which* work became eligible, not to carry the work's configuration.
CONTRACT_REQUIREMENTS = ('cpu_threads', 'ram_mb', 'disk_mb', 'gpu_count')

#: Stage dispatches run in the worker interpreter on this installation. A16-03 decides a
#: foundation family's venue per attempt; a stage has one.
STAGE_RUNTIME = 'worker'
STAGE_TARGET = 'local'


def stage_dispatches(run):
    """The dispatch messages for the stage tasks this run has made eligible.

    Eligible means what the run projection already means by it: the stage is pending, nothing
    blocks it, and every dependency has an observed successful outcome. The projection is reused
    rather than re-derived, so the outbox cannot disagree with what the UI shows an operator.
    """
    if not run.tasks or len(run.tasks) < 2:
        return []
    from .stage_workflow import project
    by_id = {task.id: task for task in run.tasks}
    found = []
    for item in project(run):
        if item['state'] != 'pending' or item['blocked'] or item['outcome'] is not None:
            continue
        task = by_id[item['id']]
        if not task.attempts:
            continue
        requirements = {name: value for name, value in (item['options'] or {}).items()
                        if name in CONTRACT_REQUIREMENTS
                        and isinstance(value, (bool, int, float, str))}
        requirements.update(stage=item['stage'], scope=item['scope'] or 'global')
        found.append(message(
            run_id=run.id, task_id=task.id, attempt_id=task.attempts[-1].id,
            # The attempt number, so a retried attempt is a different dispatch and a redelivered
            # message for the previous one cannot claim it.
            generation=task.attempts[-1].number + 1,
            runtime=STAGE_RUNTIME, target=STAGE_TARGET,
            # The pinned request is the immutable input reference: the plan identity, the
            # catalogue version, the stage contract and the effective revision, as stored.
            input_digest=hashlib.sha256((task.request_json or '').encode()).hexdigest(),
            requirements=requirements))
    return found


def enqueue_eligible(store, run, *, cursor=None, now=None):
    """Write the outbox rows for one run, in the transaction that changed its state.

    Called from the run store's report collection, which is where a stage becomes eligible. It is
    idempotent by construction -- the identity is derived from run/task/attempt/generation -- so
    being called on every report collection costs a bounded upsert per eligible stage and never
    produces a second dispatch for the same attempt.

    A run whose cancellation has been requested enqueues nothing: there is no point publishing
    work the consumer would immediately refuse.
    """
    if getattr(run, 'cancel', None) is not None:          # a requested cancellation is one
        return []
    if json.loads(run.metadata_json).get('dispatch_route', {}).get('venue') == 'broker':
        return []  # one managed Job already owns the accepted entry point and its internal stages
    written = []
    for document in stage_dispatches(run):
        if enqueue(store, document, cursor=cursor, now=now):
            written.append(document['dispatch_id'])
    return written
