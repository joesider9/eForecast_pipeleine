# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Transactional run storage, independent of the resource allocator's ledger.

Connections are operation-local (no inherited or shared transaction context). SQLite is for a
local disk; PostgreSQL is for shared deployments. Neither adapter starts or observes workers.
"""
from __future__ import annotations

from dataclasses import replace
import json
from typing import Protocol

from eforecast.config.run_model import InvalidRun, Run, identity, pinned, transition, validate
from eforecast.config.service import ServiceError
# The connection lives apart from the run model, so a forecast-only installation can open the
# same database without importing the host. StoreUnavailable is re-exported from here because
# callers have always raised and caught it from this module.
from eforecast.config.sql_store import (PostgresStore, SQLiteStore, SQLStore,    # noqa: F401
                                        StoreUnavailable, _local_path, configured)
# (SQLStore, StoreUnavailable and _local_path are re-exported: callers have always imported
#  them from here, and the tests reach for _local_path by that name.)


class RunConflict(ServiceError):
    """The identity exists, or another writer advanced this revision. Reload before deciding."""


class RunNotFound(ServiceError):
    pass


class RunStore(Protocol):
    def identity(self, kind: str, key: str) -> str: ...
    def create(self, run: Run) -> Run: ...
    def get(self, owner_id: str, case_id: str, run_id: str) -> Run: ...
    def list(self, owner_id: str, case_id: str, *, include_deleted: bool = False) -> list[Run]: ...
    def compare_and_set(self, run: Run, expected_revision: int) -> Run: ...
    def submission(self, owner_id: str, case_id: str, key: str, request_hash: str) -> Run | None: ...
    def submit(self, run: Run, key: str, request_hash: str) -> tuple[Run, bool]: ...
    def pending(self) -> list[Run]: ...
    def active_page(self, *, after=None, limit=32): ...
    def collect_reports(self, run: Run, events: list[dict], offset: int) -> Run: ...
    def children(self, owner_id, case_id, run_id, *, after='', limit=100): ...
    def child_summary(self, run: Run) -> dict: ...
    def collect_cleanup(self, run: Run, updates: dict) -> Run: ...
    def request_cancel(self, owner_id: str, case_id: str, run_id: str, cancel) -> Run: ...


from eforecast.config.dispatch_outbox import SCHEMA as _DISPATCH_SCHEMA
from eforecast.config.model_approval import SCHEMA as _APPROVAL_SCHEMA
from eforecast.config.forecast_publication import SCHEMA as _PUBLICATION_SCHEMA
from eforecast.config.verified_completions import SCHEMA as _COMPLETION_SCHEMA
from eforecast.config.dispatch_route import SCHEMA as _ROUTE_SCHEMA
from eforecast.config.kubernetes_runs import SCHEMA as _KUBERNETES_RUN_SCHEMA
from eforecast.config.dispatch_outbox import enqueue_eligible as _enqueue_eligible

SCHEMA = (
    '''CREATE TABLE IF NOT EXISTS eforecast_run_children (
       namespace TEXT NOT NULL, run_id TEXT NOT NULL, node_id TEXT NOT NULL,
       parent_id TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL, payload TEXT NOT NULL,
       PRIMARY KEY (namespace,run_id,node_id))''',
    '''CREATE INDEX IF NOT EXISTS eforecast_run_children_parent
       ON eforecast_run_children (namespace,run_id,parent_id)''',
    '''CREATE TABLE IF NOT EXISTS eforecast_run_events (
       namespace TEXT NOT NULL, run_id TEXT NOT NULL, event_id TEXT NOT NULL, payload TEXT NOT NULL,
       PRIMARY KEY (namespace,run_id,event_id))''',
    '''CREATE TABLE IF NOT EXISTS eforecast_run_identities (
       namespace TEXT NOT NULL, kind TEXT NOT NULL, locator TEXT NOT NULL, id TEXT NOT NULL,
       PRIMARY KEY (namespace, kind, locator), UNIQUE (namespace, id))''',
    '''CREATE TABLE IF NOT EXISTS eforecast_runs (
       namespace TEXT NOT NULL, id TEXT NOT NULL, owner_id TEXT NOT NULL, case_id TEXT NOT NULL,
       revision BIGINT NOT NULL, created_at DOUBLE PRECISION NOT NULL, payload TEXT NOT NULL,
       PRIMARY KEY (namespace, id))''',
    '''CREATE INDEX IF NOT EXISTS eforecast_runs_case
       ON eforecast_runs (namespace, owner_id, case_id, created_at, id)''',
    '''CREATE INDEX IF NOT EXISTS eforecast_runs_created
       ON eforecast_runs (namespace,created_at,id)''',
    '''CREATE TABLE IF NOT EXISTS eforecast_run_submissions (
       namespace TEXT NOT NULL, owner_id TEXT NOT NULL, client_key TEXT NOT NULL,
       case_id TEXT NOT NULL, run_id TEXT NOT NULL, request_hash TEXT NOT NULL,
       PRIMARY KEY (namespace, owner_id, client_key))''',
    '''CREATE TABLE IF NOT EXISTS eforecast_run_case_claims (
       namespace TEXT NOT NULL, owner_id TEXT NOT NULL, case_id TEXT NOT NULL, run_id TEXT,
       PRIMARY KEY (namespace, owner_id, case_id))''',
    # A16-04's outbox, created by the same migration as everything else: the outbox row is
    # written in the transaction that changes run state, so it belongs to this store's
    # schema, and a separate migration step would be one more thing that can be half-applied.
    *_DISPATCH_SCHEMA,
    # A15-02's approval pointer, its history and the pins an accepted forecast holds. Same
    # migration as everything else: the pointer moves inside this store's transaction, so a
    # schema that could be half-applied relative to it would be one more thing to get wrong.
    *_APPROVAL_SCHEMA,
    # A15-03's immutable issues and the last-valid pointer, in the same migration: the pointer
    # moves inside this store's transaction, so it belongs to this store's schema.
    *_PUBLICATION_SCHEMA,
    # A16-06's cumulative counter: its own rows, so deleting history cannot change it.
    *_COMPLETION_SCHEMA,
    # A16-08's route decisions, which outlive the shell that made them.
    *_ROUTE_SCHEMA,
    *_KUBERNETES_RUN_SCHEMA,
)


class _SQLStore:
    """Everything the host does with the run model; the connection comes from SQLStore."""

    def identity(self, kind, key):
        if kind not in {'owner', 'case'} or not isinstance(key, str) or not key:
            raise InvalidRun('identity needs an owner/case kind and a stable locator')
        with self._transaction() as cursor:
            self._execute(cursor, '''INSERT INTO eforecast_run_identities
                (namespace, kind, locator, id) VALUES (?, ?, ?, ?)
                ON CONFLICT (namespace, kind, locator) DO NOTHING''',
                (self.namespace, kind, key, identity()))
            self._execute(cursor, '''SELECT id FROM eforecast_run_identities
                WHERE namespace=? AND kind=? AND locator=?''', (self.namespace, kind, key))
            return cursor.fetchone()[0]

    @staticmethod
    def _record(row):
        if row is None:
            raise RunNotFound('run not found')
        try:
            record = Run.from_json(row[1])
            if (record.revision, record.id, record.owner_id, record.case_id, record.created_at) != (
                    row[0], row[2], row[3], row[4], row[5]):
                raise InvalidRun('run indexes disagree with their payload')
            return record
        except InvalidRun as error:
            raise StoreUnavailable('durable run record is unreadable; recovery is required') from error

    def create(self, run):
        validate(run)
        if run.revision != 0:
            raise InvalidRun('a new run must have revision zero')
        with self._transaction() as cursor:
            self._execute(cursor, '''INSERT INTO eforecast_runs
                (namespace,id,owner_id,case_id,revision,created_at,payload) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT (namespace,id) DO NOTHING''',
                (self.namespace, run.id, run.owner_id, run.case_id, 0, run.created_at, run.to_json()))
            if cursor.rowcount != 1:
                raise RunConflict('run identity already exists')
        return run

    def _get(self, cursor, owner_id, case_id, run_id):
        self._execute(cursor, '''SELECT revision,payload,id,owner_id,case_id,created_at FROM eforecast_runs
            WHERE namespace=? AND owner_id=? AND case_id=? AND id=?''',
            (self.namespace, owner_id, case_id, run_id))
        return self._record(cursor.fetchone())

    def _submission(self, cursor, owner_id, case_id, key, request_hash):
        # owner_id identifies workspace + tenant, not the submitting account. Keys and
        # immutable replay provenance are shared by that tenant across its cases.
        self._execute(cursor, '''SELECT case_id,run_id,request_hash FROM eforecast_run_submissions
            WHERE namespace=? AND owner_id=? AND client_key=?''', (self.namespace, owner_id, key))
        row = cursor.fetchone()
        if row is None:
            return None
        if (row[0], row[2]) != (case_id, request_hash):
            raise RunConflict('submission key already belongs to a different request')
        try:
            return self._get(cursor, owner_id, case_id, row[1])
        except RunNotFound as error:
            raise StoreUnavailable('accepted submission lost its run record; recovery is required') from error

    def submission(self, owner_id, case_id, key, request_hash):
        with self._transaction(write=False) as cursor:
            return self._submission(cursor, owner_id, case_id, key, request_hash)

    @staticmethod
    def _blocks(run):
        # Unknown execution and unresolved cleanup cannot safely admit another writer.
        return run.state != 'terminal' or run.cleanup not in {'not_required', 'complete'}

    def submit(self, run, key, request_hash):
        validate(run)
        if run.revision != 0 or run.state != 'pending' or not isinstance(key, str) or not key or len(key) > 200:
            raise InvalidRun('submission needs a pending revision-zero run and a client key of 1–200 characters')
        if not isinstance(request_hash, str) or len(request_hash) != 64 or any(c not in '0123456789abcdef' for c in request_hash):
            raise InvalidRun('submission needs a SHA-256 request hash')
        with self._transaction() as cursor:
            from . import dispatch_route
            dispatch_route.lock(self, cursor)
            # Insert first: the unique owner/key constraint serializes concurrent retries,
            # including requests naming different cases. A losing case claim rolls it back.
            self._execute(cursor, '''INSERT INTO eforecast_run_submissions
                (namespace,owner_id,client_key,case_id,run_id,request_hash) VALUES (?,?,?,?,?,?)
                ON CONFLICT (namespace,owner_id,client_key) DO NOTHING''',
                (self.namespace, run.owner_id, key, run.case_id, run.id, request_hash))
            if cursor.rowcount == 0:
                return self._submission(cursor, run.owner_id, run.case_id, key, request_hash), False
            self._execute(cursor, '''INSERT INTO eforecast_run_case_claims
                (namespace,owner_id,case_id,run_id) VALUES (?,?,?,NULL)
                ON CONFLICT (namespace,owner_id,case_id) DO NOTHING''',
                (self.namespace, run.owner_id, run.case_id))
            suffix = ' FOR UPDATE' if self.placeholder == '%s' else ''
            self._execute(cursor, '''SELECT run_id FROM eforecast_run_case_claims
                WHERE namespace=? AND owner_id=? AND case_id=?''' + suffix,
                (self.namespace, run.owner_id, run.case_id))
            claimed = cursor.fetchone()[0]
            if claimed is not None:
                active = self._get(cursor, run.owner_id, run.case_id, claimed)
                if self._blocks(active):
                    raise RunConflict(f'case already has a running or unresolved run ({active.id})')
            # A09-01 records predate the claim table: normalize their ownership in memory,
            # rather than allowing a new submission to bypass them or rewriting history.
            self._execute(cursor, '''SELECT revision,payload,id,owner_id,case_id,created_at FROM eforecast_runs
                WHERE namespace=? AND owner_id=? AND case_id=?''',
                (self.namespace, run.owner_id, run.case_id))
            for row in cursor.fetchall():
                active = self._record(row)
                if self._blocks(active):
                    raise RunConflict(f'case already has a running or unresolved run ({active.id})')
            metadata = json.loads(run.metadata_json)
            metadata['dispatch_route'] = dispatch_route.stamp(self, metadata.get('case'),
                owner_id=run.owner_id, case_id=run.case_id, cursor=cursor)
            dispatch_route.require_executable(metadata['dispatch_route']['venue'])
            run = replace(run, metadata_json=pinned(metadata))
            self._execute(cursor, '''INSERT INTO eforecast_runs
                (namespace,id,owner_id,case_id,revision,created_at,payload) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT (namespace,id) DO NOTHING''',
                (self.namespace, run.id, run.owner_id, run.case_id, 0, run.created_at, run.to_json()))
            if cursor.rowcount != 1:
                raise RunConflict('run identity already exists')
            if metadata['dispatch_route'].get('execution') is not None:
                from .kubernetes_runs import enqueue
                enqueue(self, run, cursor=cursor)
            self._execute(cursor, '''UPDATE eforecast_run_case_claims SET run_id=?
                WHERE namespace=? AND owner_id=? AND case_id=?''',
                (run.id, self.namespace, run.owner_id, run.case_id))
        return run, True

    def pending(self):
        with self._transaction(write=False) as cursor:
            self._execute(cursor, '''SELECT r.revision,r.payload,r.id,r.owner_id,r.case_id,r.created_at
                FROM eforecast_runs r JOIN eforecast_run_case_claims c
                ON c.namespace=r.namespace AND c.owner_id=r.owner_id AND c.case_id=r.case_id AND c.run_id=r.id
                WHERE r.namespace=? ORDER BY r.created_at,r.id''', (self.namespace,))
            records = [self._record(row) for row in cursor.fetchall()]
            return [run for run in records if not run.deleted and run.state in
                    {'pending', 'preparing', 'queued', 'running', 'unknown'}]

    def active_page(self, *, after=None, limit=32):
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError('active page limit must be 1–256')
        where, params = '', [self.namespace]
        if after is not None:
            where = ' AND (r.created_at>? OR (r.created_at=? AND r.id>?))'
            params += [after[0], after[0], after[1]]
        with self._transaction(write=False) as cursor:
            self._execute(cursor, '''SELECT r.revision,r.payload,r.id,r.owner_id,r.case_id,r.created_at
                FROM eforecast_runs r JOIN eforecast_run_case_claims c ON c.namespace=r.namespace
                AND c.owner_id=r.owner_id AND c.case_id=r.case_id AND c.run_id=r.id
                WHERE r.namespace=?''' + where + ' ORDER BY r.created_at,r.id LIMIT ?', (*params, limit))
            return [self._record(row) for row in cursor.fetchall()]

    def collect_reports(self, run, events, offset):
        if not (run.snapshot_json and json.loads(run.snapshot_json).get('stage_plan', {}).get('completion_version') == 1):
            return self._collect_reports(run, events, offset)
        from .stage_artifacts import verification_boundary, verify_completion, IncompleteStage
        from .run_reporting import validate_event
        # Hash large shared inputs outside the database writer transaction. Merge rechecks the
        # same receipt and stat/ctime signatures inside it; changed files cannot use the cache.
        with verification_boundary():
            if run.snapshot_json and json.loads(run.snapshot_json).get('stage_plan', {}).get('completion_version') == 1:
                handle = run.tasks[0].attempts[-1].executor
                context = {'run_id': run.id, 'task_id': run.tasks[0].id,
                    'attempt_id': run.tasks[0].attempts[-1].id, 'executor_id': handle.id}
                for event in events:
                    if event.get('type') != 'stage' or not event.get('completion'):
                        continue
                    validate_event(event, context)
                    task = next((t for t in run.tasks[1:] if t.id == event['stage_task_id']), None)
                    if task and task.state != 'terminal' and (event.get('outcome') or {}).get('kind') in {'success', 'nothing_to_do'}:
                        try:
                            verify_completion(run, task, event)
                        except (IncompleteStage, OSError, KeyError, ValueError, TypeError) as error:
                            raise InvalidRun(f'invalid stage completion: {error}') from error
            return self._collect_reports(run, events, offset)

    def _collect_reports(self, run, events, offset):
        from eforecast.config.run_model import pinned
        from eforecast.config.run_reporting import merge_child, validate_event
        handle = run.tasks[0].attempts[-1].executor
        context = {'run_id': run.id, 'task_id': run.tasks[0].id,
                   'attempt_id': run.tasks[0].attempts[-1].id, 'executor_id': handle.id}
        result_json = run.result_json
        with self._transaction() as cursor:
            for event in events:
                validate_event(event, context)
                encoded = pinned(event)
                self._execute(cursor, '''INSERT INTO eforecast_run_events (namespace,run_id,event_id,payload)
                    VALUES (?,?,?,?) ON CONFLICT (namespace,run_id,event_id) DO NOTHING''',
                    (self.namespace, run.id, event['event_id'], encoded))
                if cursor.rowcount == 0:
                    self._execute(cursor, '''SELECT payload FROM eforecast_run_events
                        WHERE namespace=? AND run_id=? AND event_id=?''', (self.namespace, run.id, event['event_id']))
                    if cursor.fetchone()[0] != encoded:
                        raise InvalidRun('event identity reused with different contents')
                    continue
                if event['type'] == 'result':
                    candidate = pinned(event['result'])
                    if result_json is not None and candidate != result_json:
                        raise InvalidRun('worker published conflicting structured outcomes')
                    result_json = candidate
                    continue
                if event['type'] == 'stage':
                    from .stage_workflow import merge
                    run = merge(run, event)
                    continue
                self._execute(cursor, '''SELECT payload FROM eforecast_run_children
                    WHERE namespace=? AND run_id=? AND node_id=?''', (self.namespace, run.id, event['node_id']))
                previous = cursor.fetchone()
                child = merge_child(json.loads(previous[0]) if previous else None, event)
                state = 'terminal' if child['outcome'] is not None else 'running' if child['execution'] else 'pending'
                self._execute(cursor, '''INSERT INTO eforecast_run_children
                    (namespace,run_id,node_id,parent_id,kind,state,payload) VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT (namespace,run_id,node_id) DO UPDATE SET state=excluded.state,payload=excluded.payload''',
                    (self.namespace, run.id, child['node_id'], child['parent_id'], child['kind'], state, pinned(child)))
            # A16-04: the dispatch record is written here, in the transaction that just made a
            # stage eligible, so "eligible with no dispatch record" is not a reachable state. It
            # is idempotent on a derived identity, so collecting reports repeatedly is free.
            _enqueue_eligible(self, run, cursor=cursor)
            return self._cas(cursor, replace(run, report_offset=offset, result_json=result_json, report_error=None), run.revision)

    def children(self, owner_id, case_id, run_id, *, after='', limit=100):
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError('child page limit must be 1–256')
        with self._transaction(write=False) as cursor:
            self._get(cursor, owner_id, case_id, run_id)  # tenant/case boundary before reading child records
            self._execute(cursor, '''SELECT node_id,payload FROM eforecast_run_children
                WHERE namespace=? AND run_id=? AND node_id>? ORDER BY node_id LIMIT ?''',
                (self.namespace, run_id, after, limit + 1))
            rows = cursor.fetchall()
            return {'items': [json.loads(row[1]) for row in rows[:limit]],
                    'next': rows[limit - 1][0] if len(rows) > limit else None}

    def collect_cleanup(self, run, updates):
        """Publish a bounded child cleanup page and its cursor in one revisioned transaction."""
        if len(updates) > 64:
            raise ValueError('cleanup page exceeds 64 children')
        with self._transaction() as cursor:
            for node_id, cleanup in updates.items():
                self._execute(cursor, '''SELECT payload FROM eforecast_run_children
                    WHERE namespace=? AND run_id=? AND node_id=?''', (self.namespace, run.id, node_id))
                row = cursor.fetchone()
                if row is None:
                    raise RunConflict('cleanup child no longer exists')
                child = json.loads(row[0])
                child['cleanup'] = cleanup
                self._execute(cursor, '''UPDATE eforecast_run_children SET payload=?
                    WHERE namespace=? AND run_id=? AND node_id=?''', (pinned(child), self.namespace, run.id, node_id))
            return self._cas(cursor, run, run.revision)

    def child_summary(self, run):
        with self._transaction(write=False) as cursor:
            self._execute(cursor, '''SELECT kind,state,COUNT(*) FROM eforecast_run_children
                WHERE namespace=? AND run_id=? GROUP BY kind,state''', (self.namespace, run.id))
            groups = [{'kind': row[0], 'state': row[1], 'count': row[2]} for row in cursor.fetchall()]
            self._execute(cursor, '''WITH RECURSIVE rooted(node_id) AS (
                SELECT node_id FROM eforecast_run_children WHERE namespace=? AND run_id=? AND parent_id=?
                UNION SELECT c.node_id FROM eforecast_run_children c JOIN rooted p ON c.parent_id=p.node_id
                WHERE c.namespace=? AND c.run_id=?)
                SELECT (SELECT COUNT(*) FROM eforecast_run_children WHERE namespace=? AND run_id=?)-COUNT(*) FROM rooted''',
                (self.namespace, run.id, run.tasks[0].attempts[-1].id, self.namespace, run.id, self.namespace, run.id))
            return {'groups': groups, 'unresolved': sum(row['count'] for row in groups if row['state'] != 'terminal'),
                    'unrooted': cursor.fetchone()[0]}

    def get(self, owner_id, case_id, run_id):
        with self._transaction(write=False) as cursor:
            self._execute(cursor, '''SELECT revision,payload,id,owner_id,case_id,created_at FROM eforecast_runs
                WHERE namespace=? AND owner_id=? AND case_id=? AND id=?''',
                (self.namespace, owner_id, case_id, run_id))
            record = self._record(cursor.fetchone())
            if (record.id, record.owner_id, record.case_id) != (run_id, owner_id, case_id):
                raise StoreUnavailable('durable run identity disagrees with its index')
            return record

    def list(self, owner_id, case_id, *, include_deleted=False):
        with self._transaction(write=False) as cursor:
            self._execute(cursor, '''SELECT revision,payload,id,owner_id,case_id,created_at FROM eforecast_runs
                WHERE namespace=? AND owner_id=? AND case_id=? ORDER BY created_at DESC,id DESC''',
                (self.namespace, owner_id, case_id))
            records = [self._record(row) for row in cursor.fetchall()]
            if any((record.owner_id, record.case_id) != (owner_id, case_id) for record in records):
                raise StoreUnavailable('durable run identity disagrees with its index')
            return [record for record in records if include_deleted or not record.deleted]

    def compare_and_set(self, run, expected_revision):
        if type(expected_revision) is not int or expected_revision < 0 or run.revision != expected_revision:
            raise InvalidRun('a transition must carry the expected revision')
        with self._transaction() as cursor:
            return self._cas(cursor, run, expected_revision)

    def request_cancel(self, owner_id, case_id, run_id, cancel):
        """Linearize user intent against telemetry/collection and coalesce concurrent requests."""
        with self._transaction() as cursor:
            suffix = ' FOR UPDATE' if self.placeholder == '%s' else ''
            self._execute(cursor, '''SELECT id FROM eforecast_runs
                WHERE namespace=? AND owner_id=? AND case_id=? AND id=?''' + suffix,
                (self.namespace, owner_id, case_id, run_id))
            current = self._get(cursor, owner_id, case_id, run_id)
            if current.cancel is not None or (current.state == 'terminal' and current.cleanup in {'complete', 'not_required'}):
                return current
            return self._cas(cursor, replace(current, cancel=cancel, cleanup='pending'), current.revision)

    def _cas(self, cursor, run, expected_revision):
        self._execute(cursor, '''SELECT revision,payload,id,owner_id,case_id,created_at FROM eforecast_runs
            WHERE namespace=? AND owner_id=? AND case_id=? AND id=?''',
            (self.namespace, run.owner_id, run.case_id, run.id))
        old = self._record(cursor.fetchone())
        if old.revision != expected_revision:
            raise RunConflict('run revision changed; reload before transitioning')
        result = transition(old, run)
        self._execute(cursor, '''UPDATE eforecast_runs SET revision=?,payload=?
            WHERE namespace=? AND owner_id=? AND case_id=? AND id=? AND revision=?''',
            (result.revision, result.to_json(), self.namespace, run.owner_id, run.case_id, run.id, expected_revision))
        if cursor.rowcount != 1:
            raise RunConflict('run revision changed; reload before transitioning')
        if not self._blocks(result):
            self._execute(cursor, '''UPDATE eforecast_run_case_claims SET run_id=NULL
                WHERE namespace=? AND owner_id=? AND case_id=? AND run_id=?''',
                (self.namespace, run.owner_id, run.case_id, run.id))
        return result


class SQLiteRunStore(_SQLStore, SQLiteStore):
    """The host's store on a local disk: the run model's tables and its methods."""

    schema = SCHEMA



class PostgresRunStore(_SQLStore, PostgresStore):
    """The host's store on PostgreSQL: the run model's tables and its methods."""

    schema = SCHEMA



def configured_store(workspace) -> RunStore:
    """The host's store, with the whole schema; `sql_store.configured` reads the environment."""
    return configured(workspace, sqlite=SQLiteRunStore, postgres=PostgresRunStore)


class LegacyJSONHistory:
    """Read-only compatibility. Never infer an exit code, outcome or PID liveness."""
    def __init__(self, folder: Path):
        self.folder = folder

    def get(self, run_id):
        if not isinstance(run_id, str) or not run_id.replace('-', '').isalnum():
            raise StoreUnavailable('legacy run identifier is invalid')
        try:
            raw = json.loads((self.folder / f'{run_id}.json').read_text())
        except FileNotFoundError as error:
            raise RunNotFound(f'run {run_id!r} not found') from error
        except (OSError, ValueError) as error:
            raise StoreUnavailable('legacy run record is unreadable') from error
        if not isinstance(raw, dict) or raw.get('id') != run_id:
            raise StoreUnavailable('legacy run record has an invalid identity')
        return {**raw, 'storage': 'legacy_json', 'read_only': True}

    def list(self):
        records = []
        for path in self.folder.glob('*.json'):
            try:
                records.append(self.get(path.stem))
            except (RunNotFound, StoreUnavailable):
                # Preserve the manager's existing list behavior for corrupt JSON history.
                continue
        return records
