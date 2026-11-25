# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The same allocator contract, for machines that cannot share a directory.

``flock`` on a shared folder is not an answer across hosts: NFS locking is unreliable and a
network filesystem gives no ordering a scheduler can trust. A database does, and this adapter
uses the two primitives the plan names -- an advisory lock for the critical section and
conditional updates for the records -- and nothing else. It holds no policy: the same
``admission`` decides, in the same place, with the same inputs.

Three rules differ from the one-host adapter, and all three are consequences of distance:

* the clock is the **server's**, so two hosts whose clocks disagree still expire a claim at the
  same moment;
* liveness is only checkable **where the process is**: a claim of this host whose pid is gone is
  removed at once, and a claim of another host can only expire;
* every transaction is short and explicit, because a lock held across a network partition is the
  one thing worse than no lock at all.
"""
import contextlib
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import socket
import threading
import time
import uuid

from .allocation_contract import Refusal, number
from .allocator_file import FileAllocator, StateUnavailable, _grace_seconds, canonical

log = logging.getLogger(__name__)

LOCK_NAMESPACE = 0x65666331          # 'efc1': the advisory-lock class of this allocator
DEFAULT_TIMEOUT = 30.0
SCHEMA = (
    """create table if not exists eforecast_claims (
           run_id text primary key,
           host text not null,
           pid integer,
           updated double precision not null,
           payload jsonb not null)""",
    """create table if not exists eforecast_fences (
           task text not null,
           trial text not null,
           fence integer not null,
           run_id text,
           attempt integer,
           allocation text,
           taken double precision,
           primary key (task, trial))""",
)
_held = threading.local()
STATE_KEY = 'ledger'
SCHEMA_LEDGER = """create table if not exists eforecast_allocations (
       namespace text not null,
       key text not null,
       payload jsonb not null,
       sealed boolean not null default false,
       updated double precision not null,
       primary key (namespace, key))"""
INITIALIZATION_VERSION = 1
SCHEMA_INITIALIZATION = """create table if not exists eforecast_allocation_initializations (
       namespace text primary key,
       version integer not null,
       initialized_at double precision not null)"""


def dsn():
    return os.environ.get('EFORECAST_ALLOCATOR_DSN')


def namespace():
    """Which deployment this is, so two of them on one server do not share a lock or a table."""
    return os.environ.get('EFORECAST_ALLOCATOR_NAMESPACE', 'default')


def timeout_seconds():
    try:
        return max(.1, float(os.environ.get('EFORECAST_ALLOCATOR_TIMEOUT', DEFAULT_TIMEOUT)))
    except ValueError:
        return DEFAULT_TIMEOUT


def _discard_inherited(connection):
    if not connection.closed:
        os.close(connection.fileno())
        connection.close()


def _transactions():
    """Transactions belong to one thread, process, database and deployment.

    A fork copies libpq sockets too. Close the child's descriptor before disposing of its
    connection, so libpq cannot send a termination message on the parent's connection.
    """
    if getattr(_held, 'pid', None) != os.getpid():
        for entry in getattr(_held, 'transactions', {}).values():
            _discard_inherited(entry['connection'])
        _held.pid, _held.transactions = os.getpid(), {}
    return _held.transactions


def _scope(dsn_value, deployment):
    return (dsn() if dsn_value is None else dsn_value,
            namespace() if deployment is None else deployment)


def connect(dsn_value=None):
    import psycopg2

    connection = psycopg2.connect(dsn() if dsn_value is None else dsn_value)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            for statement in SCHEMA:
                cursor.execute(statement)
    except BaseException:
        connection.close()
        raise
    return connection


@contextlib.contextmanager
def session(*, dsn_value=None, deployment=None):
    """A connection for one operation, or the one the critical section is already holding."""
    scope = _scope(dsn_value, deployment)
    held = _transactions().get(scope)
    if held is not None:
        yield held['connection']
        return
    connection = connect(scope[0])
    owner_pid = os.getpid()
    try:
        yield connection
    finally:
        if os.getpid() == owner_pid:
            connection.close()
        else:
            _discard_inherited(connection)


@contextlib.contextmanager
def critical_section(timeout=None, *, dsn_value=None, deployment=None):
    """Reentrant only for this thread/process and this database/deployment."""
    from eforecast.training.command_line.allocator import AllocatorBusy

    scope = _scope(dsn_value, deployment)
    transactions = _transactions()
    held = transactions.get(scope)
    if held is not None:
        try:
            yield
        except BaseException:
            held['rollback_only'] = True
            raise
        return
    connection = connect(scope[0])
    owner_pid = os.getpid()
    connection.autocommit = False
    wait = timeout_seconds() if timeout is None else timeout
    try:
        with connection.cursor() as cursor:
            cursor.execute('set local lock_timeout = %s', (f'{int(max(1, wait * 1000))}ms',))
            try:
                cursor.execute('select pg_advisory_xact_lock(%s, hashtext(%s))', (LOCK_NAMESPACE, scope[1]))
            except Exception as error:                       # noqa: BLE001 -- a refusal, not a crash
                connection.rollback()
                raise AllocatorBusy(f'the allocation lock of {scope[1]} is held elsewhere ({error})') from None
        held = {'connection': connection, 'rollback_only': False}
        transactions[scope] = held
        try:
            yield
            if os.getpid() != owner_pid:
                raise StateUnavailable('cannot finish a transaction inherited from another process')
            if held['rollback_only']:
                raise StateUnavailable('nested allocator operation failed; transaction rolled back')
            connection.commit()
        except BaseException:
            if os.getpid() == owner_pid:
                connection.rollback()
            raise
        finally:
            transactions.pop(scope, None)
    finally:
        if os.getpid() == owner_pid:
            connection.close()
        else:
            _discard_inherited(connection)


def server_time(cursor):
    cursor.execute('select extract(epoch from now())')
    return float(cursor.fetchone()[0])


# --------------------------------------------------------------------------- fences
def claim_fence(task, trial, run_id, attempt=0):
    allocation = uuid.uuid4().hex
    with session() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                insert into eforecast_fences (task, trial, fence, run_id, attempt, allocation, taken)
                values (%s, %s, 1, %s, %s, %s, extract(epoch from now()))
                on conflict (task, trial) do update
                    set fence = eforecast_fences.fence + 1, run_id = excluded.run_id,
                        attempt = excluded.attempt, allocation = excluded.allocation, taken = excluded.taken
                returning fence""", (task, trial, run_id, attempt, allocation))
            return int(cursor.fetchone()[0])


def read_fences():
    with session() as connection:
        with connection.cursor() as cursor:
            cursor.execute('select task, trial, fence, run_id, attempt, allocation, taken from eforecast_fences')
            return {f'{task}/{trial}': {'fence': fence, 'run_id': run, 'attempt': attempt,
                                        'allocation': allocation, 'taken': taken}
                    for task, trial, fence, run, attempt, allocation, taken in cursor.fetchall()}


# --------------------------------------------------------------------------- claims
def publish_claim(run_id, payload):
    """Upsert what this run holds, stamped with the server's clock rather than this host's."""
    with session() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                insert into eforecast_claims (run_id, host, pid, updated, payload)
                values (%s, %s, %s, extract(epoch from now()), %s)
                on conflict (run_id) do update
                    set host = excluded.host, pid = excluded.pid, updated = excluded.updated,
                        payload = excluded.payload
                returning updated""", (run_id, payload.get('host') or socket.gethostname(),
                                       payload.get('pid'), json.dumps(payload)))
            return float(cursor.fetchone()[0])


def read_claims(run_id, stale_seconds, alive=None):
    """Every live claim but this run's own; what is gone here is removed, what is far away expires."""
    out, forgotten = [], []
    with session() as connection:
        with connection.cursor() as cursor:
            now = server_time(cursor)
            cursor.execute('select run_id, host, pid, updated, payload from eforecast_claims where run_id <> %s',
                           (run_id,))
            rows = cursor.fetchall()
            here = socket.gethostname()
            for other, host, pid, updated, payload in rows:
                expired = now - float(updated) > stale_seconds
                gone = host == here and alive is not None and pid is not None and not alive(pid)
                if expired or gone:
                    forgotten.append((other, 'its heartbeat expired' if expired else 'its process is gone'))
                    continue
                claim = dict(payload)
                claim.update(run_id=other, host=host, pid=pid, updated=float(updated))
                out.append(claim)
            for other, why in forgotten:
                cursor.execute('delete from eforecast_claims where run_id = %s', (other,))
                log.info('shared admission: released the claim of run %s (%s)', other, why)
    return out


def release_claim(run_id):
    with session() as connection:
        with connection.cursor() as cursor:
            cursor.execute('delete from eforecast_claims where run_id = %s', (run_id,))


def reset():
    """For a test that wants an empty deployment; never called by a run."""
    with session() as connection:
        with connection.cursor() as cursor:
            cursor.execute('delete from eforecast_claims')
            cursor.execute('delete from eforecast_fences')


# --------------------------------------------------------------------------- the full contract
class PostgresAllocator(FileAllocator):
    """The same ledger, in a database, for the runs of several machines.

    Only three methods differ from the local adapter -- where the state is read, where it is
    written, and what makes an operation atomic. Everything else, which is all of the contract's
    behaviour, is inherited, so the two adapters cannot drift apart: the adapter-independent suite
    runs against both.

    The lock is one advisory lock per deployment namespace, taken for the length of a transaction
    and no longer. Times come from the **server** unless a clock is injected, so hosts whose clocks
    disagree still expire a lease at the same moment.
    """

    def __init__(self, dsn_or_none=None, capacity=None, *, namespace=None, timeout=None, clock=None,
                 boot_id=None, discover_devices=False, spawn_grace=None):
        import psycopg2                                      # noqa: F401 -- fail here, not mid-transaction

        self.dsn = dsn_or_none or dsn()
        if not self.dsn:
            raise ValueError('a PostgreSQL allocator needs EFORECAST_ALLOCATOR_DSN or an explicit dsn')
        self.namespace = namespace or globals()['namespace']()
        self.spawn_grace = spawn_grace if spawn_grace is not None else _grace_seconds()
        self.capacity = deepcopy(capacity)
        self.timeout = timeout
        self.clock = clock
        # Without an injected clock the server decides what "now" is, which is the point of this
        # adapter: two hosts with different clocks must agree on when a lease ran out.
        self.boot_id = boot_id or 'postgres-deployment'
        self.discover_devices = discover_devices
        self.directory = Path(f'postgres:{self.namespace}')   # for diagnostics only
        self._ensure_schema()

    def _ensure_schema(self):
        with session(dsn_value=self.dsn, deployment=self.namespace) as connection:
            with connection.cursor() as cursor:
                for statement in (*SCHEMA, SCHEMA_LEDGER, SCHEMA_INITIALIZATION):
                    cursor.execute(statement)

    def _now(self):
        if self.clock is not None:
            value = self.clock()
            number(value, 'allocator clock')
            return value
        with session(dsn_value=self.dsn, deployment=self.namespace) as connection:
            with connection.cursor() as cursor:
                return server_time(cursor)

    # ------------------------------------------------------------------ storage
    def _row(self, cursor):
        cursor.execute('select payload, sealed from eforecast_allocations where namespace = %s and key = %s '
                       'for update',
                       (self.namespace, STATE_KEY))
        return cursor.fetchone()

    def _read_wrapper(self):
        with session(dsn_value=self.dsn, deployment=self.namespace) as connection:
            with connection.cursor() as cursor:
                row = self._row(cursor)
        if row is None:
            return None
        if not isinstance(row[0], dict):
            raise StateUnavailable('invalid PostgreSQL allocation wrapper')
        return row[0]

    def _write_wrapper(self, payload, initial=False):
        document = payload.decode() if isinstance(payload, (bytes, bytearray)) else payload
        with session(dsn_value=self.dsn, deployment=self.namespace) as connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    insert into eforecast_allocations (namespace, key, payload, sealed, updated)
                    values (%s, %s, %s::jsonb, %s, extract(epoch from now()))
                    on conflict (namespace, key) do update
                        set payload = excluded.payload, updated = excluded.updated,
                            sealed = eforecast_allocations.sealed or excluded.sealed""",
                               (self.namespace, STATE_KEY, document, bool(initial)))
                if initial:
                    self._mark_initialized(cursor)

    def _mark_initialized(self, cursor):
        """Independent evidence, committed in the same transaction as the first ledger write."""
        cursor.execute('insert into eforecast_allocation_initializations '
                       '(namespace, version, initialized_at) values (%s, %s, extract(epoch from now())) '
                       'on conflict (namespace) do nothing', (self.namespace, INITIALIZATION_VERSION))

    def _sealed(self):
        with session(dsn_value=self.dsn, deployment=self.namespace) as connection:
            with connection.cursor() as cursor:
                cursor.execute('select version from eforecast_allocation_initializations where namespace = %s',
                               (self.namespace,))
                marker = cursor.fetchone()
                if marker is not None:
                    if marker[0] != INITIALIZATION_VERSION:
                        raise StateUnavailable('unknown allocator initialization version')
                    return True
                # Compatibility: FileAllocator._load has validated an intact older wrapper
                # before reaching this branch. Adopt its seal under the deployment lock,
                # without rewriting the payload or inferring state from a missing row.
                row = self._row(cursor)
                if row and row[1]:
                    self._mark_initialized(cursor)
                    return True
        return False

    def _seal(self):
        with session(dsn_value=self.dsn, deployment=self.namespace) as connection:
            with connection.cursor() as cursor:
                cursor.execute('update eforecast_allocations set sealed = true '
                               'where namespace = %s and key = %s', (self.namespace, STATE_KEY))
                self._mark_initialized(cursor)

    # ------------------------------------------------------------------ atomicity
    def _transaction(self, operation):
        import psycopg2

        from .allocator import AllocatorBusy

        try:
            with critical_section(timeout=self.timeout, dsn_value=self.dsn, deployment=self.namespace):
                state = self._load()
                before = canonical(state)
                result = operation(state, self._now())
                if before != canonical(state):
                    self._save(state)
                return result
        except AllocatorBusy as error:
            return Refusal('busy', str(error))
        except (StateUnavailable, psycopg2.Error, OSError, ValueError, TypeError, KeyError,
                AttributeError) as error:
            # Storage trouble and damaged state are refusals, exactly as in the local adapter.
            # Anything else -- a caller's publication callback, above all -- keeps its own
            # exception rather than being reported as a database problem.
            log.error('allocation state unavailable in %s: %s', self.namespace, error)
            return Refusal('unavailable', str(error))

    def reset_ledger(self):
        """Destructive test cleanup, never automatic recovery: forget this namespace atomically."""
        with critical_section(timeout=self.timeout, dsn_value=self.dsn, deployment=self.namespace):
            with session(dsn_value=self.dsn, deployment=self.namespace) as connection:
                with connection.cursor() as cursor:
                    cursor.execute('delete from eforecast_allocations where namespace = %s', (self.namespace,))
                    cursor.execute('delete from eforecast_allocation_initializations where namespace = %s',
                                   (self.namespace,))
