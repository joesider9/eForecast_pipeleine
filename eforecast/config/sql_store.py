# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The SQL connection the durable state sits on, knowing nothing about what is stored in it.

Split out of ``run_store`` so a machine that only forecasts can keep its approvals and its
published forecasts where the host keeps them, without importing the host's run model -- and,
through that, the training packages the inference image does not carry (``tests/test_online.py``
holds that boundary). ``run_store`` re-exports these names; the storage behaviour is unchanged.

Each subclass carries the tables it needs in ``schema``: the host declares all of them, a
forecast-only installation declares the few it uses. Every statement is ``IF NOT EXISTS``, so
the two can share one database and whichever opens it first creates what it needs.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import sqlite3

from eforecast.config.service import ServiceError


class StoreUnavailable(ServiceError):
    """Durable state could not be reached; callers must not silently continue without it."""


class SQLStore:
    """The parts that do not depend on what is being stored: placeholders and migration."""

    placeholder = '?'
    schema: tuple = ()

    def _execute(self, cursor, sql, params=()):
        return cursor.execute(sql.replace('?', self.placeholder), params)

    def _initialize(self):
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError('run store namespace must be nonempty')
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('run store timeout must be positive and finite')
        with self._transaction() as cursor:
            if self.placeholder == '%s':
                cursor.execute('SELECT pg_advisory_xact_lock(1701208946, 1)')
            for statement in self.schema:
                cursor.execute(statement)


def _local_path(path, workspace=None):
    path = Path(path).expanduser().resolve()
    if workspace is not None and path.is_relative_to(Path(workspace).expanduser().resolve()):
        raise ValueError('SQLite run state must be outside the case workspace, on local disk')
    sync_names = {'dropbox', 'onedrive', 'googledrive', 'google drive', 'nextcloud'}
    if any(part.casefold() in sync_names for part in path.parts):
        raise ValueError('SQLite run state cannot be in a synchronized folder')
    # Reject known network mounts too. Sync clients with arbitrary folder names cannot be
    # discovered reliably: the operator must still choose genuinely local storage.
    try:
        mounts = []
        for line in Path('/proc/mounts').read_text().splitlines():
            fields = line.split()
            mount = Path(fields[1].replace('\\040', ' '))
            if path.is_relative_to(mount):
                mounts.append((len(mount.parts), fields[2]))
        filesystem = max(mounts)[1] if mounts else ''
        if filesystem in {'nfs', 'nfs4', 'cifs', 'smb3', '9p', 'ceph', 'glusterfs', 'fuse.sshfs'}:
            raise ValueError('SQLite run state requires local disk; use PostgreSQL for shared storage')
    except OSError as error:
        raise ValueError('cannot verify SQLite mount; choose a verifiable local disk') from error
    return path


class SQLiteStore(SQLStore):
    def __init__(self, path, *, namespace='default', timeout=5.0, workspace=None,
                 schema=None):
        if str(path) == ':memory:':
            raise ValueError('durable run state cannot use an in-memory database')
        self.path = _local_path(path, workspace)
        self.namespace, self.timeout = namespace, float(timeout)
        if schema is not None:
            self.schema = tuple(schema)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Reserve a private file; never change an existing operator file.
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
        except OSError as error:
            raise StoreUnavailable('local run store unavailable; no storage fallback was attempted') from error
        self._initialize()

    @contextmanager
    def _transaction(self, write=True):
        connection = None
        try:
            connection = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
            connection.execute('PRAGMA synchronous=FULL')
            connection.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield connection.cursor()
            connection.commit()
        except sqlite3.Error as error:
            raise StoreUnavailable('local run store unavailable; no storage fallback was attempted') from error
        finally:
            if connection is not None:
                connection.close()  # rolls back on every exceptional exit


class PostgresStore(SQLStore):
    placeholder = '%s'

    def __init__(self, dsn, *, namespace='default', timeout=5.0, schema=None):
        try:
            import psycopg2
        except ImportError as error:
            raise StoreUnavailable('PostgreSQL run store needs psycopg2; no storage fallback was attempted') from error
        if not dsn:
            raise ValueError('PostgreSQL run storage requires a DSN')
        self._driver, self._dsn = psycopg2, dsn
        self.namespace, self.timeout = namespace, float(timeout)
        if schema is not None:
            self.schema = tuple(schema)
        self._initialize()

    @contextmanager
    def _transaction(self, write=True):
        connection = None
        try:
            connection = self._driver.connect(self._dsn, connect_timeout=max(1, math.ceil(self.timeout)))
            connection.set_session(isolation_level='READ COMMITTED')
            with connection.cursor() as cursor:
                milliseconds = str(max(1, int(self.timeout * 1000)))
                cursor.execute("SELECT set_config('statement_timeout', %s, true)", (milliseconds,))
                cursor.execute("SELECT set_config('lock_timeout', %s, true)", (milliseconds,))
                yield cursor
            connection.commit()
        except self._driver.Error as error:
            raise StoreUnavailable('PostgreSQL run store unavailable; no storage fallback was attempted') from error
        finally:
            if connection is not None:
                connection.close()


def configured(workspace, *, sqlite=SQLiteStore, postgres=PostgresStore, schema=None):
    """Open the durable state this installation is configured for.

    One place reads the environment, so the host and a forecast-only machine cannot end up
    pointing at different databases. `schema` lets a caller that needs only a few tables say so;
    the host passes its store classes, which carry the full set.
    """
    dsn = os.environ.get('EFORECAST_RUN_STORE_DSN') or os.environ.get('EFORECAST_ALLOCATOR_DSN')
    backend = os.environ.get('EFORECAST_RUN_STORE', 'postgres' if dsn else 'sqlite')
    namespace = os.environ.get('EFORECAST_RUN_STORE_NAMESPACE',
                               os.environ.get('EFORECAST_ALLOCATOR_NAMESPACE', 'default'))
    timeout = float(os.environ.get('EFORECAST_RUN_STORE_TIMEOUT', '5'))
    if backend == 'postgres':
        return postgres(dsn, namespace=namespace, timeout=timeout, schema=schema)
    if backend != 'sqlite':
        raise ValueError('EFORECAST_RUN_STORE must be sqlite or postgres')
    state = Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local' / 'state'))
    path = os.environ.get('EFORECAST_RUN_STORE_PATH', state / 'eforecast' / 'runs.sqlite3')
    return sqlite(path, namespace=namespace, timeout=timeout, workspace=workspace, schema=schema)
