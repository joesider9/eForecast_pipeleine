# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Durable case identities shared by runs, approvals and operational forecasts.

Project names are labels. A legacy name-keyed approval is never guessed into a
case: an operator can add an explicit binding without changing the old records.
This module is safe to import in the forecast-only image.
"""
import json
import logging
import os
from pathlib import Path
import uuid

log = logging.getLogger(__name__)

SCHEMA = (
    '''CREATE TABLE IF NOT EXISTS eforecast_run_identities (
       namespace TEXT NOT NULL, kind TEXT NOT NULL, locator TEXT NOT NULL, id TEXT NOT NULL,
       PRIMARY KEY (namespace,kind,locator), UNIQUE (namespace,id))''',
    '''CREATE TABLE IF NOT EXISTS eforecast_serving_bindings (
       namespace TEXT NOT NULL, legacy_key TEXT NOT NULL, case_id TEXT NOT NULL,
       PRIMARY KEY (namespace,legacy_key), UNIQUE (namespace,case_id))''',
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(store, kind, key):
    if hasattr(store, 'identity'):
        return store.identity(kind, key)
    with store._transaction() as cursor:
        store._execute(cursor, '''INSERT INTO eforecast_run_identities
            (namespace,kind,locator,id) VALUES (?,?,?,?)
            ON CONFLICT (namespace,kind,locator) DO NOTHING''',
            (store.namespace, kind, key, uuid.uuid4().hex))
        store._execute(cursor, '''SELECT id FROM eforecast_run_identities
            WHERE namespace=? AND kind=? AND locator=?''', (store.namespace, kind, key))
        return cursor.fetchone()[0]


def resolve(store, *, workspace, tenant, locator):
    owner = identity(store, 'owner', canonical({'workspace': workspace, 'tenant': tenant}))
    case = identity(store, 'case', canonical({'owner': owner, 'locator': locator}))
    return {'owner_id': owner, 'case_id': case}


def for_case(store, workspace, name):
    folder = workspace.case_dir(name)
    meta = workspace._meta(folder)
    return resolve(store,
        workspace=os.environ.get('EFORECAST_RUN_WORKSPACE_ID') or str(workspace.root.resolve()),
        tenant=workspace._tenant_of(folder, meta),
        locator=meta.get('id') or folder.relative_to(workspace.root).as_posix())


def for_models(store, path_model):
    """The identity of *these model files* -- the one fact every route into a case shares.

    A case is reached as a workspace case file, a standalone YAML, a legacy project package or
    a bare ``static_data``, and all four end at one model directory holding the scalers and
    weights a forecast serves. Two routes that reach it serve the same models, so they are one
    case. The YAML path is not that fact: the same case can be reached from two copies of its
    file, and one file can be edited to point somewhere else.
    """
    root = str(Path(path_model).resolve())
    return resolve(store, workspace=os.environ.get('EFORECAST_RUN_WORKSPACE_ID') or
                   str(Path(os.environ.get('EFORECAST_WORKSPACE') or Path(root).parent).resolve()),
                   tenant='standalone', locator=root)


def scope_of(store, case):
    """(owner, case) for a case identity, read back from the locator it was minted from."""
    with store._transaction(write=False) as cursor:
        store._execute(cursor, "SELECT locator FROM eforecast_run_identities"
                       " WHERE namespace=? AND kind=? AND id=?", (store.namespace, 'case', case))
        row = cursor.fetchone()
    try:
        return {'owner_id': json.loads(row[0])['owner'], 'case_id': case}
    except (TypeError, ValueError, KeyError):
        return {'owner_id': '', 'case_id': case}


def bound_case(store, legacy_key):
    """The case a key is bound to, explicitly by an operator or by the alias below."""
    with store._transaction(write=False) as cursor:
        store._execute(cursor, "SELECT case_id FROM eforecast_serving_bindings"
                       " WHERE namespace=? AND legacy_key=?", (store.namespace, legacy_key))
        row = cursor.fetchone()
    return row[0] if row else None


def alias_models(store, case, path_model):
    """Record that this case's models are also reached without the workspace.

    Not a label match: the alias says "the directory these approvals are about is that one",
    which is why a maintainer approving through `eforecast approve <case>` and a cron running
    the same models from a legacy project package no longer land on two identities that never
    meet. An operator's own binding is never overwritten, and a case keeps at most one alias.
    """
    if not path_model:
        return None
    key = for_models(store, path_model)['case_id']
    if key == case:
        return None
    with store._transaction() as cursor:
        store._execute(cursor, "SELECT legacy_key,case_id FROM eforecast_serving_bindings"
                       " WHERE namespace=? AND (legacy_key=? OR case_id=?)",
                       (store.namespace, key, case))
        if cursor.fetchall():
            return None                                 # already decided; leave it alone
        store._execute(cursor, "INSERT INTO eforecast_serving_bindings"
                       " (namespace,legacy_key,case_id) VALUES (?,?,?)"
                       " ON CONFLICT (namespace,legacy_key) DO NOTHING",
                       (store.namespace, key, case))
    return key


def for_config(store, config, static):
    """Resolve before rebasing into a worker's private view; carry only this resolved ID."""
    carried = static.get('_serving_scope')
    if isinstance(carried, dict) and all(isinstance(carried.get(k), str) and carried[k]
                                         for k in ('owner_id', 'case_id')):
        return dict(carried)
    if isinstance(config, (str, os.PathLike)) and Path(config).exists():
        from .service import Workspace
        ws = Workspace(admin=True)
        path = Path(config).resolve()
        for folder in ws._all_case_dirs():
            if path == folder.resolve() or path.is_relative_to(folder.resolve()):
                scope = for_case(store, ws, folder.name)
                # The same models reached without the workspace must not become a second case.
                # Recording that is convergence, not correctness of this forecast: another
                # resolver racing us has recorded the same row, and a store that refuses the
                # write must not take the forecast down with it.
                try:
                    alias_models(store, scope['case_id'], static.get('path_model'))
                except Exception:                       # noqa: BLE001
                    log.debug('models alias not recorded for %s', scope['case_id'], exc_info=True)
                return scope
    # Standalone and legacy installations resolve by the models they serve, and an existing
    # binding wins: approving through the workspace must reach the cron that runs these files.
    scope = for_models(store, static['path_model'])
    bound = bound_case(store, scope['case_id'])
    return scope_of(store, bound) if bound else scope


def for_cli(store, name):
    """The case a maintainer means on the command line, or a refusal naming what was tried.

    Two failures this closes. A typo must not mint a phantom case: `eforecast approve pv-park`
    would otherwise report an approval while the forecast of `pv_park` still has no model,
    because the identity it approved belongs to a case that does not exist. And a machine with
    no workspace -- the legacy cron host, a second workstation -- must still be able to name
    its own case, which it does by the model directory it serves.
    """
    from .service import ServiceError, Workspace
    path = Path(name).expanduser()
    if path.exists():                                   # a path is unambiguous; try it first
        if path.is_dir() and not any(path.glob('*.yaml')):
            return for_models(store, path)              # the model directory itself
        from .studies import _static
        from .config_files import load_document
        return for_config(store, str(path), _static(load_document(path), str(path)))
    try:
        known = Workspace(admin=True).exists(name)
    except ServiceError:                                # not even a well-formed case name
        known = False
    if known:
        return for_case(store, Workspace(admin=True), name)
    raise ServiceError(
        f'no case {name!r} in this workspace. Name a case of this installation, or the model '
        f'directory the forecasts serve (the folder holding scalers.pickle and Results).')


def read_keys(store, case, cursor):
    """Scoped first, then an explicitly bound legacy namespace. No name fallback."""
    store._execute(cursor, '''SELECT legacy_key FROM eforecast_serving_bindings
        WHERE namespace=? AND case_id=?''', (store.namespace, case))
    row = cursor.fetchone()
    return [case] + ([row[0]] if row and row[0] != case else [])


def bind_legacy(store, legacy_key, case):
    """Add a one-to-one reading alias; preserve every original approval/issue/pin."""
    if not legacy_key or not case or legacy_key == case:
        raise ValueError('bind a named legacy key to a different durable case identity')
    with store._transaction() as cursor:
        store._execute(cursor, '''SELECT legacy_key,case_id FROM eforecast_serving_bindings
            WHERE namespace=? AND (legacy_key=? OR case_id=?)''',
            (store.namespace, legacy_key, case))
        rows = cursor.fetchall()
        if rows and rows != [(legacy_key, case)]:
            raise ValueError('legacy name or case already has a different binding')
        store._execute(cursor, '''INSERT INTO eforecast_serving_bindings
            (namespace,legacy_key,case_id) VALUES (?,?,?)
            ON CONFLICT (namespace,legacy_key) DO NOTHING''', (store.namespace, legacy_key, case))
        store._execute(cursor, '''SELECT case_id FROM eforecast_serving_bindings
            WHERE namespace=? AND legacy_key=?''', (store.namespace, legacy_key))
        if cursor.fetchone()[0] != case:
            raise ValueError('legacy name already has a different binding')
