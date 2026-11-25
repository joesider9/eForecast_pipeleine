# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A cumulative count of completions that were accepted with a verified receipt.

A16-06 declared `eforecast_completions_verified_total` and its dashboard panel queried it,
while nothing produced it: a count over retained runs is not a cumulative counter, because
history may be deleted and a redelivery may be observed twice (R27-15).

Three properties, which is why this is a table and not a query:

* **once per acceptance.** The identity of the acceptance is the key, so a redelivered
  completion, a retried collection or a replayed message adds nothing.
* **unaffected by history.** The totals are their own rows; deleting runs, pruning events or
  reconciling a stage cannot change them.
* **durable across restarts.** The increment happens in the caller's transaction, so it commits
  with the acceptance it counts or not at all.
"""
from __future__ import annotations

from contextlib import nullcontext

SCHEMA = (
    '''CREATE TABLE IF NOT EXISTS eforecast_verified_completions (
       namespace TEXT NOT NULL, runtime TEXT NOT NULL, total INTEGER NOT NULL,
       PRIMARY KEY (namespace, runtime))''',
    '''CREATE TABLE IF NOT EXISTS eforecast_verified_completion_marks (
       namespace TEXT NOT NULL, acceptance TEXT NOT NULL, runtime TEXT NOT NULL,
       observed_at REAL NOT NULL, PRIMARY KEY (namespace, acceptance))''',
)


def _in(store, cursor):
    return nullcontext(cursor) if cursor is not None else store._transaction()


def record(store, runtime, acceptance, *, cursor=None, now=None):
    """Count one verified acceptance. Returns whether this call was the one that counted it.

    Pass the cursor of the transaction that accepted the work: the count then commits with the
    acceptance, and a rolled-back acceptance leaves no count behind.
    """
    import time
    if not runtime or not acceptance:
        raise ValueError('a verified completion needs a runtime and an acceptance identity')
    moment = time.time() if now is None else now
    with _in(store, cursor) as handle:
        store._execute(handle, '''INSERT INTO eforecast_verified_completion_marks
            (namespace,acceptance,runtime,observed_at) VALUES (?,?,?,?)
            ON CONFLICT (namespace,acceptance) DO NOTHING''',
            (store.namespace, str(acceptance), str(runtime), float(moment)))
        if handle.rowcount != 1:
            return False                                # already counted; nothing to add
        store._execute(handle, '''INSERT INTO eforecast_verified_completions
            (namespace,runtime,total) VALUES (?,?,1)
            ON CONFLICT (namespace,runtime) DO UPDATE SET total=eforecast_verified_completions.total+1''',
            (store.namespace, str(runtime)))
    return True


def totals(store, *, cursor=None):
    """``{runtime: total}``, the monotonic counts this installation has accepted."""
    with _in(store, cursor) as handle:
        store._execute(handle, '''SELECT runtime,total FROM eforecast_verified_completions
            WHERE namespace=? ORDER BY runtime''', (store.namespace,))
        return {runtime: int(total) for runtime, total in handle.fetchall()}
