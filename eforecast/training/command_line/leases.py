# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Who owns a RUNNING trial, and for how long.

A trial is RUNNING in its study from the moment it is asked until something tells the study how it
ended.  Nothing recorded *who* was running it, so every restart had to assume that no other worker
of the study was alive: ``settle_running`` failed every RUNNING trial it found, which on a shared
storage is one machine failing another machine's work in flight.  The runbook said as much and
asked the owner not to restart two machines at once.

A lease is three user attributes on the trial:

``lease_owner``      the ``run_id`` of the orchestrator that asked for it
``lease_attempt``    which attempt of that trial the owner is on, so a late result of an older
                     attempt is recognisable
``lease_heartbeat``  a unix timestamp the owner refreshes while it is alive

The lease is **held** while the heartbeat is younger than ``EFORECAST_LEASE_SECONDS`` (600 by
default, ten refresh intervals).  A run settles a trial only when the lease is its own, expired, or
absent -- a trial of a live foreign owner is left exactly as it is.  Nothing here is a lock: a
heartbeat that stops is an invitation to recover the work, not a guarantee that the worker is dead,
so recovery is bounded by the expiry and never by a guess about another machine.
"""
import collections
import logging
import os
import time

log = logging.getLogger(__name__)

LEASE_SECONDS = int(os.environ.get('EFORECAST_LEASE_SECONDS', 600))
REFRESH_SECONDS = int(os.environ.get('EFORECAST_LEASE_REFRESH_SECONDS', 60))
OWNER_KEY, ATTEMPT_KEY, HEARTBEAT_KEY = 'lease_owner', 'lease_attempt', 'lease_heartbeat'

Lease = collections.namedtuple('Lease', 'owner attempt heartbeat age held')


def read(user_attrs, now=None, lease_seconds=LEASE_SECONDS):
    """The lease recorded in a trial's user attributes; ``held`` is False when there is none."""
    now = time.time() if now is None else now
    owner = user_attrs.get(OWNER_KEY)
    heartbeat = user_attrs.get(HEARTBEAT_KEY)
    try:
        heartbeat = float(heartbeat) if heartbeat is not None else None
    except (TypeError, ValueError):
        heartbeat = None
    age = None if heartbeat is None else max(0.0, now - heartbeat)
    held = bool(owner) and age is not None and age < lease_seconds
    return Lease(owner, user_attrs.get(ATTEMPT_KEY), heartbeat, age, held)


def claim(trial, run_id, attempt=0, now=None):
    """Take the lease of a trial this run is about to work on."""
    trial.set_user_attr(OWNER_KEY, run_id)
    trial.set_user_attr(ATTEMPT_KEY, int(attempt))
    trial.set_user_attr(HEARTBEAT_KEY, time.time() if now is None else now)


def beat(trial, run_id, attempt=0, now=None, refresh_seconds=REFRESH_SECONDS):
    """Refresh the lease if it is due; returns True when it was written.

    Due rather than every tick, because every write appends to the journal file of the study.
    """
    now = time.time() if now is None else now
    current = read(getattr(trial, 'user_attrs', {}) or {}, now=now)
    if current.owner == run_id and current.age is not None and current.age < refresh_seconds:
        return False
    claim(trial, run_id, attempt=attempt, now=now)
    return True


def recoverable(user_attrs, run_id, now=None, lease_seconds=LEASE_SECONDS):
    """May this run settle the trial these attributes belong to?

    Its own work and unowned or expired work, yes; a lease another run is still refreshing, no.
    """
    lease = read(user_attrs, now=now, lease_seconds=lease_seconds)
    return (not lease.held) or lease.owner == run_id, lease


def describe(trial_number, lease):
    return (f'trial {trial_number} is held by run {lease.owner} '
            f'(heartbeat {lease.age:.0f} s ago, attempt {lease.attempt})')


class Beating:
    """Refresh a lease from a daemon thread, for a worker that has no loop of its own.

    The orchestrator beats from its own tick; a CPU trial runs to completion inside
    ``study.optimize``, so without this a trial longer than the expiry would look abandoned to
    another machine while it is still working.
    """

    def __init__(self, trial, run_id, attempt=0, refresh_seconds=REFRESH_SECONDS):
        import threading
        self.trial, self.run_id, self.attempt = trial, run_id, attempt
        self.refresh_seconds = max(1, int(refresh_seconds))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name='lease-heartbeat', daemon=True)

    def _loop(self):
        while not self._stop.wait(self.refresh_seconds):
            try:
                claim(self.trial, self.run_id, attempt=self.attempt)
            except Exception as error:  # noqa: BLE001 -- a storage hiccup must not end the trial
                log.debug('cannot refresh the lease (%r)', error)

    def __enter__(self):
        claim(self.trial, self.run_id, attempt=self.attempt)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=self.refresh_seconds)
        return False

