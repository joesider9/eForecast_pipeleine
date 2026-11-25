# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Independent bounded reconciliation. Run with ``python -m eforecast.config.run_coordinator``.

Control claims use A08's allocator contract in a separate logical namespace. They reserve
no training resources; actual workers retain their existing A08 resource reservations.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from eforecast.config.executors import LocalExecutor, location, same_location
from eforecast.config.run_model import Outcome, identity, pinned
from eforecast.config.run_store import PostgresRunStore, RunConflict, StoreUnavailable, configured_store
from eforecast.config.runs import RunManager
from eforecast.config.service import Workspace
from eforecast.training.command_line import liveness
from eforecast.training.command_line.allocation_contract import Capacity, Refusal, Request, Resources
from eforecast.training.command_line.allocator_file import FileAllocator
from eforecast.training.command_line.liveness import process_identity

log = logging.getLogger(__name__)


class ClaimLost(RuntimeError):
    pass


def require(value):
    if isinstance(value, Refusal):
        raise ClaimLost(f'controller allocation {value.code}: {value.reason}')
    return value


class GuardedStore:
    """Only short database publications run inside the A08 ownership guard."""
    def __init__(self, store, allocator, allocation, controller_json):
        self.raw, self.allocator, self.allocation = store, allocator, allocation
        self.controller_json = controller_json

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def _guard(self, run, callback):
        if run.controller_json != self.controller_json:
            raise ClaimLost('run belongs to another coordinator claim')
        return require(self.allocator.guard(self.allocation.token, callback))

    def compare_and_set(self, run, expected_revision):
        return self._guard(run, lambda: self.raw.compare_and_set(run, expected_revision))

    def collect_reports(self, run, events, offset):
        return self._guard(run, lambda: self.raw.collect_reports(run, events, offset))

    def collect_cleanup(self, run, updates):
        return self._guard(run, lambda: self.raw.collect_cleanup(run, updates))

    def check_ownership(self):
        return require(self.allocator.validate(self.allocation.token))


def control_allocator(store, directory, *, timeout=1):
    where = location()
    capacity = Capacity(where['host'], 0, 0, {})
    if isinstance(store, PostgresRunStore):
        from eforecast.training.command_line.allocator_postgres import PostgresAllocator
        return PostgresAllocator(store._dsn, capacity, namespace='a09-controller:' + store.namespace,
                                 boot_id=where['boot_id'], timeout=timeout)
    key = hashlib.sha256((str(store.path) + '|' + store.namespace).encode()).hexdigest()
    return FileAllocator(Path(directory) / 'controllers' / key, capacity, boot_id=where['boot_id'], timeout=timeout)


class Coordinator:
    def __init__(self, workspace=None, *, store=None, executor=None, allocator=None,
                 workers=2, batch=32, max_owned=64, lease_seconds=15):
        if not (1 <= workers <= 16 and 1 <= batch <= 256 and workers <= max_owned <= 256
                and 1 <= lease_seconds <= 3600):
            raise ValueError('invalid bounded coordinator settings')
        self.workspace = workspace or Workspace(admin=True)
        self.store = store if store is not None else configured_store(self.workspace.root)
        self.executor = executor if executor is not None else LocalExecutor(workspace=self.workspace.root)
        self.allocator = allocator if allocator is not None else control_allocator(self.store, self.executor.directory)
        here = location()
        self.owner = process_identity(os.getpid(), here['host'], here['boot_id'])
        if os.environ.get('EFORECAST_KUBE_RUN_ID'):
            self.owner = replace(self.owner, container_id=os.environ['EFORECAST_POD_UID'])
        self.workers, self.batch, self.max_owned, self.lease = workers, batch, max_owned, lease_seconds
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='eforecast-reconcile')
        self.owned = OrderedDict()
        self.requests, self.jobs, self.backoff = {}, {}, OrderedDict()
        self.references = {}
        self.cursor = None
        self.stopping = threading.Event()
        self.storage_census_due = 0
        self.cache_census = None
        self.kubernetes_bridge = None
        self.kubernetes_tick = None
        self.kubernetes_due = 0
        if not os.environ.get('EFORECAST_KUBE_RUN_ID'):
            from .kubernetes_runs import configured_target, Bridge
            target = configured_target()
            if target is not None:
                self.kubernetes_bridge = Bridge(self.store, self.workspace, target=target)

    def _eligible(self, run):
        from .kubernetes_runs import is_broker, in_assigned_pod
        assigned = os.environ.get('EFORECAST_KUBE_RUN_ID')
        if assigned and run.id != assigned:
            return False
        if is_broker(run) and not in_assigned_pod(self.store, run):
            return False
        request = json.loads(run.request_json)
        if request.get('workspace') != str(self.workspace.root.resolve()) or run.deleted:
            return False
        if self.backoff.get(run.id, 0) > time.monotonic():
            return False
        if run.id not in self.requests and len(self.requests) >= 256:
            return False
        if run.tasks and run.tasks[0].attempts and run.tasks[0].attempts[-1].executor:
            handle = run.tasks[0].attempts[-1].executor
            if handle.adapter != self.executor.adapter:
                return False
            intent = json.loads(handle.identity_json)
            if intent.get('location') and not same_location(intent):
                return False  # another host must reconcile its own worker; do not overwrite running
        return bool(request.get('entrypoint'))

    def _claim(self, run):
        request = self.requests.setdefault(run.id, Request(run.id, 'a09/coordinator/' + run.id,
            identity(), self.owner.host, Resources(), owner=self.owner, purpose='controller',
            lease_seconds=self.lease, spawn_seconds=self.lease))
        entry = self.allocator.acquire(request)
        if isinstance(entry, Refusal):
            # A heartbeat expiring is not death. A08 eligibility plus verified namespace/
            # birth liveness decide recovery; no model-worker allocation is touched here.
            snapshot = require(self.allocator.snapshot())
            for previous in snapshot:
                if previous.request.task_id == request.task_id:
                    if previous.request == request:
                        entry = require(self.allocator.renew(previous.token))
                        break
                    require(self.allocator.reconcile(liveness.probe(), token=previous.token))
            if isinstance(entry, Refusal):
                entry = self.allocator.acquire(request)
        if isinstance(entry, Refusal):
            if entry.code not in {'busy', 'unavailable'}:
                self.requests.pop(run.id, None)
            return False
        attached = self.allocator.attach(entry.token, self.owner)
        if isinstance(attached, Refusal) and attached.code == 'spawn_expired':
            # This zero-resource controller never attached. Relinquish its exact token,
            # rather than keeping an unattachable self-owned intent forever.
            require(self.allocator.release(entry.token))
            self.requests.pop(run.id, None)
            return False
        entry = require(attached)
        controller = pinned({'token': asdict(entry.token), 'owner': asdict(self.owner)})
        try:
            latest = self.store.get(run.owner_id, run.case_id, run.id)
            require(self.allocator.guard(entry.token, lambda: self.store.compare_and_set(
                replace(latest, controller_json=controller), latest.revision)))
        except StoreUnavailable:
            # If a commit reply was lost, the next tick uses this exact accepted ownership.
            latest = self.store.get(run.owner_id, run.case_id, run.id)
            if latest.controller_json != controller:
                raise
        except BaseException:
            self.allocator.release(entry.token)
            self.requests.pop(run.id, None)
            raise
        self.owned[run.id] = (entry, controller, time.monotonic() + self.lease / 3)
        return True

    def _release(self, run_id, *, delay=0):
        current = self.owned.pop(run_id, None)
        if current:
            result = self.allocator.release(current[0].token)
            if isinstance(result, Refusal):
                log.warning('controller release retained for %s: %s', run_id, result.reason)
        self.requests.pop(run_id, None)
        self.references.pop(run_id, None)
        if delay:
            self.backoff[run_id] = time.monotonic() + delay
            while len(self.backoff) > 256:
                self.backoff.popitem(last=False)

    def _work(self, entry, controller):
        store = GuardedStore(self.store, self.allocator, entry, controller)
        manager = RunManager(self.workspace, store=store, executor=self.executor)
        try:
            # Identities are indexed with the claim's current run, not a caller's case name.
            owner, case_id = self.references[entry.request.run_id]
            run = store.get(owner, case_id, entry.request.run_id)
            if run.controller_json != controller:
                raise ClaimLost('coordinator was replaced')
            case = json.loads(run.metadata_json)['case']
            from eforecast.config import run_cleanup
            if run.cancel is not None or (run.state == 'terminal' and run.cleanup not in {'complete', 'not_required'}):
                return run_cleanup.step(manager, run)
            if run.state != 'terminal':
                manager.prepare(case, run.id)
            elif manager._executor_handle(run) is not None:
                manager._observe_executor(run)
            run = store.get(owner, case_id, run.id)
            if run.state == 'terminal' and run.cleanup not in {'complete', 'not_required'}:
                return run_cleanup.step(manager, run)
            return run
        finally:
            manager._preparer.shutdown(wait=False)

    def tick(self):
        """One bounded scan and dispatch pass. Main thread renews while workers prepare/collect."""
        if self.kubernetes_bridge and time.monotonic() >= self.kubernetes_due:
            if self.kubernetes_tick is None or self.kubernetes_tick.done():
                if self.kubernetes_tick is not None:
                    try:
                        self.kubernetes_tick.result()
                    except Exception:
                        log.exception('Kubernetes handoff unavailable; durable intent retained')
                self.kubernetes_tick = self.pool.submit(self.kubernetes_bridge.tick)
                self.kubernetes_due = time.monotonic() + 1
        for key, proc in list(getattr(self.executor, 'processes', {}).items()):
            if proc.poll() is not None:
                self.executor.processes.pop(key, None)
        for run_id, job in list(self.jobs.items()):
            if not job.done():
                continue
            self.jobs.pop(run_id)
            try:
                run = job.result()
                if run.cleanup in {'pending', 'running'} and not run.report_error:
                    continue  # retain/renew the control claim while asynchronous cleanup retries
                if run.cleanup == 'unknown':
                    self._release(run_id, delay=2)
                    self.references.pop(run_id, None)
                    continue
                if run.state == 'terminal' or run.state == 'unknown' or run.report_error:
                    delay = 15 if run.cleanup in {'pending', 'running', 'unknown'} or run.state == 'unknown' or run.report_error else 2
                    self._release(run_id, delay=delay)
                    self.references.pop(run_id, None)
            except (RunConflict, ClaimLost):
                self._release(run_id, delay=1)
            except Exception:
                log.exception('reconciliation failed for %s; durable work retained', run_id)
                self._release(run_id, delay=2)
        for run_id, (entry, controller, due) in list(self.owned.items()):
            if time.monotonic() >= due:
                try:
                    renewed = require(self.allocator.renew(entry.token))
                    self.owned[run_id] = (renewed, controller, time.monotonic() + self.lease / 3)
                except ClaimLost:
                    self._release(run_id, delay=2)
        page = self.store.active_page(after=self.cursor, limit=self.batch)
        self.cursor = (page[-1].created_at, page[-1].id) if page else None
        for run in page:
            if len(self.owned) >= self.max_owned:
                break
            if run.id not in self.owned and run.id not in self.jobs and self._eligible(run) and self._claim(run):
                self.references[run.id] = (run.owner_id, run.case_id)
        for run_id in list(self.owned):
            if len(self.jobs) >= self.workers:
                break
            if run_id not in self.jobs:
                entry, controller, _ = self.owned[run_id]
                self.jobs[run_id] = self.pool.submit(self._work, entry, controller)
                self.owned.move_to_end(run_id)

    def run(self, interval=1):
        while not self.stopping.is_set():
            try:
                self.tick()
            except Exception:
                log.exception('coordinator scan unavailable; no durable work forgotten')
            self.stopping.wait(interval)

    def close(self):
        self.stopping.set()
        self.pool.shutdown(wait=True)
        for run_id in list(self.owned):
            self._release(run_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--max-owned', type=int, default=64)
    parser.add_argument('--lease-seconds', type=float, default=15)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--service-lock-fd', type=int)
    args = parser.parse_args(argv)
    if not .05 <= args.interval <= 60:
        parser.error('interval must be between .05 and 60 seconds')
    logging.basicConfig(level=logging.INFO)
    if args.service_lock_fd is not None:
        os.fstat(args.service_lock_fd)  # serve's inherited singleton lock stays held for this process
    if not (1 <= args.workers <= 16 and 1 <= args.batch <= 256 and args.workers <= args.max_owned <= 256
            and 1 <= args.lease_seconds <= 3600):
        parser.error('invalid bounded coordinator settings')
    stopping = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopping.set())
    while not stopping.is_set():
        try:
            coordinator = Coordinator(Workspace(root=args.workspace, admin=True) if args.workspace else None,
                workers=args.workers, batch=args.batch, max_owned=args.max_owned, lease_seconds=args.lease_seconds)
        except Exception:
            log.exception('coordinator initialization unavailable; retrying without changing accepted work')
            stopping.wait(2)
            continue
        coordinator.stopping = stopping
        try:
            coordinator.run(args.interval)
        finally:
            coordinator.close()


def ensure_service(workspace):
    """Start one detached local service, or discover its inherited lifetime lock.

    The normal serve launcher uses this; external deployments run the CLI under their
    service manager. API restart/shutdown never signals this service or its workers.
    """
    from eforecast.config.executors import locked
    executor = LocalExecutor(workspace=workspace.root)
    # A service per local store/workspace, with no owner case in its private directory.
    marker = pinned({'workspace': str(workspace.root.resolve()),
                    'store': os.environ.get('EFORECAST_RUN_STORE_PATH'),
                    'dsn': os.environ.get('EFORECAST_RUN_STORE_DSN') or os.environ.get('EFORECAST_ALLOCATOR_DSN'),
                    'namespace': os.environ.get('EFORECAST_RUN_STORE_NAMESPACE', 'default')})
    key = hashlib.sha256(marker.encode()).hexdigest()
    folder = executor.directory / 'services' / key
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with locked(folder) as descriptor:
        if descriptor is None:
            return None
        log_fd = os.open(folder / 'coordinator.log', os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        with os.fdopen(log_fd, 'ab') as stream:
            proc = subprocess.Popen([sys.executable, '-m', 'eforecast.config.run_coordinator',
                '--workspace', str(workspace.root.resolve()), '--service-lock-fd', str(descriptor)],
                cwd=str(Path(__file__).resolve().parents[2]), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(descriptor,))
    return proc


if __name__ == '__main__':
    main()
