# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Local-disk allocator implementation. This API does not spawn or kill processes.

One authoritative allocations.json is committed under the host lock. Legacy claim files are
not its ledger. launch_allocations connects projections and workers; cross-run recovery is separate.
Leases use monotonic host time plus boot identity; expiry alone never releases resources.
"""
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import logging
import os
import re
from pathlib import Path
import time
import uuid

from .allocation_contract import (Allocation, Capacity, Device, GpuCapacity, ProcessIdentity, Reconciled,
                                  Refusal, Released, Request, Resources, Settled, Token, name, number)

log = logging.getLogger(__name__)


def _grace_seconds():
    """``None`` means "as long again as the request's own spawn deadline"."""
    raw = os.environ.get('EFORECAST_SPAWN_GRACE_SECONDS')
    if raw is None:
        return None
    try:
        return max(0., float(raw))
    except ValueError:
        return None
STATE_FILE = 'allocations.json'
SEAL_FILE = '.allocator-initialized'
VERSION = 1


class StateUnavailable(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def request_from(data):
    value = dict(data)
    value['resources'] = Resources(**value['resources'])
    value['device'] = Device(**value['device']) if value.get('device') is not None else None
    value['owner'] = ProcessIdentity(**value['owner']) if value.get('owner') is not None else None
    return Request(**value)


def allocation_from(data):
    value = dict(data)
    value['token'] = Token(**value['token'])
    value['request'] = request_from(value['request'])
    value['process'] = ProcessIdentity(**value['process']) if value.get('process') is not None else None
    result = Allocation(**value)
    for key in ('created_at', 'renewed_at', 'expires_at', 'spawn_deadline'):
        number(getattr(result, key), key)
    number(result.revision, 'allocation revision', integer=True)
    if result.status not in ('reserved', 'running', 'expired_held'):
        raise ValueError('unknown allocation state')
    if result.process and result.process.host != result.request.host:
        raise ValueError('process and allocation hosts differ')
    return result


class FileAllocator:
    """The contract for one physical host; directory must be on local, unsynchronized storage.

    Capacity is a consistent host budget shared by every client. GPU placement, estimates and
    unmanaged-usage headroom remain caller policy. clock/boot_id are injectable for contract
    tests; live callers use the allocator's own clock and kernel boot identity.
    """

    def __init__(self, directory, capacity: Capacity, *, timeout=None, clock=None, boot_id=None,
                 discover_devices=False, spawn_grace=None):
        self.directory = Path(directory).resolve()
        # How long after its own spawn deadline a reservation that never became a process may be
        # reclaimed by anyone. Past the deadline its owner can no longer attach anything to it --
        # ``_owned`` refuses -- so the launch has already failed; the grace only keeps a run from
        # reclaiming a reservation whose controller is still in the middle of a slow start.
        self.spawn_grace = spawn_grace if spawn_grace is not None else _grace_seconds()
        if self.spawn_grace is not None:
            number(self.spawn_grace, 'spawn grace')
        self.capacity = deepcopy(capacity)
        self.timeout = timeout
        if timeout is not None:
            number(timeout, 'lock timeout')
        self.clock = clock or time.monotonic
        self.boot_id = boot_id or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.discover_devices = discover_devices
        self.path = self.directory / STATE_FILE
        self.seal = self.directory / SEAL_FILE

    def _now(self):
        value = self.clock()
        number(value, 'allocator clock')
        return value

    def _atomic_write(self, path, payload):
        temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        try:
            with temporary.open('xb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            with suppress(OSError):
                temporary.unlink()

    # ---------------------------------------------------------------- storage
    # Everything below the next three methods is the ledger's logic and knows nothing about where
    # the state lives. A second store -- PostgreSQL, for the runs of several machines -- overrides
    # exactly these three and inherits the rest, so both adapters answer one contract.
    def _read_wrapper(self):
        """The stored ``{version, state, sha256}``, or ``None`` when nothing was ever written."""
        return json.loads(self.path.read_bytes()) if self.path.exists() else None

    def _write_wrapper(self, payload, initial=False):
        self._atomic_write(self.path, payload)
        if initial:
            self._atomic_write(self.seal, str(VERSION).encode())

    def _sealed(self):
        """Whether this ledger was initialized; a missing state afterwards is not an empty one."""
        return self.seal.exists() and self.seal.read_text() == str(VERSION)

    def _seal(self):
        self._atomic_write(self.seal, str(VERSION).encode())

    def _save(self, state):
        payload = canonical(state)
        self._write_wrapper(canonical({'version': VERSION, 'state': state,
                                       'sha256': hashlib.sha256(payload).hexdigest()}))

    def _load(self):
        wrapper = self._read_wrapper()
        if wrapper is None:
            if self._sealed():
                raise StateUnavailable('allocation state disappeared after initialization; reconciliation required')
            state = {'capacities': {self.capacity.host: asdict(self.capacity)},
                     'boots': {self.capacity.host: self.boot_id},
                     'allocations': {}, 'fences': {}, 'attempts': {}, 'receipts': {}}
            payload = canonical(state)
            self._write_wrapper(canonical({'version': VERSION, 'state': state,
                                           'sha256': hashlib.sha256(payload).hexdigest()}), initial=True)
            return state
        if type(wrapper.get('version')) is not int or wrapper['version'] != VERSION:
            raise StateUnavailable('unknown allocation state version')
        state = wrapper['state']
        if hashlib.sha256(canonical(state)).hexdigest() != wrapper.get('sha256'):
            raise StateUnavailable('allocation state digest mismatch')
        if 'capacity' in state and 'capacities' not in state:
            # Written before one ledger could hold several hosts.
            state['capacities'] = {state['capacity']['host']: state.pop('capacity')}
        if not isinstance(state.get('capacities'), dict) or not state['capacities']:
            raise StateUnavailable('capacities must be a nonempty mapping of host budgets')
        current = asdict(self.capacity)
        previous = state['capacities'].get(current['host'])
        if previous is None:
            # Another machine of the same deployment, registering its own budget. It never changes
            # anyone else's, and it is written only after the rest of the state has been validated.
            registered_capacity = current
        elif previous != current:
            # Schedulers may see different subsets of the same host's cards. Registering another
            # UUID must not reset the budget or require every CUDA-visible ordinal to be identical.
            if (not self.discover_devices
                    or any(previous[key] != current[key] for key in ('host', 'cpu_threads', 'ram_mb'))
                    or any(value != previous['devices'][key] for key, value in current['devices'].items()
                           if key in previous['devices'])):
                raise StateUnavailable('host capacity differs from the initialized allocator budget')
            merged = {**previous['devices'], **current['devices']}
            self.capacity = Capacity(current['host'], current['cpu_threads'], current['ram_mb'],
                                     {key: GpuCapacity(**value) for key, value in merged.items()})
            registered_capacity = asdict(self.capacity)
        else:
            registered_capacity = None
        if 'boot_id' in state and 'boots' not in state:          # the one-host shape
            state['boots'] = {next(iter(state['capacities'])): state.pop('boot_id')}
        if not isinstance(state.get('boots'), dict):
            raise StateUnavailable('boots must be a mapping of host kernel identities')
        # A kernel identity belongs to one machine. Another host's boot is not this host's
        # business; this host's own, if it changed, means everything it had recorded is stale.
        mine = state['boots'].get(self.capacity.host)
        if mine is None:
            registered_boot = self.boot_id
        elif mine != self.boot_id:
            raise StateUnavailable('allocation state belongs to another boot; reconciliation required')
        else:
            registered_boot = None
        for key in ('allocations', 'fences', 'attempts', 'receipts'):
            if not isinstance(state[key], dict):
                raise StateUnavailable(f'{key} must be a mapping')
        for fence in state['fences'].values():
            number(fence, 'persisted fence', positive=True, integer=True)
        for key, value in state['allocations'].items():
            entry = allocation_from(value)
            if (entry.token.allocation_id != key or entry.request.host not in state['capacities']
                    or entry.token.fence > state['fences'].get(entry.request.task_id, 0)
                    or key not in state['receipts']):
                raise StateUnavailable('inconsistent allocation identity')
            state['allocations'][key] = asdict(entry)  # optional fields in older records normalize on read
        for key, receipt in state['receipts'].items():
            token = Token(**receipt['token'])
            request = request_from(receipt['request'])
            if (token.allocation_id != key or type(receipt['closed']) is not bool
                    or receipt['closed'] != (key not in state['allocations'])
                    or state['attempts'].get(self._attempt(request)) != key
                    or token.fence > state['fences'].get(request.task_id, 0)):
                raise StateUnavailable('inconsistent allocation receipt')
            if key in state['allocations']:
                entry = allocation_from(state['allocations'][key])
                # Resize may change only the committed resource quantities. It must never
                # change the token or another part of the original acquisition identity.
                if (entry.token != token
                        or replace(entry.request, resources=request.resources) != request):
                    raise StateUnavailable('allocation differs from its acquisition identity')
            completed = receipt.get('completion')
            if completed is not None:
                if (not isinstance(completed, dict) or completed.get('status') not in ('accepted', 'done')
                        or not isinstance(completed.get('result'), dict)
                        or not isinstance(completed.get('publication_id'), str)
                        or not re.fullmatch('[0-9a-f]{64}', completed['publication_id'])):
                    raise StateUnavailable('invalid completion receipt')
                number(completed['accepted_at'], 'completion acceptance time')
                if completed['status'] == 'done':
                    number(completed['completed_at'], 'completion settlement time')
        if any(value not in state['receipts'] for value in state['attempts'].values()):
            raise StateUnavailable('attempt index references an unknown allocation')
        if not self._sealed():
            if any(state[key] for key in ('allocations', 'fences', 'attempts', 'receipts')):
                raise StateUnavailable('initialized allocator seal is missing')
            self._seal()
        if registered_boot is not None:
            state['boots'][self.capacity.host] = registered_boot
        if registered_capacity is not None and registered_capacity != state['capacities'].get(
                registered_capacity['host']):
            state['capacities'][registered_capacity['host']] = registered_capacity
            self._save(state)
        elif registered_boot is not None:
            self._save(state)
        return state

    def _transaction(self, operation):
        from .allocator import AllocatorBusy, file_critical_section
        try:
            with file_critical_section(self.directory, self.timeout):
                state = self._load()
                before = canonical(state)
                result = operation(state, self._now())
                if before != canonical(state):
                    self._save(state)
                return result
        except AllocatorBusy as error:
            return Refusal('busy', str(error))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, StateUnavailable) as error:
            log.error('allocation state unavailable at %s: %s', self.directory, error)
            return Refusal('unavailable', str(error))

    @staticmethod
    def _attempt(request):
        return json.dumps([request.run_id, request.task_id, request.attempt_id], separators=(',', ':'))

    def _fit(self, state, request, exclude=None):
        if request.host != self.capacity.host:
            return Refusal('host', 'request does not belong to this allocator host')
        # Only this host's reservations count against this host's budget: a shared ledger holds
        # the work of every machine in the deployment, and a card on one is not a card on another.
        entries = [entry for entry in (allocation_from(value) for key, value in state['allocations'].items()
                                       if key != exclude)
                   if entry.request.host == request.host]
        resources = request.resources
        if resources.cpu_threads + sum(a.request.resources.cpu_threads for a in entries) > self.capacity.cpu_threads:
            return Refusal('cpu', 'CPU thread budget would be exceeded')
        if resources.ram_mb + sum(a.request.resources.ram_mb for a in entries) > self.capacity.ram_mb:
            return Refusal('ram', 'host RAM budget would be exceeded')
        if request.device is None:
            return None
        capacity = self.capacity.devices.get(request.device.uuid)
        if capacity is None:
            return Refusal('device', 'requested GPU UUID is not in this host budget')
        peers = [a.request.resources for a in entries if a.request.device == request.device]
        if peers and (resources.exclusive or any(peer.exclusive for peer in peers)):
            return Refusal('exclusive', 'GPU is held exclusively or the request needs it alone')
        limit = min([capacity.slots, *[r.gpu_slot_limit for r in [resources, *peers]
                                      if r.gpu_slot_limit is not None]])
        if resources.gpu_slots + sum(peer.gpu_slots for peer in peers) > limit:
            return Refusal('slots', 'GPU slot limit would be exceeded')
        if resources.gpu_mb + sum(peer.gpu_mb for peer in peers) > capacity.memory_mb:
            return Refusal('vram', 'GPU memory budget would be exceeded')
        return None

    def _owned(self, state, token, now, *, allow_expired_control=False):
        receipt = state['receipts'].get(token.allocation_id)
        if not receipt or receipt['token'] != asdict(token):
            return Refusal('stale', 'allocation id and fence do not match')
        task = receipt['request']['task_id']
        if state['fences'].get(task) != token.fence:
            return Refusal('stale', 'this task has a newer owner')
        value = state['allocations'].get(token.allocation_id)
        if value is None:
            return Refusal('closed', 'allocation is already closed')
        result = allocation_from(value)
        if result.expires_at <= now and not (allow_expired_control and result.request.purpose == 'controller'):
            return Refusal('expired', 'lease expired; resources remain held until reconciliation')
        if result.process is None and result.spawn_deadline <= now and not (
                allow_expired_control and result.request.purpose == 'controller'):
            return Refusal('spawn_expired', 'spawn deadline expired; reconciliation required')
        return result

    def _settling(self, state, task_id):
        """The reason a task is held by an unfinished settlement of its current owner, or ``None``."""
        fence = state['fences'].get(task_id)
        for key, receipt in state['receipts'].items():
            intent = receipt.get('settlement')
            completion = receipt.get('completion')
            if (completion and completion.get('status') == 'accepted' and receipt['request']['task_id'] == task_id
                    and receipt['token'].get('fence') == fence):
                return 'accepted publication awaits study settlement'
            if (intent and intent.get('status') == 'accepted' and receipt['request']['task_id'] == task_id
                    and receipt['token'].get('fence') == fence):
                return intent.get('reason')
        return None

    def _reclaim_at(self, entry):
        """When an unattached reservation may be reclaimed by anyone on this host."""
        grace = entry.request.spawn_seconds if self.spawn_grace is None else self.spawn_grace
        return entry.spawn_deadline + grace

    def pending_settlements(self):
        """Accepted study operations survive release of their worker's resource reservation."""
        def operation(state, now):
            return tuple(deepcopy({'token': receipt['token'], 'request': receipt['request'],
                                   'settlement': receipt['settlement']})
                         for receipt in state['receipts'].values()
                         if (receipt.get('settlement') or {}).get('status') == 'accepted')
        return self._transaction(operation)

    def pending_completions(self):
        return self._transaction(lambda state, now: tuple(deepcopy({'token': receipt['token'],
            'request': receipt['request'], 'completion': receipt['completion']})
            for receipt in state['receipts'].values()
            if (receipt.get('completion') or {}).get('status') == 'accepted'))

    def settlement(self, token):
        def operation(state, now):
            receipt = state['receipts'].get(token.allocation_id)
            if not receipt or receipt['token'] != asdict(token):
                return Refusal('stale', 'allocation id and fence do not match')
            return deepcopy(receipt.get('settlement'))
        return self._transaction(operation)

    def accept_settlement(self, token, reason, *, target=None):
        """Record, under ownership, that this attempt will not finish. The task is held meanwhile.

        Accepted completed work is never turned into a failure. Repeating the intent returns the
        same record, so a lost acknowledgement is safe to retry.
        """
        name(reason, 'settlement reason')
        if target is not None:
            if not isinstance(target, dict):
                raise ValueError('settlement target must be a mapping')
            target = json.loads(json.dumps(target, allow_nan=False))

        def operation(state, now):
            receipt = state['receipts'].get(token.allocation_id)
            if not receipt or receipt['token'] != asdict(token):
                return Refusal('stale', 'allocation id and fence do not match')
            existing = receipt.get('settlement')
            if existing is not None and existing.get('status') != 'abandoned':
                if target is not None and existing.get('target') not in (None, target):
                    return Refusal('settlement_conflict', 'settlement is bound to another study trial')
                if target is not None and existing.get('target') is None:
                    # Old accepted receipts remain readable and may acquire their missing binding
                    # from the original run, without rewriting an existing binding.
                    existing['target'] = target
                return deepcopy(existing)   # its own intent, whatever the lease says by now
            entry = self._owned(state, token, now)
            if isinstance(entry, Refusal):  # a new intent needs live, current ownership
                return entry
            if receipt.get('completion') is not None:
                return Refusal('completion_conflict', 'this allocation already accepted a publication')
            value = {'reason': reason, 'status': 'accepted', 'accepted_at': now}
            if target is not None:
                value['target'] = target
            receipt['settlement'] = value
            return deepcopy(value)
        return self._transaction(operation)

    def finish_settlement(self, token):
        """Release the task again once the study has been told. Idempotent, and never a new intent."""
        def operation(state, now):
            receipt = state['receipts'].get(token.allocation_id)
            if not receipt or receipt['token'] != asdict(token):
                return Refusal('stale', 'allocation id and fence do not match')
            intent = receipt.get('settlement')
            if intent is None:
                return Refusal('settlement_conflict', 'no settlement was accepted for this allocation')
            if intent['status'] == 'done':
                return Settled(token, False)
            if intent['status'] != 'accepted' or state['fences'].get(receipt['request']['task_id']) != token.fence:
                return Refusal('stale', 'settlement is no longer the current task intent')
            intent.update(status='done', completed_at=now)
            return Settled(token, True)
        return self._transaction(operation)

    def acquire(self, request, *, fence_floor=0):
        number(fence_floor, 'legacy fence floor', integer=True)
        from eforecast.training.compute_ready import admit
        return self._transaction(lambda state, now: admit(self,state,now,request,fence_floor=fence_floor))

    def _acquire(self, state, now, request, fence_floor=0):
        """Admission inside an existing transaction; policy queues may share this boundary."""
        if request.owner is not None and request.owner.boot_id != self.boot_id:
            return Refusal('process_identity', 'controller belongs to another host boot')
        key = self._attempt(request)
        previous = state['attempts'].get(key)
        if previous is not None:
            receipt = state['receipts'][previous]
            if asdict(request_from(receipt['request'])) != asdict(request):
                return Refusal('attempt_conflict', 'this attempt was submitted with different inputs/resources')
            return self._owned(state, Token(**receipt['token']), now)
        for value in state['allocations'].values():
            other = allocation_from(value)
            if other.request.task_id == request.task_id and (other.expires_at > now or
                    other.request.purpose == 'controller' or request.purpose == 'controller'):
                return Refusal('task_busy', 'a live lease already owns this task')
        pending = self._settling(state, request.task_id)
        if pending is not None:
            # Taking the task now would let the predecessor's cleanup tell the study FAIL
            # about work this allocation is doing. It waits for that settlement to finish.
            return Refusal('settling', f'the previous attempt is being settled ({pending})')
        refusal = self._fit(state, request)
        if refusal:
            return refusal
        fence = max(state['fences'].get(request.task_id, 0), fence_floor) + 1
        token = Token(uuid.uuid4().hex, fence)
        result = Allocation(token, request, now, now, now + request.lease_seconds,
                            now + request.spawn_seconds)
        state['fences'][request.task_id] = fence
        state['allocations'][token.allocation_id] = asdict(result)
        state['attempts'][key] = token.allocation_id
        state['receipts'][token.allocation_id] = {'token': asdict(token), 'request': asdict(request), 'closed': False}
        return result

    def validate(self, token):
        """Read the current ownership from the same ledger used by acquisition and mutation."""
        return self._transaction(lambda state, now: self._owned(state, token, now))

    def guard(self, token, publish):
        """Run one short publication under current ownership; callback errors stay distinct.

        This is the same linearization boundary used by accept_completion, without accepting
        an outcome. Never run preparation, process probes or training under this guard.
        """
        class CallbackFailed(Exception):
            pass
        def operation(state, now):
            entry = self._owned(state, token, now)
            if isinstance(entry, Refusal):
                return entry
            try:
                return publish()
            except BaseException as error:
                raise CallbackFailed() from error
        try:
            return self._transaction(operation)
        except CallbackFailed as error:
            raise error.__cause__

    def completion(self, token):
        def operation(state, now):
            receipt = state['receipts'].get(token.allocation_id)
            if not receipt or receipt['token'] != asdict(token):
                return Refusal('stale', 'allocation id and fence do not match')
            return deepcopy(receipt.get('completion'))
        return self._transaction(operation)

    def publish_completed(self, token, key, publication_id, publish):
        """Promote prepared output after quiescence, under the last accepted task fence.

        Large copies/hashes belong outside this critical section. The callback must be a
        short idempotent pointer/rename. A successor may use freed compute while copying;
        if it acquired the task, the predecessor can no longer change its public pointer.
        """
        name(key, 'publication key')
        if not isinstance(publication_id, str) or not re.fullmatch('[0-9a-f]{64}', publication_id):
            raise ValueError('publication id must be a SHA-256 digest')
        class CallbackFailed(Exception):
            pass
        def operation(state, now):
            receipt = state['receipts'].get(token.allocation_id)
            if not receipt or receipt['token'] != asdict(token):
                return Refusal('stale', 'allocation id and fence do not match')
            previous = receipt.get('publications', {}).get(key)
            if previous is not None:
                return Settled(token, False) if previous == publication_id else Refusal(
                    'publication_conflict', 'publication key already has different output')
            completed = receipt.get('completion') or {}
            if completed.get('status') != 'done' or (completed.get('result') or {}).get('state') != 'success':
                return Refusal('incomplete', 'publication requires a completed successful worker')
            if state['fences'].get(receipt['request']['task_id']) != token.fence:
                return Refusal('stale', 'a successor owns this task; late promotion refused')
            try:
                publish()
            except BaseException as error:
                raise CallbackFailed() from error
            receipt.setdefault('publications', {})[key] = publication_id
            return Settled(token, True)
        try:
            return self._transaction(operation)
        except CallbackFailed as error:
            raise error.__cause__

    def accept_completion(self, token, publication_id, result, *, publish=None, expected_revision=None):
        """Validate, optionally publish, and record acceptance in one critical section.

        Ownership linearizes at validation, before publication; no successor can acquire during
        the callback. Payload hashing and study settlement belong outside this transaction.
        Callback errors retain their original exception, distinct from ledger unavailability.
        """
        if not isinstance(publication_id, str) or not re.fullmatch('[0-9a-f]{64}', publication_id):
            raise ValueError('publication id must be a SHA-256 digest')
        if not isinstance(result, dict):
            raise ValueError('completion result must be a mapping')
        if expected_revision is not None:
            number(expected_revision, 'expected allocation revision', integer=True)
        class PublishFailed(Exception):
            pass
        def operation(state, now):
            entry = self._owned(state, token, now)
            if isinstance(entry, Refusal):
                return entry
            if expected_revision is not None and entry.revision != expected_revision:
                return Refusal('changed', 'allocation changed before completion acceptance')
            receipt = state['receipts'][token.allocation_id]
            settlement = receipt.get('settlement')
            if settlement is not None and settlement.get('status') != 'abandoned':
                return Refusal('settlement_conflict', 'this attempt is being settled as a failure')
            previous = receipt.get('completion')
            if previous is not None:
                return deepcopy(previous) if (previous['publication_id'] == publication_id
                                               and previous['result'] == result) else Refusal(
                    'completion_conflict', 'this allocation already accepted a different publication')
            if publish is not None:
                try:
                    publish()
                except BaseException as error:
                    raise PublishFailed() from error
            value = {'publication_id': publication_id, 'status': 'accepted',
                     'accepted_at': now, 'result': deepcopy(result)}
            receipt['completion'] = value
            return deepcopy(value)
        try:
            return self._transaction(operation)
        except PublishFailed as error:
            raise error.__cause__

    def finish_completion(self, token, publication_id):
        """Finish an already accepted intent; never accepts a new result or changes ownership.

        The study operation is idempotent and runs outside the host lock. Its durable intent may
        be finished after resource release/expiry, including recovery after an acknowledgement loss.
        """
        def operation(state, now):
            receipt = state['receipts'].get(token.allocation_id)
            if not receipt or receipt['token'] != asdict(token):
                return Refusal('stale', 'allocation id and fence do not match')
            value = receipt.get('completion')
            if value is None or value['publication_id'] != publication_id:
                return Refusal('completion_conflict', 'no matching accepted publication')
            if value['status'] == 'done':
                return Settled(token, False)
            value.update(status='done', completed_at=now)
            return Settled(token, True)
        return self._transaction(operation)

    def attach(self, token, process):
        def operation(state, now):
            entry = self._owned(state, token, now)
            if isinstance(entry, Refusal):
                return entry
            if process.host != entry.request.host or process.boot_id != self.boot_id:
                return Refusal('process_identity', 'process host/boot does not match the allocation')
            if entry.process is not None:
                return entry if entry.process == process else Refusal('process_identity', 'allocation has another process')
            result = replace(entry, process=process, status='running', revision=entry.revision + 1)
            state['allocations'][token.allocation_id] = asdict(result)
            return result
        return self._transaction(operation)

    def renew(self, token, lease_seconds=None):
        if lease_seconds is not None:
            number(lease_seconds, 'lease', positive=True)
        def operation(state, now):
            entry = self._owned(state, token, now, allow_expired_control=True)
            if isinstance(entry, Refusal):
                return entry
            result = replace(entry, renewed_at=now,
                             expires_at=now + (lease_seconds if lease_seconds is not None else entry.request.lease_seconds),
                             revision=entry.revision + 1)
            state['allocations'][token.allocation_id] = asdict(result)
            return result
        return self._transaction(operation)

    def resize(self, token, resources):
        def operation(state, now):
            entry = self._owned(state, token, now)
            if isinstance(entry, Refusal):
                return entry
            if entry.request.resources == resources:
                return entry  # an acknowledgement retry does not create a new revision
            request = replace(entry.request, resources=resources)
            refusal = self._fit(state, request, exclude=token.allocation_id)
            if refusal:
                return refusal
            result = replace(entry, request=request, revision=entry.revision + 1)
            state['allocations'][token.allocation_id] = asdict(result)
            return result
        return self._transaction(operation)

    def release(self, token, *, expected_revision=None):
        """Release current ownership, optionally only if an observed revision is unchanged."""
        if expected_revision is not None:
            number(expected_revision, 'expected allocation revision', integer=True)
        def operation(state, now):
            entry = self._owned(state, token, now, allow_expired_control=True)
            if isinstance(entry, Refusal):
                return Released(token, False) if entry.code == 'closed' else entry
            if expected_revision is not None and entry.revision != expected_revision:
                return Refusal('changed', 'allocation changed before conditional release')
            del state['allocations'][token.allocation_id]
            state['receipts'][token.allocation_id]['closed'] = True
            return Released(token, True)
        return self._transaction(operation)

    def snapshot(self):
        return self._transaction(lambda state, now: tuple(allocation_from(value) for value in state['allocations'].values()))

    def retire_stopped(self, token, probe, reason):
        """Reclaim a positively dead exact attempt and pin its study settlement atomically.

        Unlike lease expiry, this requires a real stopped-worker observation, checked against
        the unchanged allocation revision. An accepted completion always wins over cancellation.
        The durable settlement keeps the task closed after releasing its resource charge.
        """
        name(reason, 'retirement reason')
        def select(state, now):
            receipt = state['receipts'].get(token.allocation_id)
            if not receipt or receipt['token'] != asdict(token):
                return Refusal('stale', 'allocation id and fence do not match')
            value = state['allocations'].get(token.allocation_id)
            return allocation_from(value) if value else None
        original = self._transaction(select)
        if isinstance(original, Refusal):
            return original
        if original is not None:
            if probe(original) != 'dead':
                return Refusal('unconfirmed', 'owned worker is not confirmed stopped')
        def operation(state, now):
            receipt = state['receipts'].get(token.allocation_id)
            if not receipt or receipt['token'] != asdict(token):
                return Refusal('stale', 'allocation id and fence do not match')
            current = state['allocations'].get(token.allocation_id)
            if current != (asdict(original) if original else None):
                return Refusal('changed', 'allocation changed during stopped-worker check')
            if current is not None:
                target = original.request.settlement_json
                if receipt.get('completion') is None and not receipt.get('settlement'):
                    if state['fences'].get(original.request.task_id) != token.fence:
                        return Refusal('stale', 'a newer attempt owns this task; no failure settlement accepted')
                    receipt['settlement'] = {'reason': reason, 'status': 'accepted', 'accepted_at': now,
                                             'target': json.loads(target) if target else None}
                del state['allocations'][token.allocation_id]
                receipt['closed'] = True
            return deepcopy(receipt.get('settlement'))
        return self._transaction(operation)

    def reconcile(self, probe, *, token=None, unspawned_only=False):
        """Reconcile observations with the full token and unchanged allocation revision.

        Ordinary stopped-worker cleanup passes its token and cannot even probe unrelated
        reservations. A recovery coordinator may omit it to scan expiry candidates.
        Own unspawned cleanup additionally requires an unattached reservation past its spawn
        deadline. Eligibility uses this transaction's clock, including PostgreSQL server time.
        """
        def eligible(value, now):
            unspawned = value['process'] is None and value['spawn_deadline'] <= now
            return unspawned if unspawned_only else value['expires_at'] <= now or unspawned

        def select(state, now):
            if token is not None:
                receipt = state['receipts'].get(token.allocation_id)
                if not receipt or receipt['token'] != asdict(token):
                    return Refusal('stale', 'allocation id and fence do not match')
            return tuple(allocation_from(value) for key, value in state['allocations'].items()
                         if (token is None or key == token.allocation_id)
                         and eligible(value, now))
        candidates = self._transaction(select)
        if isinstance(candidates, Refusal):
            return candidates
        observations = []
        for entry in candidates:  # no lock across potentially slow process/container probes
            try:
                alive = probe(entry)
                if alive not in ('alive', 'dead', 'unknown'):
                    alive = 'unknown'
            except Exception as error:
                log.warning('liveness probe of %s failed: %s', entry.token.allocation_id, error)
                alive = 'unknown'
            observations.append((entry, alive))
        def operation(state, now):
            results = []
            def retain_settlement(entry, reason):
                receipt = state['receipts'][entry.token.allocation_id]
                if (entry.request.managed_run_id is not None and entry.request.settlement_json is not None
                        and state['fences'].get(entry.request.task_id) == entry.token.fence
                        and receipt.get('completion') is None and not receipt.get('settlement')):
                    receipt['settlement'] = {'reason': reason, 'status': 'accepted', 'accepted_at': now,
                                             'target': json.loads(entry.request.settlement_json)}
            for original, alive in observations:
                key = original.token.allocation_id
                current = state['allocations'].get(key)
                if current != asdict(original):
                    results.append(Reconciled(original.token, 'changed', 'allocation changed during the liveness check'))
                elif alive == 'dead':
                    retain_settlement(original, 'managed worker confirmed gone during allocator recovery')
                    del state['allocations'][key]
                    state['receipts'][key]['closed'] = True
                    results.append(Reconciled(original.token, 'released', 'confirmed no live owned process/container'))
                    log.info('reconciled allocation %s: confirmed dead', key)
                elif original.process is None and alive != 'alive' and now >= self._reclaim_at(original):
                    # Nothing was ever attached and nothing can be: the spawn deadline refuses it.
                    # A controller that is observably alive keeps its unattached reservation --
                    # only it knows whether it is still trying -- and reclaims its own leaks itself.
                    retain_settlement(original, 'managed reservation never attached before its spawn deadline')
                    del state['allocations'][key]
                    state['receipts'][key]['closed'] = True
                    waited = now - original.spawn_deadline
                    results.append(Reconciled(original.token, 'released',
                                              f'nothing was attached within the spawn deadline '
                                              f'({waited:.0f}s past it; owner {alive})'))
                    log.info('reconciled allocation %s: no process ever appeared (%.0fs past its spawn '
                             'deadline, owner %s)', key, waited, alive)
                else:
                    if original.status != 'expired_held':
                        state['allocations'][key] = asdict(replace(original, status='expired_held',
                                                                 revision=original.revision + 1))
                    results.append(Reconciled(original.token, 'retained', f'liveness {alive}; resources remain held'))
            return tuple(results)
        return self._transaction(operation)
