# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Storage-independent allocation contract. Placement/estimation belongs to the caller.

Capacity is the host budget, not a case's concurrency preference. A request names its chosen
device and complete logical task (including model tree/trial), with a distinct attempt identity.
All mutations carry the opaque allocation id AND its monotonically increasing task fence.
"""
from dataclasses import dataclass, field
from math import isfinite
from typing import Callable, Literal, Protocol

from .liveness import ProcessIdentity      # noqa: F401 -- a process is identified where its liveness is asked


def number(value, name, positive=False, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
            or value < 0 or (positive and value <= 0) or (integer and type(value) is not int)):
        raise ValueError(f'{name} must be a finite {"positive" if positive else "nonnegative"} '
                         f'{"integer" if integer else "number"}')


def name(value, field_name):
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError(f'{field_name} must be a nonempty identity')


@dataclass(frozen=True)
class Device:
    host: str
    uuid: str

    def __post_init__(self):
        name(self.host, 'device host')
        name(self.uuid, 'device UUID')
        if not self.uuid.startswith(('GPU-', 'MIG-')):
            raise ValueError('device identity must be a GPU/MIG UUID, never an ordinal')


@dataclass(frozen=True)
class Resources:
    cpu_threads: int = 0
    ram_mb: float = 0
    gpu_mb: float = 0
    gpu_slots: int = 0
    exclusive: bool = False
    gpu_slot_limit: int | None = None

    def __post_init__(self):
        for key in ('cpu_threads', 'ram_mb', 'gpu_mb', 'gpu_slots'):
            number(getattr(self, key), key, integer=key in ('cpu_threads', 'gpu_slots'))
        if type(self.exclusive) is not bool:
            raise ValueError('exclusive must be a boolean')
        if (self.gpu_mb or self.exclusive) and not self.gpu_slots:
            raise ValueError('a GPU reservation must also reserve a slot')
        if self.gpu_slot_limit is not None:
            number(self.gpu_slot_limit, 'GPU slot limit', positive=True, integer=True)
            if not self.gpu_slots:
                raise ValueError('a GPU slot limit requires a GPU reservation')


@dataclass(frozen=True)
class GpuCapacity:
    memory_mb: float
    slots: int

    def __post_init__(self):
        number(self.memory_mb, 'GPU capacity', positive=True)
        number(self.slots, 'GPU slots', positive=True, integer=True)


@dataclass(frozen=True)
class Capacity:
    host: str
    cpu_threads: int
    ram_mb: float
    devices: dict[str, GpuCapacity] = field(default_factory=dict)

    def __post_init__(self):
        name(self.host, 'host')
        number(self.cpu_threads, 'CPU capacity', integer=True)
        number(self.ram_mb, 'RAM capacity')
        for key, value in self.devices.items():
            Device(self.host, key)
            if not isinstance(value, GpuCapacity):
                raise ValueError('device capacity must be GpuCapacity')


@dataclass(frozen=True)
class Request:
    run_id: str
    task_id: str
    attempt_id: str
    host: str
    resources: Resources
    device: Device | None = None
    lease_seconds: float = 60
    spawn_seconds: float = 30
    owner: ProcessIdentity | None = None
    purpose: Literal['workload', 'controller'] = 'workload'
    managed_run_id: str | None = None
    managed_executor_id: str | None = None
    settlement_json: str | None = None
    # A process tree may outlive its supervisor. Its death alone is not quiescence;
    # the executor must collect its sealed descendant-cleanup receipt before release.
    supervision_id: str | None = None

    def __post_init__(self):
        for key in ('run_id', 'task_id', 'attempt_id', 'host'):
            name(getattr(self, key), key)
        for key in ('managed_run_id', 'managed_executor_id', 'supervision_id'):
            if getattr(self, key) is not None:
                name(getattr(self, key), key)
        if self.settlement_json is not None:
            import json
            if not isinstance(json.loads(self.settlement_json), dict):
                raise ValueError('settlement target must be a JSON mapping')
        if not isinstance(self.resources, Resources):
            raise ValueError('resources must be Resources')
        number(self.lease_seconds, 'lease', positive=True)
        number(self.spawn_seconds, 'spawn deadline', positive=True)
        if self.purpose not in {'workload', 'controller'}:
            raise ValueError('unknown allocation purpose')
        has_resources = any((self.resources.cpu_threads, self.resources.ram_mb,
                             self.resources.gpu_mb, self.resources.gpu_slots))
        if self.purpose == 'workload' and not has_resources:
            raise ValueError('an allocation must reserve resources')
        if self.purpose == 'controller' and (has_resources or self.device is not None or self.owner is None):
            raise ValueError('a controller claim needs an owner and reserves no compute resources')
        if self.device is not None and not isinstance(self.device, Device):
            raise ValueError('device must be Device')
        if self.device is not None and self.device.host != self.host:
            raise ValueError('the selected device must belong to the requested host')
        if bool(self.resources.gpu_slots) != (self.device is not None):
            raise ValueError('GPU slots and a device must be requested together')
        if self.owner is not None and not isinstance(self.owner, ProcessIdentity):
            raise ValueError('owner must be ProcessIdentity')
        if self.owner is not None and self.owner.host != self.host:
            raise ValueError('controller identity must belong to the requested host')


@dataclass(frozen=True)
class Token:
    allocation_id: str
    fence: int

    def __post_init__(self):
        name(self.allocation_id, 'allocation id')
        number(self.fence, 'fence', positive=True, integer=True)


@dataclass(frozen=True)
class Allocation:
    token: Token
    request: Request
    created_at: float
    renewed_at: float
    expires_at: float
    spawn_deadline: float
    revision: int = 0
    process: ProcessIdentity | None = None
    status: Literal['reserved', 'running', 'expired_held'] = 'reserved'


@dataclass(frozen=True)
class Refusal:
    code: str
    reason: str


@dataclass(frozen=True)
class Released:
    token: Token
    changed: bool


@dataclass(frozen=True)
class Settled:
    token: Token
    changed: bool


@dataclass(frozen=True)
class Reconciled:
    token: Token
    action: Literal['released', 'retained', 'changed']
    reason: str


Liveness = Literal['alive', 'dead', 'unknown']


class Allocator(Protocol):
    def guard(self, token: Token, publish: Callable[[], object]) -> object | Refusal: ...
    def acquire(self, request: Request, *, fence_floor: int = 0) -> Allocation | Refusal: ...
    def validate(self, token: Token) -> Allocation | Refusal: ...
    def completion(self, token: Token) -> dict | None | Refusal: ...
    # A settlement is the opposite of a completion: this attempt will not finish, and the study
    # must be told so. The intent is recorded while its owner still holds the fence, and the task
    # cannot be acquired again until it is finished -- otherwise the successor's own result would
    # be overwritten by its predecessor's cleanup, which the study has no way to undo.
    def settlement(self, token: Token) -> dict | None | Refusal: ...
    def pending_settlements(self) -> tuple[dict, ...] | Refusal: ...
    def pending_completions(self) -> tuple[dict, ...] | Refusal: ...
    def accept_settlement(self, token: Token, reason: str, *, target: dict | None = None) -> dict | Refusal: ...
    def finish_settlement(self, token: Token) -> Settled | Refusal: ...
    def accept_completion(self, token: Token, publication_id: str, result: dict,
                          *, publish: Callable[[], None] | None = None,
                          expected_revision: int | None = None) -> dict | Refusal: ...
    def finish_completion(self, token: Token, publication_id: str) -> Settled | Refusal: ...
    def publish_completed(self, token: Token, key: str, publication_id: str,
                          publish: Callable[[], None]) -> Settled | Refusal: ...
    def attach(self, token: Token, process: ProcessIdentity) -> Allocation | Refusal: ...
    def renew(self, token: Token, lease_seconds: float | None = None) -> Allocation | Refusal: ...
    def resize(self, token: Token, resources: Resources) -> Allocation | Refusal: ...
    def release(self, token: Token, *, expected_revision: int | None = None) -> Released | Refusal: ...
    def snapshot(self) -> tuple[Allocation, ...] | Refusal: ...
    def retire_stopped(self, token: Token, probe: Callable[[Allocation], Liveness],
                       reason: str) -> dict | None | Refusal: ...
    def reconcile(self, probe: Callable[[Allocation], Liveness], *,
                  token: Token | None = None,
                  unspawned_only: bool = False) -> tuple[Reconciled, ...] | Refusal: ...
