# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Is the owner of a reservation still there? Observed where it can be, never assumed.

An expired lease says nothing about whether the worker stopped. A08 is explicit: an expired
heartbeat starts reconciliation, it does not free resources. So the only thing that may release
another run's reservation is a **positive** observation that its process is gone -- and the only
place that observation can be made is the machine the process was on.

Three answers, and the honest one matters most:

``dead``      the kernel this process belongs to is gone, or its PID is gone, or the PID is now
              another process (birth ticks differ), or its container no longer exists;
``alive``     the process or the container is there. A stopped process (``SIGSTOP``) is alive;
``unknown``   another machine, an unverified PID namespace, an unreadable ``/proc``, or a Docker
              daemon that could not answer.
              The reservation keeps its resources, and the operator is told rather than surprised.

All three answer one question: may this reservation be released. ``observe(..., which='controller')``
asks a different one -- is anybody still waiting for this run -- which has its own answer and its
own consequence: see ``identity_of``.

Telling *another machine* from *an earlier boot of this one* needs an identity that outlives a
reboot, which the boot-derived host string is not: see ``machine_id``.

Probes run outside the allocator's lock, because asking Docker can take as long as it likes.
"""
from dataclasses import dataclass
import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _name(value, field_name):
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError(f'{field_name} must be a nonempty identity')


def _whole(value, field_name, positive=False):
    if type(value) is not int or value < 0 or (positive and value <= 0):
        raise ValueError(f'{field_name} must be a {"positive" if positive else "nonnegative"} integer')


@dataclass(frozen=True)
class ProcessIdentity:
    """A PID is numbered in ``pid_namespace``; legacy missing evidence stays unknown.

    This is the namespace assigning the recorded number, which can differ from a
    child's active namespace after it enters a nested PID namespace.
    """
    host: str
    boot_id: str
    pid: int | None = None
    start_ticks: int | None = None
    container_id: str | None = None
    pid_namespace: str | None = None
    #: Which machine, as opposed to which boot. Absent on reservations recorded before this
    #: field existed, which therefore fall back to comparing host strings. See ``liveness``.
    machine_id: str | None = None

    def __post_init__(self):
        _name(self.host, 'process host')
        _name(self.boot_id, 'process boot')
        if self.pid is not None:
            _whole(self.pid, 'PID', positive=True)
            _whole(self.start_ticks, 'process start ticks')
        elif self.start_ticks is not None:
            raise ValueError('process start ticks require a PID')
        if self.container_id is not None:
            _name(self.container_id, 'container id')
        if self.pid_namespace is not None:
            _name(self.pid_namespace, 'PID namespace')
        if self.machine_id is not None:
            _name(self.machine_id, 'machine id')
        if self.pid is None and self.container_id is None:
            raise ValueError('a process identity needs a PID/start identity or a container id')



BOOT_FILE = '/proc/sys/kernel/random/boot_id'
#: Stable for the life of the installation, unlike the boot id. Read in this order.
MACHINE_FILES = ('/etc/machine-id', '/var/lib/dbus/machine-id')
#: The prefix of a host identity ``host_id`` derived from the boot id rather than being told.
BOOT_HOST = 'linux-boot-'
DOCKER_TIMEOUT = 20


def boot_id():
    try:
        return Path(BOOT_FILE).read_text().strip() or None
    except OSError:
        return None


def machine_id():
    """Which machine this is, across reboots.

    ``host_id`` below answers *which boot*, because it is derived from the boot id whenever no
    explicit ``EFORECAST_ALLOCATOR_HOST`` is set. That conflates two different questions, and it
    cost this module its most useful verdict: ``observe`` compared hosts first, so a reservation
    from a previous boot of this very machine looked like one belonging to a host that cannot be
    observed, and the ``dead`` branch for a kernel that is gone was unreachable. Every reboot then
    stranded whatever was active at shutdown, for good.

    So record the machine separately. Same machine and an older boot is a positive observation
    that the process is gone; a different machine stays unknown, as it must.
    """
    for name in MACHINE_FILES:
        try:
            value = Path(name).read_text().strip()
        except OSError:
            continue
        if value:
            return value
    return None


def host_id(boot=None):
    """The same host identity the allocator records, so a probe compares like with like."""
    boot = boot or boot_id()
    return os.environ.get('EFORECAST_ALLOCATOR_HOST') or (f'{BOOT_HOST}{boot}' if boot else None)


def boot_derived(host):
    """Whether this host string is one ``host_id`` made from a boot id rather than was told.

    Only such a string changes by itself at every reboot. An ``EFORECAST_ALLOCATOR_HOST`` names
    a host on purpose, and two of those can share one machine, so a difference between two told
    names is never a reboot -- it is two hosts that must not touch each other's work.
    """
    return isinstance(host, str) and host.startswith(BOOT_HOST)


def start_ticks(pid):
    """Field 22 of ``/proc/<pid>/stat``: when this PID was born, so a reused number is not it."""
    stat = Path('/proc') / str(pid) / 'stat'
    return int(stat.read_text().rsplit(')', 1)[1].split()[19])


def pid_namespace():
    """The calling process's PID-numbering namespace, if it can be read."""
    try:
        return os.readlink('/proc/self/ns/pid') or None
    except OSError:
        return None


def procfs_pid_namespace():
    """Identify the namespace whose PID numbers this procfs mount exposes.

    An inherited/bind-mounted procfs may expose an ancestor's numbers. ``NSpid`` is
    ordered from the procfs mount's namespace down to the reader's active namespace.
    A single matching PID proves these are the same. Otherwise PID 1 of the procfs
    view identifies its namespace. If neither is readable, visibility is unknown.
    """
    namespace = pid_namespace()
    try:
        status = Path('/proc/self/status').read_text().splitlines()
        pids = next(line.split()[1:] for line in status if line.startswith('NSpid:'))
        if namespace is not None and [int(pid) for pid in pids] == [os.getpid()]:
            return namespace
    except (OSError, ValueError, StopIteration):
        pass
    try:
        return os.readlink('/proc/1/ns/pid') or None
    except OSError:
        return None


def process_liveness(identity):
    if identity.pid_namespace is None:
        return 'unknown', f'PID {identity.pid} has no recorded PID namespace; visibility is unverified'
    namespace = procfs_pid_namespace()
    if namespace is None:
        return 'unknown', f'cannot verify the local procfs PID namespace for PID {identity.pid}'
    if identity.pid_namespace != namespace:
        return 'unknown', (f'PID {identity.pid} belongs to namespace {identity.pid_namespace}; '
                           f'local procfs exposes {namespace}')
    try:
        ticks = start_ticks(identity.pid)
    except FileNotFoundError:
        return 'dead', f'no process {identity.pid}'
    except PermissionError as error:
        return 'unknown', f'cannot read /proc/{identity.pid} ({error})'
    except (OSError, ValueError, IndexError) as error:
        return 'unknown', f'unreadable /proc/{identity.pid} ({error})'
    if identity.start_ticks is not None and ticks != identity.start_ticks:
        return 'dead', f'PID {identity.pid} was reused: born at {ticks}, not {identity.start_ticks}'
    return 'alive', f'PID {identity.pid} is running'


def container_liveness(container_id, inspect=None):
    inspect = inspect or _inspect
    try:
        state = inspect(container_id)
    except Exception as error:                      # noqa: BLE001 -- a daemon that cannot answer is unknown
        return 'unknown', f'docker could not be asked about {container_id[:12]} ({error})'
    if state is None:
        return 'dead', f'container {container_id[:12]} no longer exists'
    if state in ('running', 'created', 'restarting', 'paused'):
        return 'alive', f'container {container_id[:12]} is {state}'
    if state == 'unknown':
        return 'unknown', f'the state of container {container_id[:12]} could not be read'
    return 'dead', f'container {container_id[:12]} is {state}'


def _inspect(container_id):
    """``None`` when Docker says it does not exist; the state name otherwise."""
    done = subprocess.run(['docker', 'inspect', '--format', '{{.State.Status}}', container_id],
                          capture_output=True, text=True, timeout=DOCKER_TIMEOUT)
    if done.returncode == 0:
        return done.stdout.strip() or 'unknown'
    message = (done.stderr or '').lower()
    if 'no such object' in message or 'no such container' in message:
        return None
    raise RuntimeError((done.stderr or done.stdout).strip()[:200])


def identity_of(allocation, which=None):
    """Which process a verdict about this reservation concerns, and what to call it.

    By default, whatever the strongest evidence is: the worker once one is attached, and the
    controller that reserved before that. ``which='controller'`` asks about the launching
    process even when a worker is attached, which is a different question. "May this
    reservation be released" is answered by the worker. "Is anyone still waiting for this
    work" is answered by the controller -- and a run whose worker is alive while its
    controller is gone is an orphan: it will finish, hold the machine until it does, and
    nobody will ever come back for the result.
    """
    if which == 'controller':
        request = allocation.request
        return (request.owner if request is not None else None), 'controller'
    if allocation.process is not None:
        return allocation.process, 'worker'
    return allocation.request.owner, 'controller'


def observe(allocation, *, host=None, boot=None, inspect=None, machine=None, which=None):
    """``(liveness, reason)`` for one reservation, from the strongest evidence available.

    ``which='controller'`` asks about the process that launched the run instead, however far
    the run itself has got: see ``identity_of``.
    """
    boot = boot or boot_id()
    host = host or host_id(boot)
    identity, where = identity_of(allocation, which)
    if identity is None:
        return 'unknown', ('no controller identity was recorded' if allocation.process is not None
                           else 'nothing has been attached and no controller identity was recorded')
    # A machine id refines the host check; it never replaces it. Getting that backwards let a
    # record naming a foreign host be observed, and signalled, merely because it happened to sit
    # on this machine -- and two logical hosts routinely share one machine.
    mine = machine if machine is not None else machine_id()
    theirs = getattr(identity, 'machine_id', None)
    same_machine = mine is not None and theirs is not None and theirs == mine
    if host is not None and identity.host != host:
        # The one difference that is not another host: two boot-derived names on one machine,
        # which is this machine after a reboot. The boot check below is what may call that dead.
        if not (same_machine and boot_derived(identity.host) and boot_derived(host)):
            return 'unknown', f'its {where} belongs to {identity.host}, which this host cannot observe'
    elif mine is not None and theirs is not None and not same_machine:
        # The same host name on two machines: the name agrees, the hardware does not.
        return 'unknown', (f'its {where} belongs to machine {theirs[:8]}, '
                           f'which this host cannot observe')
    if boot is not None and identity.boot_id != boot:
        return 'dead', f'its {where} belongs to kernel boot {identity.boot_id[:8]}, which is gone'
    if identity.container_id:
        verdict, reason = container_liveness(identity.container_id, inspect=inspect)
        return verdict, f'{where}: {reason}'
    if identity.pid is not None:
        verdict, reason = process_liveness(identity)
        if (verdict == 'dead' and where == 'worker'
                and getattr(allocation.request, 'supervision_id', None)):
            return 'unknown', f'{reason}; supervised descendants require executor cleanup evidence'
        if verdict == 'dead' and where == 'controller' and allocation.process is None:
            reason = f'{reason}, and nothing was ever attached'
        return verdict, f'{where}: {reason}'
    return 'unknown', f'its {where} identity names neither a process nor a container'


def probe(**options):
    """A probe for ``Allocator.reconcile``; the reasons are collected for the report."""
    seen = {}

    def ask(allocation):
        verdict, reason = observe(allocation, **options)
        seen[allocation.token.allocation_id] = (verdict, reason)
        return verdict

    ask.seen = seen
    return ask


def process_identity(pid, host, boot_id):
    """Record the namespace numbering the caller's PID, not the child's active namespace."""
    pid = int(pid)
    namespace = pid_namespace()
    # Popen.pid is numbered in our namespace. A child may have entered a nested namespace,
    # so /proc/<pid>/ns/pid would describe a different numbering domain. For ourselves,
    # /proc/self is unambiguous even when procfs was mounted in an ancestor namespace.
    ticks = start_ticks('self' if pid == os.getpid() else pid)
    if pid != os.getpid() and (namespace is None or namespace != procfs_pid_namespace()):
        namespace = None
        log.warning('PID %s recorded without namespace evidence: local procfs visibility is unverified', pid)
    return ProcessIdentity(host, boot_id, pid, ticks, pid_namespace=namespace, machine_id=machine_id())
