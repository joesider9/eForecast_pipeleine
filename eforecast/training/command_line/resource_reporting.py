# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""What a run is seen to use: CPUs this process may use, memory its processes hold, thread limits, the status file."""
import contextlib
import json
import logging
import math
import os
import sys
from pathlib import Path
import tempfile
import time

log = logging.getLogger(__name__)


class ResourceUnavailable(RuntimeError):
    pass


def cpu_inventory():
    import psutil
    logical = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = logical
    quota = None
    try:
        # The process's cgroup may be nested below the mount root.
        relative = next(line.split('::', 1)[1].strip() for line in Path('/proc/self/cgroup').read_text().splitlines()
                        if '::' in line)
        mount = Path('/sys/fs/cgroup')
        group = mount / relative.lstrip('/')
        candidates = [group / 'cpu.max', *[p / 'cpu.max' for p in group.parents if p == mount or mount in p.parents]]
        quotas = []
        for path in candidates:
            if path.exists():
                limit, period = path.read_text().split()
                if limit != 'max':
                    quotas.append(int(limit) / int(period))
        quota = min(quotas) if quotas else None
    except (OSError, ValueError, StopIteration):
        pass
    if quota is None:
        try:
            limit = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
            period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
            if limit > 0:
                quota = limit / period
        except (OSError, ValueError):
            pass
    return {'physical_cores': psutil.cpu_count(logical=False), 'logical_cpus': logical,
            'affinity_cpus': affinity, 'cgroup_quota_cpus': quota,
            'effective_cpus': min(affinity, quota) if quota is not None else affinity}


def host_total_mb():
    """Physical host RAM in MB."""
    try:
        with open('/proc/meminfo') as stream:
            for line in stream:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float('inf')


def cpu_slots():
    return max(1, math.floor(cpu_inventory()['effective_cpus']))


def process_tree(pids):
    """Current host PSS and process ids of ``pids`` and their descendants; RSS is reported beside it."""
    import psutil
    processes, complete = {}, bool(pids)
    for pid in set(pids):
        try:
            parent = psutil.Process(pid)
            for process in [parent, *parent.children(recursive=True)]:
                processes[process.pid] = process
        except (psutil.Error, ValueError):
            complete = False
    pss, rss, threads = 0.0, 0.0, 0
    for process in processes.values():
        try:
            info = process.memory_full_info()
            rss += info.rss / 2 ** 20
            if hasattr(info, 'pss'):
                pss += info.pss / 2 ** 20
            else:
                complete = False
            threads += process.num_threads()
        except psutil.Error:
            complete = False
    return {'pids': sorted(processes), 'pss_mb': pss, 'rss_mb': rss,
            'os_threads': threads, 'confidence': 'measured_pss' if complete else 'partial_or_unavailable'}


def memory_charge(measured_mb, reservations):
    """Memory in use on a card plus what its young trials have yet to take of their estimates. A trial whose share cannot be told counts in full."""
    growth = sum(max(0., reserved - observed) for reserved, observed in reservations)
    return None if measured_mb is None else measured_mb + growth


from eforecast.execution_policy import THREAD_ENV


def thread_environment(threads):
    return {**dict.fromkeys(THREAD_ENV, str(max(1, int(threads)))),
            'EFORECAST_TRIAL_THREADS': str(max(1, int(threads))), 'EFORECAST_LOADER_WORKERS': '0'}


@contextlib.contextmanager
def library_threads(threads):
    """Estimator parallelism and native pools share one allocation; loaders run in-process."""
    from threadpoolctl import threadpool_limits
    import torch
    previous = {key: os.environ.get(key) for key in THREAD_ENV}
    old_torch = torch.get_num_threads()
    cv2 = sys.modules.get('cv2')
    numexpr = sys.modules.get('numexpr')
    old_numexpr = numexpr.get_num_threads() if numexpr else None
    if numexpr:
        numexpr.set_num_threads(threads)
    old_cv = cv2.getNumThreads() if cv2 else None
    if cv2:
        cv2.setNumThreads(1)
    os.environ.update({key: str(threads) for key in THREAD_ENV})
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # interop can only be set before work starts
        pass
    try:
        # Native BLAS is serial under a parallel estimator to avoid nested N x N pools.
        with threadpool_limits(limits=1, user_api='blas'), threadpool_limits(limits=threads, user_api='openmp'):
            yield
    finally:
        torch.set_num_threads(old_torch)
        if cv2:
            cv2.setNumThreads(old_cv)
        if numexpr:
            numexpr.set_num_threads(old_numexpr)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def write_status(payload, path=None):
    """Atomic per-run snapshot for CLI records and the existing UI run endpoint."""
    path = path or os.environ.get('EFORECAST_RESOURCE_STATUS_FILE')
    if not path:
        return
    path = Path(path)
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump({'updated': time.time(), **payload}, stream, indent=1, allow_nan=False)
        os.replace(temporary, path)
        return True
    except OSError as error:
        log.warning('resource snapshot unavailable at %s: %s', path, error)
        return False
    finally:
        if temporary and os.path.exists(temporary):
            with contextlib.suppress(OSError):
                os.unlink(temporary)
