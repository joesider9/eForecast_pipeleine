# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Runtime preflight outside A08 transactions; reservations still authorize work.

Occupancy plus unmaterialized growth is conservative when PID/NVML attribution is
missing. This is sampled headroom, not protection from unmanaged future allocations.
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar

from .command_line.allocation_contract import Refusal
from .command_line.resource_reporting import process_tree

_observation=ContextVar('compute_headroom_observation',default=None)


def ledger_id(store):
    return str(store.directory),getattr(store,'namespace',None),getattr(store,'dsn',None)


def unchanged(store,state,request):
    """Cheap atomic validation; perform no physical probes under the A08 lock."""
    sample=_observation.get()
    if sample is None:return None
    expected_store,host,expected=sample
    actual=tuple(sorted((uid,a['revision']) for uid,a in state['allocations'].items()
                        if a['request']['host']==host))
    if expected_store!=ledger_id(store) or host!=request.host or actual!=expected:
        return Refusal('headroom_changed','reservations changed after physical sampling; refresh headroom before admission')


@contextmanager
def observing(store,request):
    observed=[]
    refused=check(store,request,observed=observed)
    token=_observation.set((ledger_id(store),request.host,tuple(sorted(observed))))
    try:yield refused
    finally:_observation.reset(token)


def check(store,request,*,observed=None):
    import psutil
    entries=store.snapshot()
    if isinstance(entries,Refusal):return entries
    if observed is not None:
        observed.extend((a.token.allocation_id,a.revision) for a in entries if a.request.host==request.host)
    if any((a.request.run_id,a.request.task_id,a.request.attempt_id)==
           (request.run_id,request.task_id,request.attempt_id) for a in entries):
        return None  # recover a committed acquisition, never count it twice
    local=[a for a in entries if a.request.host==request.host]
    from .command_line import liveness
    namespace=liveness.pid_namespace()
    attribution={};credited=set()
    for a in local:
        p=a.process
        verified=(p and p.pid and p.boot_id==store.boot_id and namespace and
                  namespace==liveness.procfs_pid_namespace() and p.pid_namespace==namespace and
                  p.start_ticks is not None and liveness.start_ticks(p.pid)==p.start_ticks)
        pids=[p.pid] if verified and (a.request.resources.ram_mb or a.request.resources.gpu_mb) else []
        info=process_tree(pids)
        if credited.intersection(info['pids']):info={'pids':[],'pss_mb':0.}
        credited.update(info['pids']);attribution[a.token.allocation_id]=info
    growth=sum(max(0.,a.request.resources.ram_mb-attribution[a.token.allocation_id]['pss_mb']) for a in local)
    free=psutil.virtual_memory().available/2**20-growth-float(os.environ.get('EFORECAST_HOST_HEADROOM_MB','8000'))
    if free<request.resources.ram_mb:
        return Refusal('headroom','physical RAM plus reserved growth cannot fit the request; no compute held')
    if request.device is None:return None
    try:
        import pynvml as nv
        nv.nvmlInit()
        try:
            try:handle=nv.nvmlDeviceGetHandleByUUID(request.device.uuid)
            except TypeError:handle=nv.nvmlDeviceGetHandleByUUID(request.device.uuid.encode())
            free=nv.nvmlDeviceGetMemoryInfo(handle).free/2**20
            processes={}
            for query in (nv.nvmlDeviceGetComputeRunningProcesses,nv.nvmlDeviceGetGraphicsRunningProcesses):
                for p in query(handle):
                    value=getattr(p,'usedGpuMemory',None)
                    if isinstance(value,int) and 0<=value<2**60:processes[p.pid]=max(value,processes.get(p.pid,0))
            peers=[a for a in local if a.request.device==request.device]
            growth=sum(max(0.,a.request.resources.gpu_mb-sum(processes.get(pid,0) for pid in
                attribution[a.token.allocation_id]['pids'])/2**20) for a in peers)
            free-=growth+float(os.environ.get('EFORECAST_GPU_HEADROOM_MB','512'))
            if free<request.resources.gpu_mb:
                return Refusal('headroom','physical GPU occupancy plus reserved growth cannot fit the request; no compute held')
        finally:nv.nvmlShutdown()
    except Exception as error:
        return Refusal('telemetry','GPU headroom is unknown; no compute held: '+str(error)[:300])
    return None
