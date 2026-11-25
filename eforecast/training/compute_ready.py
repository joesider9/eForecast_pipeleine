# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""One optional ordering policy, inside the existing A08 atomic transaction.

Tickets reserve nothing. _acquire/_fit remain the policy-free resource authority.
File and PostgreSQL share this code and clock; no daemon or per-adapter queue.
"""
from dataclasses import asdict
import hashlib
import json

from .command_line.allocation_contract import Refusal,number
from .command_line.allocator_file import allocation_from,request_from

KEY='compute_ready_v1'
DEFAULTS={'max_entries':256,'max_bypass':4,'age_seconds':30.,'heartbeat_seconds':30.}


def key(request):
    return hashlib.sha256(json.dumps([request.host,request.managed_run_id or request.run_id,
                                     request.task_id],separators=(',',':')).encode()).hexdigest()


def state_of(state):
    q=state.get(KEY)
    if q is None:return None
    if not isinstance(q,dict) or set(q)!={'version','config','next','entries'} or q['version']!=1:
        raise ValueError('invalid shared compute ordering state')
    config=q['config']
    if not isinstance(config,dict) or set(config)!=set(DEFAULTS):raise ValueError('invalid shared compute policy')
    for name in ['max_entries','max_bypass']:
        number(config[name],name,positive=True,integer=True)
    for name in ['age_seconds','heartbeat_seconds']:number(config[name],name,positive=True)
    if not 1<=config['max_entries']<=256 or config['max_bypass']>64:
        raise ValueError('shared compute queue bounds exceeded')
    if type(q['next']) is not int or q['next']<0 or not isinstance(q['entries'],dict) or len(q['entries'])>config['max_entries']:
        raise ValueError('invalid shared compute ticket index')
    ordinals=set()
    for uid,item in q['entries'].items():
        if set(item)!={'request','ordinal','created','expires','ready','reason','bypassed','limit','alias'}:
            raise ValueError('invalid shared compute ticket')
        req=request_from(item['request'])
        if key(req)!=uid:raise ValueError('shared compute ticket identity differs')
        for name in ['ordinal','bypassed']:number(item[name],name,integer=True)
        for name in ['created','expires']:number(item[name],name)
        if item['ordinal']>=q['next'] or item['ordinal'] in ordinals:
            raise ValueError('invalid shared compute ordinal')
        ordinals.add(item['ordinal'])
        if type(item['ready']) is not bool or type(item['limit']) is not int or item['limit']<1:
            raise ValueError('invalid shared compute eligibility')
        if not isinstance(item['reason'],str) or len(item['reason'])>1024 or (item['alias'] is not None and not isinstance(item['alias'],str)):
            raise ValueError('invalid shared compute wait metadata')
    return q


def impossible(store,request):
    r,c=request.resources,store.capacity
    if request.host!=c.host:return Refusal('host','request belongs to another host')
    if r.cpu_threads>c.cpu_threads or r.ram_mb>c.ram_mb:
        return Refusal('impossible','request exceeds installation CPU/RAM cap; change explicit resources')
    if request.device:
        d=c.devices.get(request.device.uuid)
        if d is None:return Refusal('device','GPU UUID is outside the installation budget')
        if r.gpu_mb>d.memory_mb or r.gpu_slots>min(d.slots,r.gpu_slot_limit or d.slots):
            return Refusal('impossible','request exceeds installation GPU memory/slot cap')


def enroll(store,state,now,request,*,limit=None,alias=None):
    q=state_of(state)
    if q is None or request.purpose=='controller' or store._attempt(request) in state['attempts']:return None
    bad=impossible(store,request)
    if bad:return bad
    if request.owner is not None and request.owner.boot_id!=state['boots'][request.host]:
        return Refusal('boot','request owner belongs to another host boot')
    uid=key(request);entry=q['entries'].get(uid)
    if (entry and entry['request']['attempt_id']!=request.attempt_id and entry['expires']>now
            and not (alias is not None and alias==entry['alias'])):
        return Refusal('queued','another attempt owns this logical task ticket; withdraw its exact attempt before replacement')
    if entry is None:
        if len(q['entries'])>=q['config']['max_entries']:
            return Refusal('queue_full','bounded compute ticket capacity reached; cancel/retire inactive tickets')
        entry={'ordinal':q['next'],'created':now,'bypassed':0,'limit':2**31-1,'alias':None};q['next']+=1
        q['entries'][uid]=entry
    entry.update(request=asdict(request),expires=now+q['config']['heartbeat_seconds'],
                 ready=True,reason='')
    if limit is not None:entry['limit']=limit
    if alias is not None:entry['alias']=alias
    return None


def admit(store,state,now,request,*,fence_floor=0,limit=2**31-1,alias=None):
    from .compute_headroom import unchanged
    stale=unchanged(store,state,request)
    if stale is not None:return stale
    q=state_of(state)
    if q is None or request.purpose=='controller':return store._acquire(state,now,request,fence_floor)
    # Lost acknowledgements are the original allocation, not another queue turn.
    if store._attempt(request) in state['attempts']:return store._acquire(state,now,request,fence_floor)
    refused=enroll(store,state,now,request,limit=limit,alias=alias)
    if refused is not None:return refused
    uid=key(request);entry=q['entries'][uid]
    active=[allocation_from(v) for v in state['allocations'].values()]
    held_tasks={a.request.task_id for a in active}
    # Scan completion/settlement history once, rather than once per ready ticket.
    # The file adapter already validates/loads the full ledger; queue lookup is
    # bounded at256 and adds O(history + pending*active), not pending*history.
    for receipt in state['receipts'].values():
        task=receipt['request']['task_id']
        if receipt['token']['fence']!=state['fences'].get(task):continue
        completion=receipt.get('completion') or {};settlement=receipt.get('settlement') or {}
        if (completion.get('status')=='accepted' or
                settlement.get('status')=='accepted' and settlement.get('reason') is not None):held_tasks.add(task)
    eligible=[]
    for other,item in sorted(q['entries'].items(),key=lambda p:p[1]['ordinal']):
        wanted=request_from(item['request'])
        if wanted.host!=request.host:continue
        if not item['ready'] or item['expires']<=now:
            continue  # retain age/history; this ticket holds no capacity
        if impossible(store,wanted):continue
        blocked=wanted.task_id in held_tasks
        count=sum(a.request.host==wanted.host and a.request.run_id==wanted.run_id
                  and a.request.settlement_json==wanted.settlement_json for a in active)
        if blocked or count>=item['limit']:
            item['reason']='waiting for owned task cleanup/settlement or caller concurrency limit'
            continue
        eligible.append((other,item,wanted))
    if not eligible:return Refusal('queued','no eligible shared compute turn; ownership/limit remains protected')
    head,old,wanted=eligible[0]
    chosen=None
    for other,item,proposed in eligible:
        fits=store._fit(state,proposed) is None
        if other==head:
            if fits:chosen=other;break
            old['reason']='waiting for capacity; eligible younger work may backfill within the bypass bound'
            if old['bypassed']>=q['config']['max_bypass'] or now-old['created']>=q['config']['age_seconds']:
                old['reason']='draining host compute for oldest ready request; finite running jobs/continued readiness required'
                break
        elif fits:chosen=other;break
    if chosen!=uid:
        reason=(old['reason'] if chosen is None else 'an older feasible request has the shared compute turn')
        entry['reason']=reason
        return Refusal('queued',reason)
    result=store._acquire(state,now,request,fence_floor)
    if not isinstance(result,Refusal):
        if uid!=head:old['bypassed']+=1
        del q['entries'][uid]
    return result


class ComputeQueue:
    def __init__(self,store):self.store=store

    def enqueue(self,request,*,alias=None):
        return self.store._transaction(lambda state,now:enroll(self.store,state,now,request,alias=alias))

    def configure(self,**config):
        chosen={**DEFAULTS,**config}
        def op(state,now):
            old=state_of(state)
            if old and old['config']==chosen:return chosen
            legacy=state.get('cpu_ready_v1',{}).get('entries',{})
            if state['allocations'] or old and old['entries'] or any(e['expires']>now for e in legacy.values()):
                return Refusal('busy','drain allocations and pending tickets before changing the shared policy')
            state[KEY]={'version':1,'config':chosen,'next':old['next'] if old else 0,'entries':{}}
            state_of(state)
            return chosen
        return self.store._transaction(op)

    def disable(self):
        def op(state,now):
            q=state_of(state)
            if q and (state['allocations'] or q['entries']):
                return Refusal('busy','drain allocations and pending tickets before disabling the shared policy')
            state.pop(KEY,None)
            return True
        return self.store._transaction(op)

    def pause(self,request,reason):
        def op(state,now):
            q=state_of(state);item=q['entries'].get(key(request)) if q else None
            if item and item['request']['attempt_id']==request.attempt_id:
                item.update(ready=False,reason=str(reason)[:1024])
            return True
        return self.store._transaction(op)

    def defer(self,request,reason,*,unknown=False):
        """Capacity wait stays ready to drain; true physical/unknown waits step aside."""
        def op(state,now):
            q=state_of(state);item=q['entries'].get(key(request)) if q else None
            if item and item['request']['attempt_id']==request.attempt_id:
                item.update(ready=not unknown and self.store._fit(state,request) is not None,
                            expires=now+q['config']['heartbeat_seconds'],reason=str(reason)[:1024])
            return True
        return self.store._transaction(op)


    def withdraw(self,request=None,*,alias=None):
        def op(state,now):
            q=state_of(state)
            if not q:return True
            for uid,item in list(q['entries'].items()):
                if (request is not None and uid==key(request) and item['request']['attempt_id']==request.attempt_id
                        or alias is not None and item['alias']==alias):
                    del q['entries'][uid]
            return True  # never releases an allocation
        return self.store._transaction(op)

    def retire(self,uid,attempt):
        """Operator cancellation of one exact ungranted ticket, never an allocation."""
        def op(state,now):
            q=state_of(state);item=q['entries'].get(uid) if q else None
            if item is not None:
                if item['request']['attempt_id']!=attempt:
                    return Refusal('changed','ticket now belongs to a different attempt')
                del q['entries'][uid]
            return True
        return self.store._transaction(op)

    def status(self,*,run_id=None):
        def op(state,now):
            q=state_of(state)
            if q is None:return {'enabled':False,'entries':[]}
            return {'enabled':True,'config':q['config'],'entries':[
                {'id':uid,'run_id':v['request']['managed_run_id'] or v['request']['run_id'],
                 'task_id':v['request']['task_id'],'host':v['request']['host'],
                 'attempt_id':v['request']['attempt_id'],
                 'resources':v['request']['resources'],'device':v['request']['device'],
                 'age_seconds':max(0.,now-v['created']),'ordinal':v['ordinal'],'bypassed':v['bypassed'],
                 'ready':v['ready'] and v['expires']>now,
                 'reason':v['reason'] if v['expires']>now else 'readiness heartbeat absent; age retained, no capacity held'}
                for uid,v in sorted(q['entries'].items(),key=lambda p:p[1]['ordinal'])
                if run_id is None or (v['request']['managed_run_id'] or v['request']['run_id'])==run_id]}
        return self.store._transaction(op)
