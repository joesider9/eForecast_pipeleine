# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Neutral pinned-profile envelope. No host, database, hardware or ML imports.

A profile supplies preferences, never capacity. Physical input adapters are explicit,
versioned extensions of the device/thread policy; unsupported numerical choices stay off.
"""
import hashlib
import json


def supported(definition):
    policy, workload = definition['policy'], definition['key']['workload']
    numerical = policy.get('numerical')
    if numerical is None:
        if policy['precision'] != 'reference' or policy['compile'] or workload['precision'] != 'reference':
            return False
    else:
        if (numerical.get('version') != 1 or policy['precision'] != workload['precision']
                or policy.get('batches') is not None or policy.get('gpu_budget') is not None
                or policy['device'] != 'cuda' or workload['model'] != 'MLP'
                or workload['layout'] != 'row-major' or workload['dtype'] not in {'float32','float64'}
                or policy['compile'] and policy['precision'] not in {'reference','fp32'}):
            return False  # combined AMP/compile needs an independent quality gate
        try:
            major, minor = map(int, definition['key']['hardware']['compute_capability'].split('.'))
        except (ValueError, AttributeError, TypeError):
            return False
        if (policy['precision'] == 'bf16' or numerical['matmul'] != 'highest') and major < 8:
            return False  # native BF16/TF32, not an emulated broad library flag
        if policy['precision'] == 'fp16' and major < 7:
            return False
    batches = policy.get('batches')
    if batches is not None:
        return (batches.get('version') == 1 and policy.get('gpu_budget') is None
                and 0 <= policy['loader_workers'] <= 4
                and workload['residency'] == policy['residency'] == 'mapped'
                and workload['dtype'] in {'float32','float64'}
                and workload['layout'] in {'row-major','sequence','grid','image'}
                and (policy['device'] == 'cuda' or not batches['pinned_bytes'] and not batches['overlap'])
                and (not batches['overlap'] or batches['pinned_bytes'] > 0))
    if policy['loader_workers']:
        return False
    budget = policy.get('gpu_budget')
    if budget is None:
        return policy['residency'] == workload['residency'] == 'host'
    return (policy['device'] == 'cuda' and workload['model'] in {'MLP', 'LSTM'}
            and policy['residency'] in {'host', 'device'} and policy['residency'] == workload['residency']
            and workload['dtype'] in {'float32', 'float64'}
            and workload['layout'] in {'row-major', 'sequence'}
            and set(budget) == {'data_bytes', 'host_bytes', 'workspace_mb', 'context_mb', 'max_rows'}
            and all(type(value) is int and value > 0 for value in budget.values())
            and budget['context_mb'] >= 256)


def check_grant(plan, resources, device_uuid):
    """Before acquisition/loading: a complete explicit envelope must fit the request."""
    validate(plan)
    policy = (plan.get('definition') or {}).get('policy', {})
    if policy.get('numerical'):
        if not device_uuid or not resources.get('gpu_slots'):
            raise ValueError('numerical CUDA policy requires an allocated UUID and slot')
        if resources['ram_mb'] < 2048 or resources.get('gpu_mb',0) < 1024:
            raise ValueError('numerical adapter requires its host library and CUDA context floor')
        if policy['compile'] and (resources['cpu_threads'] < 2 or resources['ram_mb'] < 8192
                                  or resources.get('gpu_mb',0) < 4096):
            raise ValueError('compiler requires CPU2/RAM8192/GPU4096 process-tree envelope')
        if policy['compile'] and policy['threads'] + 1 > resources['cpu_threads']:
            raise ValueError('compiler coordinator plus native model threads exceed CPU request')
    batches = policy.get('batches')
    if batches is not None:
        workers = policy['loader_workers']
        window=max(1,batches['prefetch_batches'])
        if (3*workers+window+4)*batches['batch_bytes']+batches['pinned_bytes'] > batches['host_bytes']:
            raise ValueError('bounded batch buffers exceed the host allowance')
        if workers and (window+1)*(batches['batch_bytes']+65536)+512*1024 > batches['ipc_bytes']:
            raise ValueError('bounded batch transport exceeds the scratch allowance')
        if policy['threads'] + workers > resources['cpu_threads']:
            raise ValueError('model threads plus CPU loader children exceed the CPU request')
        if batches['host_bytes'] + (2048 + workers*512)*2**20 > resources['ram_mb']*2**20:
            raise ValueError('batch/loader/library envelope exceeds the host RAM request')
        if policy['device'] == 'cuda':
            if not device_uuid or not resources.get('gpu_slots'):
                raise ValueError('CUDA staging requires an allocated UUID and slot')
            if batches['gpu_bytes'] + (batches['workspace_mb']+batches['context_mb'])*2**20 > resources.get('gpu_mb',0)*2**20:
                raise ValueError('batch/context/model envelope exceeds the GPU request')
        return
    budget = (plan.get('definition') or {}).get('policy', {}).get('gpu_budget')
    if budget is None:
        return
    if not device_uuid or not resources.get('gpu_slots'):
        raise ValueError('GPU input profile requires an allocated UUID and slot')
    if budget['data_bytes'] + (budget['context_mb'] + budget['workspace_mb']) * 2**20 > resources.get('gpu_mb', 0) * 2**20:
        raise ValueError('GPU input/context/model envelope exceeds the GPU request')
    if budget['host_bytes'] + 2048 * 2**20 > resources['ram_mb'] * 2**20:
        raise ValueError('GPU input host allowance plus library floor exceeds the RAM request')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate(plan):
    if not isinstance(plan, dict) or set(plan) != {
            'version', 'id', 'requested_id', 'status', 'reason', 'definition', 'execution',
            'key_id', 'numerical_id', 'scope_id'} or plan['version'] != 1:
        raise ValueError('invalid pinned profile plan')
    if plan['id'] != identity({k: v for k, v in plan.items() if k != 'id'}):
        raise ValueError('pinned profile plan changed')
    if plan['status'] not in {'selected', 'calibration', 'fallback'}:
        raise ValueError('invalid profile resolution status')
    if plan['status'] == 'fallback':
        if plan['execution'] != {'threads': None, 'device': None} or plan['definition'] is not None:
            raise ValueError('fallback cannot activate a profile')
        return plan
    definition = plan['definition']
    if not isinstance(definition, dict) or identity(definition) != plan['requested_id']:
        raise ValueError('pinned profile definition changed')
    if identity(definition['key']) != plan['key_id']:
        raise ValueError('pinned profile compatibility key changed')
    workload = definition['key']['workload']
    if numerical_identity(definition) != plan['numerical_id']:
        raise ValueError('pinned numerical identity changed')
    policy = definition['policy']
    if not supported(definition):
        raise ValueError('profile needs an unsupported execution adapter')
    if plan['execution'] != {k: policy[k] for k in ('threads', 'device')}:
        raise ValueError('profile execution differs from its pinned policy')
    return plan


def numerical_identity(definition):
    workload, policy = definition['key']['workload'], definition['policy']
    value = {k: workload[k] for k in ('model','implementation','graph','batch_size','dtype','precision','optimizer')}
    if policy.get('numerical') is not None:
        value['adapter'] = {'version':1,'precision':policy['precision'],
                            'matmul':policy['numerical']['matmul'],'compile':policy['compile'],
                            'backend':policy['numerical']['backend'] if policy['compile'] else None}
    return identity(value)


def preferences(requested):
    """Explicit caller choices win; the grant will still normalize these preferences."""
    result = dict(requested)
    plan = result.get('profile_plan')
    if plan is not None:
        validate(plan)
        for key, value in plan['execution'].items():
            if value is not None:
                result.setdefault(key, value)
    return result
