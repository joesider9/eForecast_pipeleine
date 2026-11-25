# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Version 1 numerical-window protocol for isolated model adapters (stdlib only).

References describe immutable .npy arrays, never Python objects. Bind storage locations
outside this document. Axes are (sample, time, feature); timestamps are UTC nanoseconds.
Validation here checks the protocol, NOT file bytes, numerical finiteness, admission,
completion or publication. Those remain the existing worker/manifest authority's job.
"""
from copy import deepcopy
import hashlib
import json
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

VERSION = 1
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_DTYPES = {'<f4', '<f8', '<i8', '|b1'}


class ContractError(ValueError):
    """An invalid portable document, with the field that needs correction."""


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ContractError(f'contract must be finite JSON: {error}') from error


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _object(value, keys, field):
    if type(value) is not dict or set(value) != set(keys.split()):
        raise ContractError(f'{field}: expected fields {keys}')


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ContractError(f'{field}: expected nonempty text')


def _integer(value, field, minimum=1):
    if type(value) is not int or value < minimum:
        raise ContractError(f'{field}: expected integer >= {minimum}')


def _names(value, field, empty=False):
    if type(value) is not list or (not empty and not value):
        raise ContractError(f'{field}: expected an ordered list')
    for item in value:
        _text(item, field)
    if len(set(value)) != len(value):
        raise ContractError(f'{field}: duplicate names')


def _sha(value, field):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ContractError(f'{field}: expected a SHA-256 digest')


def _identity(value):
    _object(value, 'case_id run_id task_id attempt_id allocation_id fence', 'identity')
    for key, item in value.items():
        if key == 'fence':
            _integer(item, 'identity.fence')
        else:
            _text(item, f'identity.{key}')


def _runtime(value):
    _object(value, 'id revision', 'runtime')
    _text(value['id'], 'runtime.id')
    _sha(value['revision'], 'runtime.revision')


def _component(value, field):
    """Native checkpoint bytes are opaque outside the runtime named by the envelope."""
    _object(value, 'adapter_revision parameters checkpoint artifacts', field)
    _sha(value['adapter_revision'], f'{field}.adapter_revision')
    if type(value['parameters']) is not dict:
        raise ContractError(f'{field}.parameters: expected JSON configuration')
    checkpoint = value['checkpoint']
    if checkpoint is not None:
        _object(checkpoint, 'id revision', f'{field}.checkpoint')
        _text(checkpoint['id'], f'{field}.checkpoint.id')
        _text(checkpoint['revision'], f'{field}.checkpoint.revision')
        if not re.fullmatch(r'(?:[0-9a-f]{40}|(?:sha256:)?[0-9a-f]{64})', checkpoint['revision']):
            raise ContractError(f'{field}.checkpoint.revision: use a commit or content digest, not a branch/tag')
    if type(value['artifacts']) is not list:
        raise ContractError(f'{field}.artifacts: expected opaque artifact references')
    ids = []
    for artifact in value['artifacts']:
        _object(artifact, 'artifact_id sha256 bytes', f'{field}.artifacts')
        _text(artifact['artifact_id'], f'{field}.artifacts.artifact_id')
        _sha(artifact['sha256'], f'{field}.artifacts.sha256')
        _integer(artifact['bytes'], f'{field}.artifacts.bytes')
        ids.append(artifact['artifact_id'])
    if len(set(ids)) != len(ids):
        raise ContractError(f'{field}.artifacts: duplicate artifact id')


def _array(value, shape, dtypes, field):
    _object(value, 'artifact_id format sha256 bytes dtype shape order', field)
    _text(value['artifact_id'], f'{field}.artifact_id')
    _sha(value['sha256'], f'{field}.sha256')
    _integer(value['bytes'], f'{field}.bytes')
    if value['format'] != 'npy' or value['order'] != 'C':
        raise ContractError(f'{field}: expected a C-order npy array')
    if not isinstance(value['dtype'], str) or value['dtype'] not in _DTYPES or value['dtype'] not in dtypes:
        raise ContractError(f'{field}.dtype: unsupported numerical dtype')
    if (type(value['shape']) is not list or any(type(n) is not int for n in value['shape'])
            or value['shape'] != shape):
        raise ContractError(f'{field}.shape: expected {shape}')
    size = 1
    for n in shape:
        _integer(n, f'{field}.shape')
        size *= n
    itemsize = {'<f4': 4, '<f8': 8, '<i8': 8, '|b1': 1}[value['dtype']]
    if value['bytes'] < size * itemsize:
        raise ContractError(f'{field}.bytes: smaller than the numerical payload')


def validate_request(document):
    """Validate metadata and return an independent JSON-compatible document.

    A window can represent a row model (context=1) or a sequence model. More general
    extraction/tensor operations retain their A10 contracts until an adapter is added.
    """
    canonical(document)
    _object(document, 'version operation identity runtime method component device input output', 'request')
    if type(document['version']) is not int or document['version'] != VERSION:
        raise ContractError('version: unsupported runtime contract')
    if document['operation'] not in ('fit', 'predict'):
        raise ContractError('operation: expected fit or predict')
    if document['device'] not in ('cpu', 'cuda'):
        raise ContractError('device: expected cpu or cuda; placement resolves the physical UUID')
    _identity(document['identity'])
    _runtime(document['runtime'])
    _component(document['component'], 'component')
    _text(document['method'], 'method')
    data = document['input']
    _object(data, 'dataset_id partition_id view_id scaler_id samples context horizon frequency_ns '
            'timezone targets past_covariates future_covariates target_units target_scale missing arrays', 'input')
    for key in ('dataset_id', 'partition_id', 'view_id'):
        _text(data[key], f'input.{key}')
    for key in ('samples', 'context', 'horizon', 'frequency_ns'):
        _integer(data[key], f'input.{key}')
    _text(data['timezone'], 'input.timezone')
    try:
        ZoneInfo(data['timezone'])
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ContractError('input.timezone: use an IANA timezone; indices must already be UTC ns') from error
    for key in ('targets', 'past_covariates', 'future_covariates'):
        _names(data[key], f'input.{key}', empty=key != 'targets')
    if set(data['targets']) & (set(data['past_covariates']) | set(data['future_covariates'])):
        raise ContractError('input: target names cannot also name covariates')
    if (type(data['target_units']) is not list or len(data['target_units']) != len(data['targets'])):
        raise ContractError('input.target_units: one unit per target')
    for unit in data['target_units']:
        _text(unit, 'input.target_units')
    if data['target_scale'] not in ('physical', 'global_scaled'):
        raise ContractError('input.target_scale: expected physical or global_scaled')
    if data['target_scale'] == 'global_scaled':
        _text(data['scaler_id'], 'input.scaler_id')
    elif data['scaler_id'] is not None:
        _text(data['scaler_id'], 'input.scaler_id')
    if data['missing'] not in ('forbid', 'mask'):
        raise ContractError('input.missing: expected forbid or mask; masked values use finite fillers')
    n, c, h, t = (data[k] for k in ('samples', 'context', 'horizon', 'targets'))
    widths = {'history': [n, c, len(t) + len(data['past_covariates'])],
              'history_index': [n, c], 'forecast_index': [n, h]}
    if data['future_covariates']:
        widths['future'] = [n, h, len(data['future_covariates'])]
    if document['operation'] == 'fit':
        widths['target'] = [n, h, len(t)]
    if data['missing'] == 'mask':
        widths.update({key + '_mask': shape for key, shape in list(widths.items()) if not key.endswith('_index')})
    if type(data['arrays']) is not dict or set(data['arrays']) != set(widths):
        raise ContractError(f'input.arrays: expected {sorted(widths)}')
    for key, shape in widths.items():
        types = {'<i8'} if key.endswith('_index') else {'|b1'} if key.endswith('_mask') else {'<f4', '<f8'}
        _array(data['arrays'][key], shape, types, f'input.arrays.{key}')
    output = document['output']
    _object(output, 'target_scale units inversion quantiles', 'output')
    if output['target_scale'] not in ('physical', data['target_scale']):
        raise ContractError('output.target_scale: no implicit new scaling')
    inversion = 'global_once' if data['target_scale'] == 'global_scaled' and output['target_scale'] == 'physical' else 'none'
    if output['inversion'] != inversion:
        raise ContractError(f'output.inversion: expected {inversion}')
    units = data['target_units'] if output['target_scale'] == 'physical' else ['global_scaled'] * len(t)
    if output['units'] != units:
        raise ContractError(f'output.units: expected {units}')
    levels = output['quantiles']
    if (type(levels) is not list or any(type(q) not in (int, float) or not 0 < q < 1 for q in levels)
            or levels != sorted(set(levels))):
        raise ContractError('output.quantiles: expected unique increasing levels in (0, 1)')
    return deepcopy(document)


def validate_capability(request, capability):
    """Check an adapter declaration separately from format validation/qualification."""
    value = validate_request(request)
    if not isinstance(capability, dict):
        raise ContractError('capability: expected adapter declaration')
    for field, key in (('method', 'methods'), ('operation', 'operations'), ('device', 'devices')):
        declared = capability.get(key)
        _names(declared, f'capability.{key}')
        if value[field] not in declared:
            raise ContractError(f'{field}: unsupported by the selected adapter')
    data = value['input']
    for key in ('past_covariates', 'future_covariates', 'multiple_targets', 'missing_mask', 'quantiles'):
        if key in capability and type(capability[key]) is not bool:
            raise ContractError(f'capability.{key}: expected boolean')
    for key in ('past_covariates', 'future_covariates'):
        if data[key] and not capability.get(key, False):
            raise ContractError(f'input.{key}: unsupported by the selected adapter')
    if len(data['targets']) > 1 and not capability.get('multiple_targets', False):
        raise ContractError('input.targets: adapter supports one target only')
    for key in ('context', 'horizon'):
        limit = capability.get('max_' + key)
        if limit is not None:
            _integer(limit, f'capability.max_{key}')
            if data[key] > limit:
                raise ContractError(f'input.{key}: exceeds adapter limit {limit}')
    if data['missing'] == 'mask' and not capability.get('missing_mask', False):
        raise ContractError('input.missing: adapter does not support masked values')
    if value['output']['quantiles'] and not capability.get('quantiles', False):
        raise ContractError('output.quantiles: unsupported by the selected adapter')
    return value


def validate_result(document, request):
    request = validate_request(request)
    canonical(document)
    _object(document, 'version request_sha256 identity runtime method component arrays target_scale units inversion', 'result')
    if type(document['version']) is not int or document['version'] != VERSION:
        raise ContractError('result.version: unsupported runtime contract')
    if document['request_sha256'] != fingerprint(request):
        raise ContractError('result.request_sha256: result belongs to a different request')
    for key in ('identity', 'runtime', 'method'):
        if document[key] != request[key]:
            raise ContractError(f'result.{key}: mismatched request/attempt/runtime')
    _component(document['component'], 'result.component')
    if request['operation'] == 'predict':
        if document['component'] != request['component']:
            raise ContractError('result.component: prediction changed the selected model')
    else:
        for key in ('adapter_revision', 'parameters', 'checkpoint'):
            if document['component'][key] != request['component'][key]:
                raise ContractError(f'result.component.{key}: changed during fit')
        if not document['component']['artifacts']:
            raise ContractError('result.component.artifacts: fit must publish its model')
    for key in ('target_scale', 'units', 'inversion'):
        if document[key] != request['output'][key]:
            raise ContractError(f'result.{key}: mismatched output semantics')
    data = request['input']
    shape = [data['samples'], data['horizon'], len(data['targets'])]
    names = {'prediction', 'forecast_index'}
    if request['output']['quantiles']:
        names.add('quantiles')
    arrays = document['arrays']
    if type(arrays) is not dict or set(arrays) != names:
        raise ContractError(f'result.arrays: expected {sorted(names)}')
    _array(arrays['prediction'], shape, {'<f4', '<f8'}, 'result.arrays.prediction')
    if arrays['forecast_index'] != data['arrays']['forecast_index']:
        raise ContractError('result.forecast_index: dates/order changed')
    if 'quantiles' in arrays:
        _array(arrays['quantiles'], shape + [len(request['output']['quantiles'])],
               {'<f4', '<f8'}, 'result.arrays.quantiles')
    return deepcopy(document)
