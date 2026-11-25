# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Maintainer-owned runtime definitions and non-destructive legacy resolution.

The registry describes approved routes; it is not a qualification certificate. Local
definitions name the existing interpreter; isolated model definitions name an immutable
image. New submissions check its source compatibility before freezing the selection.
No model package, configuration loader or web framework is imported here.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys

from .runtime_contract import ContractError, fingerprint

_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
_IMAGE = re.compile(r'(?:[^\s]+@)?sha256:[0-9a-f]{64}\Z')
MAX_REGISTRY_BYTES = 1024 * 1024


class RuntimeUnavailable(ContractError):
    pass


def effective_python(environment=None):
    env = os.environ if environment is None else environment
    return env.get('EFORECAST_PYTHON') or sys.executable


def selections(methods):
    if not isinstance(methods, dict):
        raise ContractError('methods: expected a mapping')
    value = methods.get('runtime_ids', {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ContractError('methods.runtime_ids: expected a method-to-runtime mapping')
    for key, runtime in value.items():
        if (not isinstance(key, str) or not _ID.fullmatch(key)
                or not isinstance(runtime, str) or not _ID.fullmatch(runtime)):
            raise ContractError('methods.runtime_ids: use named method/runtime ids, not paths')
    result = dict(value)
    params = methods.get('method_params') or methods
    for method, parameters in params.items():
        if isinstance(parameters, dict) and parameters.get('runtime_id') and method not in result:
            result[method] = parameters['runtime_id']
    # These newly introduced families have no native implementation. Freeze their route
    # even when the caller accepted the image default instead of writing runtime_ids.
    for method, runtime in {'TimeGPT': 'foundation', 'Timestral': 'foundation',
                            'T0': 'foundation', 'Lag-Llama': 'lag_llama'}.items():
        enabled = any((methods.get(key) or {}).get(method) for key in ('methods_cpu', 'global_methods'))
        if enabled and method not in result and 'default' not in result:
            result[method] = runtime
    for method, runtime in result.items():
        if (not isinstance(method, str) or not _ID.fullmatch(method)
                or not isinstance(runtime, str) or not _ID.fullmatch(runtime)):
            raise ContractError('runtime selection: use named method/runtime ids, not paths')
    return result


def validate_definition(value):
    if not isinstance(value, dict) or set(value) != {'id', 'kind', 'locator', 'methods', 'devices', 'contract'}:
        raise ContractError('runtime definition: expected id, kind, locator, methods, devices, contract')
    if not isinstance(value['id'], str) or not _ID.fullmatch(value['id']):
        raise ContractError('runtime.id: invalid identifier')
    if type(value['contract']) is not int or value['contract'] != 1:
        raise ContractError('runtime.contract: unsupported version')
    if value['kind'] not in ('local', 'container') or not isinstance(value['locator'], str):
        raise ContractError('runtime.kind/locator: expected local interpreter or pinned container')
    locator = value['locator']
    if value['kind'] == 'local' and (not Path(locator).is_absolute() or '\x00' in locator):
        raise ContractError('runtime.locator: local interpreter must be an absolute filename')
    if value['kind'] == 'container' and not _IMAGE.fullmatch(locator):
        raise ContractError('runtime.locator: container images require an immutable sha256 digest')
    methods, devices = value['methods'], value['devices']
    if (not isinstance(methods, list) or not methods or
            any(not isinstance(m, str) or (m != '*' and not _ID.fullmatch(m)) for m in methods)
            or len(set(methods)) != len(methods)):
        raise ContractError('runtime.methods: expected unique method identifiers')
    if (not isinstance(devices, list) or not devices or
            any(d not in ('cpu', 'cuda') for d in devices) or len(set(devices)) != len(devices)):
        raise ContractError('runtime.devices: expected cpu and/or cuda')
    return deepcopy(value)


def registry(environment=None, *, python=None):
    env = os.environ if environment is None else environment
    current = python or effective_python(env)
    native = validate_definition({'id': 'native', 'kind': 'local', 'locator': os.path.abspath(current),
                                  'methods': ['*'], 'devices': ['cpu', 'cuda'], 'contract': 1})
    result = {'native': native}
    path = env.get('EFORECAST_RUNTIME_REGISTRY')
    if path:
        try:
            with open(path, 'rb') as stream:
                raw = stream.read(MAX_REGISTRY_BYTES + 1)
            if len(raw) > MAX_REGISTRY_BYTES:
                raise ContractError('runtime registry exceeds 1 MiB')
            document = json.loads(raw)
        except (OSError, ValueError) as error:
            raise ContractError(f'runtime registry cannot be read: {error}') from error
        if (not isinstance(document, dict) or set(document) != {'version', 'runtimes'}
                or type(document['version']) is not int or document['version'] != 1
                or not isinstance(document['runtimes'], list)):
            raise ContractError('runtime registry: expected version 1 and runtimes list')
        for entry in document['runtimes']:
            item = validate_definition(entry)
            if item['id'] in result:
                raise ContractError(f'runtime registry: duplicate/reserved id {item["id"]}')
            result[item['id']] = item
    return result


def resolve(methods, environment=None, *, python=None):
    selected = selections(methods)
    # A broken optional registry must not change execution of an existing unconfigured case.
    env = dict(os.environ if environment is None else environment)
    if not selected:
        env.pop('EFORECAST_RUNTIME_REGISTRY', None)
    available = registry(env, python=python)
    missing = set(selected.values()) - set(available)
    if missing & {'foundation', 'lag_llama', 'lag-llama'}:
        from eforecast.runtime_launcher import registry_entries
        for item in registry_entries()['runtimes']:
            available.setdefault(item['id'], item)
            if item['id'] == 'lag_llama':
                available.setdefault('lag-llama', dict(item, id='lag-llama'))
    definitions = {'native': available['native']}
    for method, runtime_id in selected.items():
        if runtime_id not in available:
            raise RuntimeUnavailable(f'methods.runtime_ids.{method}: runtime {runtime_id!r} is not registered')
        item = available[runtime_id]
        if method == 'default' and '*' not in item['methods']:
            raise RuntimeUnavailable('default runtime must cover all methods; select a family-specific route instead')
        if method != 'default' and '*' not in item['methods'] and method not in item['methods']:
            raise RuntimeUnavailable(f'runtime {runtime_id!r} does not support method {method}')
        definitions[runtime_id] = item
    body = {'version': 1, 'selection': selected, 'definitions': definitions}
    return dict(body, revision=fingerprint(body))


def validate_binding(value):
    if not isinstance(value, dict) or set(value) != {'version', 'selection', 'definitions', 'revision'}:
        raise ContractError('runtime binding: invalid pinned document')
    if type(value['version']) is not int or value['version'] != 1:
        raise ContractError('runtime binding: unsupported version')
    selected = selections({'runtime_ids': value['selection']})
    definitions = value['definitions']
    if not isinstance(definitions, dict) or set(definitions) != {'native', *selected.values()}:
        raise ContractError('runtime binding: missing or unrelated definitions')
    for key, definition in definitions.items():
        item = validate_definition(definition)
        if item['id'] != key:
            raise ContractError('runtime binding: definition id mismatch')
    if definitions['native']['kind'] != 'local':
        raise ContractError('runtime binding: native must be local')
    for method, key in selected.items():
        supported = definitions[key]['methods']
        if '*' not in supported and (method == 'default' or method not in supported):
            raise ContractError('runtime binding: method not supported by definition')
    if value['revision'] != fingerprint({k: v for k, v in value.items() if k != 'revision'}):
        raise ContractError('runtime binding: digest mismatch')
    return deepcopy(value)


def require_available(binding, *, python=None, check_source=False):
    binding = validate_binding(binding)
    current = os.path.abspath(python or effective_python())
    # Comparing the real binary would erase virtualenv identity through symlinks.
    for runtime_id in set(binding['selection'].values()):
        item = binding['definitions'][runtime_id]
        if item['kind'] == 'container':
            from eforecast.runtime_launcher import CASE_METHODS
            from eforecast import image_runtime
            if not item['methods'] or '*' in item['methods'] or any(m not in CASE_METHODS for m in item['methods']):
                raise RuntimeUnavailable(f'runtime {runtime_id!r}: no isolated adapter for its methods')
            if image_runtime.image_id(item['locator']) is None:
                raise RuntimeUnavailable(f'runtime {runtime_id!r}: pinned image is unavailable; no fallback was applied')
            if check_source:
                from eforecast import image_release
                from eforecast.runtime_adapters import adapter_name
                for adapter in {adapter_name(CASE_METHODS[m]) for m in item['methods']}:
                    expected = image_release.source_digests(adapter)
                    observed = image_release.image_sources(image_runtime.image_id(item['locator']),
                                                           tuple(sorted(expected)), '/opt/eforecast')
                    if observed != expected:
                        raise RuntimeUnavailable(f'runtime {runtime_id!r}: image source is stale or unverifiable; '
                                                 'rebuild and qualify this image before submitting new work')
            continue
        if item['kind'] != 'local' or os.path.abspath(item['locator']) != current:
            raise RuntimeUnavailable(f'runtime {runtime_id!r} needs an isolated executor (A16-02/03); '
                                     'select native for the existing interpreter; no fallback was applied')
        if set(item['devices']) != {'cpu', 'cuda'}:
            raise RuntimeUnavailable(f'runtime {runtime_id!r} restricts devices; device-specific routing needs '
                                     'the A16-02/03 executor; no fallback was applied')
    return binding


def runtime_for(binding, method):
    """Portable envelope reference; a component pins its own definition, not every sibling."""
    binding = validate_binding(binding)
    key = binding['selection'].get(method, binding['selection'].get('default', 'native'))
    return {'id': key, 'revision': fingerprint(binding['definitions'][key])}


def bind_static(static, environment=None, *, binding=None):
    """Only an explicit new selection changes artifact lineage; legacy dictionaries stay intact."""
    selected = selections(static)
    if binding is None:
        binding = static.get('_runtime_binding')
    accepted = validate_binding(binding) if binding is not None else resolve(static, environment)
    if accepted['selection'] != selected:
        raise ContractError('runtime binding: accepted selection differs from effective configuration')
    result = deepcopy(static)
    if selected:
        result['_runtime_binding'] = accepted
    return result, accepted


def execution_binding(static):
    """Use frozen definitions even if the registry has changed or disappeared since submission."""
    if not selections(static):
        return None
    value = static.get('_runtime_binding') or resolve(static, python=sys.executable)
    binding = require_available(value, python=sys.executable)
    if binding['selection'] != selections(static):
        raise ContractError('runtime binding: changed selection after acceptance')
    return binding
