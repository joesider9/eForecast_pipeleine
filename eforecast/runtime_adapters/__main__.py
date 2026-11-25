# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The entry point a model image is run with -- A16-02.

    python -m eforecast.runtime_adapters families
    python -m eforecast.runtime_adapters capability <method>
    python -m eforecast.runtime_adapters run <task directory>

`run` reads request.json, verifies every array against its recorded digest, forecasts, and
publishes result.json.  A failure is published too, as failure.json with a remedy, because
a model image that merely exits non-zero leaves the host guessing which of the request, the
checkpoint, the cache or the credential was wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from eforecast import runtime_contract as contract
from eforecast.runtime_adapters import (CATALOGUE, UnknownMethod, adapter, installed,
                                        methods, served, transport)


class Refused(ValueError):
    """The request is well formed but this image will not serve it."""


def adopt_secrets():
    """Move credentials from the read-only mount into this process's environment.

    The launcher mounts them as files rather than passing `docker run -e`, because an `-e`
    value is visible in `docker inspect` to anyone who can reach the daemon. The model
    packages read environment variables, so the last hop happens here, inside the process
    that needs them, and the values are returned so they can be kept out of what we publish.
    """
    directory = os.environ.get('EFORECAST_SECRETS')
    adopted = []
    if not directory or not os.path.isdir(directory):
        return adopted
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path) or not name.isidentifier():
            continue
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                value = handle.read().strip()
        except OSError:
            continue
        if value:
            os.environ[name] = value
            adopted.append(value)
    return adopted


def _redact(text, values):
    """Keep an adopted credential out of anything this process writes down."""
    for value in values:
        if value and len(value) > 3:
            text = text.replace(value, '[redacted]')
    return text


def _arrays(task_dir, request):
    """Every request array, each checked against the digest the request recorded."""
    loaded = {}
    for name, descriptor in request['input']['arrays'].items():
        loaded[name] = transport.load(task_dir, descriptor, f'input.arrays.{name}')
    return loaded


def run(task_dir):
    """Serve one request and return the published result document."""
    adopt_secrets()
    request = contract.validate_request(transport.read_document(task_dir, transport.REQUEST))
    method = request['method']
    served = adapter(method)
    contract.validate_capability(request, served.capability(method))

    if request['operation'] != 'predict':
        raise Refused(f'operation {request["operation"]}: these are pretrained checkpoints; '
                      'this image predicts and does not fit')
    if request['output']['inversion'] != 'none':
        # the scaler is a host artifact and deliberately does not cross the boundary, so the
        # image cannot invert; asking for scaled output keeps the inversion in one place
        raise Refused('output.inversion global_once: this image holds no scaler. Ask for '
                      'output.target_scale "global_scaled" and invert once on the host.')

    prediction, spread = served.forecast(request, _arrays(task_dir, request))

    arrays = {'prediction': transport.store(task_dir, 'prediction', prediction, dtype='<f8'),
              # echoed unchanged: the contract compares this against the request's own
              # descriptor, so an adapter cannot quietly return a different set of dates
              'forecast_index': request['input']['arrays']['forecast_index']}
    if request['output']['quantiles']:
        arrays['quantiles'] = transport.store(task_dir, 'quantiles', spread, dtype='<f8')

    result = {'version': contract.VERSION,
              'request_sha256': contract.fingerprint(request),
              'identity': request['identity'],
              'runtime': request['runtime'],
              'method': method,
              'component': request['component'],
              'arrays': arrays,
              'target_scale': request['output']['target_scale'],
              'units': request['output']['units'],
              'inversion': request['output']['inversion']}
    contract.validate_result(result, request)
    transport.write_document(task_dir, transport.RESULT, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='eforecast.runtime_adapters',
                                     description='serve one model family through the runtime contract')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('families', help='the methods this image claims')
    declare = commands.add_parser('capability', help='what this image supports for one method')
    declare.add_argument('method')
    serve = commands.add_parser('run', help='serve the request in a task directory')
    serve.add_argument('task_dir')
    arguments = parser.parse_args(argv)

    if arguments.command == 'families':
        print(json.dumps({'served': served(), 'known': methods(),
                          'adapters': {name: installed(name) for name in sorted(CATALOGUE)}},
                         indent=1, sort_keys=True))
        return 0
    if arguments.command == 'capability':
        try:
            print(json.dumps(adapter(arguments.method).capability(arguments.method), indent=1, sort_keys=True))
        except (UnknownMethod, ValueError) as error:
            print(f'{error}', file=sys.stderr)
            return 2
        return 0

    secrets = adopt_secrets()
    try:
        result = run(arguments.task_dir)
    except Exception as error:                      # published, not swallowed: see the docstring
        # a model package can quote the key it was handed; the failure document is read by
        # the host and can reach a support report, so it is redacted before it is written
        failure = {'version': contract.VERSION, 'error': type(error).__name__,
                   'detail': _redact(str(error), secrets),
                   'remedy': _redact(_remedy(error), secrets),
                   'traceback': [_redact(line, secrets)
                                 for line in traceback.format_exc().splitlines()[-12:]]}
        try:
            transport.write_document(arguments.task_dir, transport.FAILURE, failure)
        except OSError:
            pass                                    # an unwritable task directory is the host's finding
        print(_redact(f'{type(error).__name__}: {error}', secrets), file=sys.stderr)
        return 1
    print(json.dumps({'published': transport.RESULT,
                      'prediction': result['arrays']['prediction']['shape']}, indent=1))
    return 0


def _remedy(error):
    """One sentence a colleague can act on, chosen by what actually went wrong."""
    text = str(error).lower()
    if isinstance(error, transport.TransportError):
        return ('The task directory does not hold the arrays the request describes. Re-stage '
                'the request; a digest mismatch means the payload changed after it was written.')
    if isinstance(error, (UnknownMethod, Refused)):
        return str(error)
    if 'connection' in text or 'offline' in text or 'resolve' in text or 'timed out' in text:
        return ('The checkpoint is not in this runtime\'s cache and the image could not reach '
                'Hugging Face. Pre-stage the checkpoint into the mounted cache, or allow the '
                'network for one warm-up run.')
    if 'api key' in text or 'unauthorized' in text or 'token' in text or '401' in text:
        return ('The hosted model needs its credential. Store it once with '
                '`eforecast secret set <name>` and run again; it is injected as an environment '
                'variable and never baked into the image.')
    if isinstance(error, contract.ContractError):
        return 'The request does not satisfy the runtime contract; the detail names the field.'
    return 'Unexpected adapter failure; the traceback lines above belong in a support report.'


if __name__ == '__main__':
    sys.exit(main())
