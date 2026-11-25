# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Complete prediction, combination and operational forecasts, each in a worker of its own.

The inference image consumes the neutral worker envelope; it never imports this
module. Host entry points install a scoped callback, not a global model cache. Nothing is
reserved for the worker: it is started with some threads and, if asked, a card, and what it
left is published here once it has exited and its inputs are found unchanged.
"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import time
import uuid

from eforecast.bounded_pools import capacity
from eforecast.compute_context import using_dataset_backend, using_resident_backend
from eforecast.common_utils import file_stamps
from eforecast.common_utils.stage_context import current_stage
from eforecast.inference_inputs import observations, configuration_state, changed, moved
from eforecast.training.command_line.resource_reporting import ResourceUnavailable
from . import graph_publication, model_compute, run_reporting
from .executors import atomic_json, digest

log = logging.getLogger(__name__)


class InferenceFailure(ResourceUnavailable):
    def __init__(self, state, cleanup, log_path):
        self.state, self.cleanup, self.log_path = state, cleanup, str(log_path)
        super().__init__(f'inference {state}; cleanup {cleanup}; private log: {log_path}')


def input_identity(evidence):
    """Which inputs a forecast was actually made from, as the worker recorded them.

    Not the provider's name and not the issue time: A15-03's first version hashed the NWP model
    labels and the warning text, so corrected weather under the same model name produced the
    same input identity, and an unrelated change of wording produced a different one (R27-16).
    This digests the input records and dataset observations the worker published -- the same
    evidence the host already verifies has not changed since it was prepared.
    """
    # The five numbers of each stamp, as this identity has always been made: what a sync client
    # writes beside a file is no input of a forecast (``file_stamps``).
    return 'inputs-' + digest({'inputs': evidence['online_inputs'],
                               'observations': file_stamps.bare(evidence['observations'])})


def qualified_serving(hold, path, intervals, *, models=None):
    """What the approved bundle must agree to before a routine forecast runs on it.

    Returns the reference's serving section. Every refusal here is a way a forecast would
    otherwise be served by something nobody qualified: a legacy approval with no replay behind
    it, retained bytes that changed, a reference that no longer matches its manifest, a runtime
    whose packages have moved since the replay, or a predictor path or interval coverage that
    was never run. R27-03 is the reason these are checked here rather than trusted: a stored
    number is not evidence that a model still produces it.

    Separated from the host path so each refusal can be exercised directly. The host has one
    caller and no branch of its own.
    """
    from eforecast import model_bundle
    from eforecast.bundle_serving import identity, reference_digest, runtime_identity
    models = models or hold.get('root')
    if not hold.get('qualified'):
        raise ValueError(f'legacy approval retained for the models at {models}; capture and '
                         'approve a serving reference before routine publication')
    manifest = model_bundle.read(hold['root'])
    if model_bundle.missing(manifest, hold['root']) or identity(manifest) != hold['bundle']:
        raise ValueError('the pinned bundle changed; restore its retained bytes before forecasting')
    fixture = model_bundle.read_fixture(hold['root'])
    if not fixture or reference_digest(fixture) != manifest.get('reference_sha256'):
        raise ValueError('the pinned reference changed; restore the qualified bundle')
    serving = fixture.get('serving') or {}
    if serving.get('runtime') != runtime_identity():
        raise ValueError('serving runtime changed; requalify before routine publication')
    # A caller that names no predictor gets the one the bundle replayed, not a default that
    # happens to be spelled the same: comparing two defaults and then computing a third is how
    # an unqualified predictor reaches a published forecast. The host sets the resolved path on
    # the request, so the check and the computation cannot disagree.
    if path is not None and list(path) != serving.get('path'):
        raise ValueError(f'this predictor path has not been qualified in the approved bundle: '
                         f'it replayed {serving.get("path")}')
    if not set(intervals or []).issubset(serving.get('intervals') or []):
        raise ValueError('the requested intervals have not been qualified in the approved '
                         f'bundle: it replayed {serving.get("intervals") or "none"}')
    return serving


@contextmanager
def hosted(workers=None):
    """SDK boundary: all model/data work occurs in one-shot workers, each a process of its own."""
    with (using_dataset_backend(workers) if workers is not None else nullcontext()):
        with using_resident_backend('inference', dispatch):
            yield


def load(config):
    from eforecast_online.forecast import load_static_data
    path = Path(config) if isinstance(config, (str, os.PathLike)) and Path(config).exists() else None
    before = configuration_state(path) if path is not None else {}
    static = load_static_data(config)
    if path is not None and changed(before, configuration_state(path)):
        raise ResourceUnavailable('inference configuration changed during initialization')
    return static, before


def metadata_sites(static, configs):
    """Modern GRAPH references are JSON; legacy discovery happens inside a worker."""
    from eforecast.common_utils import artifact
    from eforecast.common_utils.graph_bundle import bundle_root
    configs = dict(configs) if isinstance(configs, dict) else observations({}, configs)
    if static.get('_graph_serving'):
        graph = static['_graph_serving']
        return {**graph['sites'], static['_id']: static}, configs, sorted({v['path'] for v in graph['groups'].values()})
    statics, bundles = {static['_id']: static}, set()
    for folder in Path(static['path_model']).glob('global/GRAPH/test_*'):
        manifest = artifact.read(folder) or {}
        if artifact.inspect(folder)[0] == 'rejected':
            continue
        reference = manifest.get('bundle') or {}
        if not isinstance(reference, dict) or not reference.get('path'):
            continue
        bundle = bundle_root(reference['path'])
        bundles.add(str(bundle))
        meta = json.loads((bundle / 'meta.json').read_text())
        for name, config in meta.get('configs', {}).items():
            if name not in statics:
                sd, paths = load(config)
                statics[name] = sd
                configs.update(paths)
    return statics, configs, sorted(bundles)


def check_inputs(evidence):
    problem = moved('inference inputs changed; private output was not published', evidence['observations'],
                    observations(evidence['statics'], evidence['configs'], evidence['bundles']))
    if problem:
        raise ResourceUnavailable(problem)


def copy_model(result, root, evidence, static, guard):
    """Private stage handoff, or a rollback-protected direct cache publication."""
    from eforecast.common_utils.lineage import digest_file
    moves, backups = [], []
    try:
        for relative in result['value']['changed']:
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe inference output path')
            source = root / 'artifacts/model' / path
            record = result['files'][source.relative_to(root).as_posix()]
            target = Path(static['path_model']) / path
            for ancestor in [target.parent, *target.parent.parents]:
                if ancestor == Path(static['path_model']):
                    break
                if ancestor.is_symlink():
                    raise ValueError('inference output traverses an immutable directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name('.inference-copy-' + uuid.uuid4().hex)
            shutil.copyfile(source, temporary)
            if digest_file(str(temporary)) != record['sha256']:
                raise ValueError('inference output differs from accepted bytes')
            moves.append((temporary, target))
        guard()
        # Keys follow values. The enclosing stage writes its receipt last; a standalone
        # caller's caches use these backups.
        for temporary, target in sorted(moves, key=lambda pair: str(pair[1]).endswith('.key.json')):
            backup = target.with_name('.inference-backup-' + uuid.uuid4().hex)
            existed = target.exists() or target.is_symlink()
            if existed:
                os.replace(target, backup)
            backups.append((backup, target, existed))
            os.replace(temporary, target)
        for relative in result['value'].get('deleted', []):
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe inference cache deletion')
            target = Path(static['path_model']) / path
            if target.exists() or target.is_symlink():
                backup = target.with_name('.inference-backup-' + uuid.uuid4().hex)
                os.replace(target, backup)
                backups.append((backup, target, True))
    except BaseException:
        for backup, target, existed in reversed(backups):
            target.unlink(missing_ok=True)
            if existed:
                os.replace(backup, target)
        raise
    finally:
        for temporary, _ in moves:
            temporary.unlink(missing_ok=True)
        for backup, _, _ in backups:
            backup.unlink(missing_ok=True)


def publish_service(result, root, evidence, folder, guard):
    """Retain verified bytes, then record the issue in SQL, then project the SQL pointer.

    SQL may commit before a filesystem outage. The pointer includes the immutable payload
    path/hash, so export_latest repairs that projection without rerunning a forecast. No worker
    writes this database, and the forecast store itself decides whether this issue supersedes
    what is published.
    """
    from eforecast.common_utils.lineage import digest_file
    from . import forecast_publication as publication
    folder = Path(folder).absolute()
    launch = result['context']['launch_id']
    generation = folder / '.inference-generations' / launch
    generation.mkdir(parents=True, exist_ok=True)
    records = {}
    for source in (root / 'artifacts/service').iterdir():
        record = result['files'][source.relative_to(root).as_posix()]
        target = generation / source.name
        shutil.copyfile(source, target)
        if digest_file(str(target)) != record['sha256']:
            raise ValueError('operational output changed during handoff')
        with target.open('rb') as stream:
            os.fsync(stream.fileno())
        records[source.name] = record
    body = {'version': 3, 'launch_id': launch,
            'context': result['context'], 'files': records, 'dependencies': evidence['dependencies'],
            'input_observations': evidence['observations'], 'request': evidence['request'],
            'online_inputs': evidence['online_inputs']}
    atomic_json(generation / 'inference_manifest.json', body)
    graph_publication.sync_directory(generation)
    graph_publication.sync_directory(generation.parent)
    prepared = observations({}, [generation / name for name in records])
    packet = json.loads((generation / 'publication.json').read_text())
    status = json.loads((generation / 'status.json').read_text())
    hold = packet.pop('hold')
    if not hold:
        raise ValueError('routine publication has no pinned approved bundle')
    durable = publication.store()
    # SQL can now retain a reference to these bytes, including if export later fails.
    guard()
    if changed(prepared, observations({}, prepared)):
        raise ValueError('prepared operational output changed before publication')
    verdict = publication.publish(durable, hold['case'], **packet, bundle=hold['bundle'],
        input_version=input_identity(evidence),
        hold=hold, acceptance=lambda: True,
        payload_ref={'path': str(generation / 'forecast.csv'),
                     'sha256': records['forecast.csv']['sha256'], 'bytes': records['forecast.csv']['bytes'],
                     'launch_id': launch})
    status['publication'].update(verdict, recorded=True)
    status['state'] = 'ok' if verdict['published'] else 'rejected'
    status['files'] = [str(generation / 'forecast.csv')]
    atomic_json(generation / 'accepted_status.json', status)
    atomic_json(generation / 'commit.json', {'version': 2, 'manifest_id': digest(body),
                                            'issue_id': verdict['issue_id']})
    # Projection is recoverable and serializes with publication. Failure never falls back to
    # exporting this attempt: a newer issue might have won in the meantime.
    try:
        publication.export_latest(durable, hold['case'], folder)
    except Exception as error:
        status['state'] = 'publication_pending'
        status['warnings'].append(f'Accepted forecast retained; export needs retry: {error}')
    atomic_json(folder / 'status.json', status)
    return status


def publish_csv(result, root, evidence, target, guard):
    from eforecast.common_utils.lineage import digest_file
    target = Path(target).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    ident = uuid.uuid4().hex
    temporary = target.with_name('.inference-csv-' + ident)
    companion = Path(str(target) + '.manifest.json')
    temporary_manifest = target.with_name('.inference-manifest-' + ident)
    record = result['files']['artifacts/forecast.csv']
    backups, published = [], False
    try:
        shutil.copyfile(root / 'artifacts/forecast.csv', temporary)
        if digest_file(str(temporary)) != record['sha256']:
            raise ValueError('forecast CSV changed during handoff')
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        manifest = {'version': 1, 'kind': 'forecast_csv', 'file': target.name, 'payload': record,
                    'context': result['context'], 'dependencies': evidence['dependencies'], 'request': evidence['request'],
                    'online_inputs': evidence['online_inputs']}
        atomic_json(temporary_manifest, manifest)
        prepared = observations({}, [temporary, temporary_manifest])
        guard()
        if changed(prepared, observations({}, prepared)):
            raise ValueError('prepared forecast CSV changed before publication')
        for source, destination in ((temporary, target), (temporary_manifest, companion)):
            backup = destination.with_name('.inference-previous-' + uuid.uuid4().hex)
            existed = destination.exists() or destination.is_symlink()
            if existed:
                os.replace(destination, backup)
            backups.append((backup, destination, existed))
            os.replace(source, destination)
        published = True  # companion last: from here the new forecast stands
        graph_publication.sync_directory(target.parent)
    except BaseException:
        if not published:
            for backup, destination, existed in reversed(backups):
                destination.unlink(missing_ok=True)
                if existed:
                    os.replace(backup, destination)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        for backup, _, _ in backups:
            backup.unlink(missing_ok=True)


def _run(operation, static, arguments, *, configs=(), plan=None, workers=None, observation=None,
         response_path=None):
    observation = observation if observation is not None else {}
    observation['cleanup'] = 'not_started'
    static, arguments = deepcopy(static), deepcopy(arguments)
    requested = os.environ.get('EFORECAST_INFERENCE_DEVICE', 'cpu')
    workers = workers or model_compute.model_workers(requested)
    statics, configs, bundles = metadata_sites(static, configs)
    device = model_compute.device_for(requested)
    if device is None and str(requested).lower() not in {'cpu', 'none'}:
        log.warning('inference asks for device %r and no visible GPU answers it; it runs on the CPU', requested)
    effective = requested if device else 'cpu'
    # A prediction is made with two threads, wherever it is made: a network's numbers depend on
    # the count. Predicting every trial of a case is the one operation with many independent
    # pieces, so on the CPU it is also given the case's CPUs and predicts its trials side by
    # side, each with those two threads (ensemble.predict_trials). EFORECAST_INFERENCE_THREADS
    # is the count said outright: one worker with that many threads, one trial at a time.
    explicit = os.environ.get('EFORECAST_INFERENCE_THREADS')
    wanted = int(explicit) if explicit else 2
    threads = max(1, min(capacity(), wanted if wanted > 0 else 2))
    granted = threads
    if operation in {'predict', 'trials'} and device is None and not explicit:
        granted = max(threads, min(capacity(), int(static.get('n_cpus') or 0) or capacity()))
    beside = granted >= 2 * threads
    # The worker runs on the CPUs. The foundation models of the case that ask for a card are
    # what a card does fifty times faster, so the worker is told which cards answer them and
    # predicts those trials in readers of their own, one a card, beside the others.
    cards = model_compute.foundation_cards(static) if beside and operation in {'predict', 'trials'} else {}
    if cards:
        arguments['cards'] = cards
    stage = current_stage.get()
    folder = arguments.get('out_dir') or str(Path(static['path_group']) / 'predictions') if operation == 'service' else static['path_model']
    resources = {Path(sd['path_model']) / '.stage-artifacts' for sd in statics.values()}
    if operation == 'service':
        resources.add(Path(folder) / '.inference-writer')
    if operation == 'forecast_file' and arguments.get('file'):
        resources.add(Path(arguments['file']).absolute().parent / '.inference-writer')
    # A stage already holds its case lock. Standalone work uses the same lock family;
    # it can never race a graph exporter or ordinary stage writer of these cases.
    owners = nullcontext(None) if stage is not None else graph_publication.ownership(resources)
    with owners as owner:
        if isinstance(configs, dict) and changed(configs, observations({}, configs)):
            raise ResourceUnavailable('inference configuration changed after initialization')
        before = observations(statics, configs, bundles)
        values = {'operation': operation, 'statics': statics, 'main': static['_id'], 'arguments': arguments,
                  'configs': configs, 'bundles': bundles, 'observations': before, 'output_folder': str(folder), 'plan': plan}
        ident = uuid.uuid4().hex
        ctx = run_reporting.context()
        task = 'inference:' + hashlib.sha256(str(Path(static['path_model']).absolute()).encode()).hexdigest() + ':' + operation
        request = workers.request(ctx['run_id'] if ctx else 'inference:' + ident, task, ident, granted, device=device)
        handle = None
        try:
            run_reporting.check_cancelled()
            if stage is not None:
                stage.check()
            if owner is not None:
                owner.check()
            handle = workers.prepare(request,
                'eforecast.combine_predictions.combination_worker:execute' if plan is not None else
                'eforecast.combine_predictions.inference_worker:execute', {'operation': operation},
                input_files={'inference.pickle': pickle.dumps(values, protocol=5)},
                execution={'device': effective, 'threads': threads})
            observation['cleanup'] = 'pending'
            log.info('inference %s starts: device %s (asked %s), %d thread(s)%s%s', operation, effective, requested,
                     threads, f', its trials side by side on {granted} CPUs' if beside else '',
                     f"; {', '.join(cards)} on {len({card for named in cards.values() for card in named})} card(s)"
                     if cards else '')
            if operation == 'service':
                from dataclasses import asdict
                from . import model_approval, forecast_publication
                # The hold on the approved bundle is released when this launch is seen to have
                # ended, so the hold remembers which launch that is.
                model_approval.attach(forecast_publication.store(), static['_approved_hold'],
                    {'executor': asdict(handle), 'forecast_attempt': static['_approved_hold']['attempt_id']})
            result = workers.wait(handle, timeout=None)
            observation['cleanup'] = 'complete'
            log_path = workers.executor._spec(handle)[1]['log']
            observation['log'] = str(log_path)
            if result['state'] != 'success':
                raise InferenceFailure(result['state'], 'complete', log_path)
            for line in (result.get('value') or {}).get('said') or ():
                log.warning('%s', line)     # what the worker only wrote to a log that goes with its folder
            root = workers.root(handle)
            import joblib
            evidence = joblib.load(root / 'artifacts/evidence.pickle')
            def guard():
                run_reporting.check_cancelled()
                if owner is not None:
                    owner.check()
                if stage is not None:
                    stage.check()
                check_inputs(evidence)
            guard()
            if operation == 'service':
                value = publish_service(result, root, evidence, folder, guard)
                workers.discard_result(handle)
                return value
            elif operation == 'forecast_file':
                if arguments.get('file'):
                    publish_csv(result, root, evidence, arguments['file'], guard)
                else:
                    import sys
                    with (root / 'artifacts/forecast.txt').open() as stream:
                        shutil.copyfileobj(stream, sys.stdout)
            elif not arguments.get('online') and operation not in {'online', 'forecast'}:
                copy_model(result, root, evidence, static, guard)
            if plan is not None:
                stage.validated_payloads = evidence['validated']
                value = None   # keep every large frame and deserialization in the worker
            elif response_path is not None:
                # A bridge is a transport, not another owner of decoded arrays.
                # Stream the worker's serialized result; only the actual SDK caller
                # materializes its return frames.
                from eforecast.common_utils.lineage import digest_file
                target = Path(response_path).absolute()
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name('.inference-response-' + ident)
                try:
                    shutil.copyfile(root / 'artifacts/result.pickle', temporary)
                    if digest_file(str(temporary)) != result['files']['artifacts/result.pickle']['sha256']:
                        raise ValueError('serialized inference result changed during transport')
                    prepared = observations({}, [temporary])
                    guard()
                    if changed(prepared, observations({}, prepared)):
                        raise ValueError('serialized inference response changed before handoff')
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                value = None
            else:
                value = joblib.load(root / 'artifacts/result.pickle')
            workers.discard_result(handle)
            return value
        except BaseException:
            if handle is not None:
                workers.cancel(handle, force=True)
                try:
                    workers.wait(handle, timeout=30)
                    observation['cleanup'] = 'complete'
                except (Exception, KeyboardInterrupt):
                    observation['cleanup'] = 'unknown'
                    log.warning('the inference worker was told to stop and has not been seen to exit')
            raise


def run_stage(key, static, args, plan, *, workers=None):
    arguments = {'refit': args.refit}
    if key == 'predict':
        arguments['selected_only'] = not bool(set(args.stage) & {'predict', 'select', 'all'})
    if key == 'combine':
        arguments['fit'] = bool(set(args.stage) & {'combine', 'all'})
    return _run(key, static, arguments, plan=plan, workers=workers)


def dispatch(operation, *, static=None, config=None, _response_path=None, **arguments):
    if operation == 'bundle_capture':
        from .bundle_compute import capture
        return capture(static, **arguments)
    if operation == 'bundle_reproduce':
        from .bundle_compute import validate
        return validate(**arguments, _action='reproduce')
    if operation == 'bundle_validate':
        from .bundle_compute import validate
        return validate(**arguments)
    configs = {}
    if static is None:
        static, configs = load(config)
    observation, hold, durable = {}, None, None
    try:
        if operation == 'service':
            from eforecast_online.service import issue_time, forecast_stamps
            from . import forecast_publication as publication
            from .serving_scope import for_config
            from eforecast.bundle_serving import static_for
            durable = publication.store()
            scope = for_config(durable, config, static)
            from .model_approval import ApprovalError, reconcile_holds
            from .executors import LocalExecutor
            from .run_model import ExecutorHandle
            def settled(execution):
                # A hold taken before launches were remembered names an allocation in a store
                # that no longer exists; nothing can vouch for its worker, and the issue
                # record below is then the whole of the evidence.
                if 'executor' in execution:
                    if LocalExecutor().discover(ExecutorHandle(**execution['executor'])).state != 'finished':
                        return False
                # The worker ends before the host publishes. An issue record is also needed
                # before another submitter may retire this hold; otherwise copying may still run.
                with durable._transaction() as cursor:
                    durable._execute(cursor, '''SELECT issue_id FROM eforecast_forecast_issues
                        WHERE namespace=? AND case_id=? AND attempt=? LIMIT 1''',
                        (durable.namespace,scope['case_id'],execution.get('forecast_attempt')))
                    return cursor.fetchone() is not None
            reconcile_holds(durable, scope['case_id'], settled)
            static['_serving_scope'] = scope
            issue = issue_time(static, arguments.get('at'))
            attempt = uuid.uuid4().hex
            try:
                hold = publication.begin(durable, scope['case_id'], f'{scope["case_id"]}:{issue}',
                                         attempt_id=attempt)
            except ApprovalError as error:
                # Nothing is approved for this case. The durable identity is opaque, so an
                # operator needs the two things they can act on: which model directory this
                # forecast serves, and what to approve.
                raise ValueError(
                    f'{error} -- this forecast serves the models at {static["path_model"]} '
                    f'(case {scope["case_id"]}). Approve a bundle for that case with '
                    f'"eforecast approve <case> <bundle folder>", or run a preview.') from error
            except Exception as error:
                # A store or pin failure is not "nobody approved this", and must not be
                # answered with advice to approve something: the authority could not be read,
                # so the published result stands and the reason is reported as it is.
                raise ValueError(
                    f'the approval for the models at {static["path_model"]} '
                    f'(case {scope["case_id"]}) could not be read, so this forecast cannot '
                    f'publish and the last authoritative result stands: {error}') from error
            serving = qualified_serving(
                hold, arguments.get('path'), arguments.get('intervals'),
                models=f'{static["path_model"]} (case {scope["case_id"]})')
            arguments['path'] = tuple(serving['path'])   # what was replayed is what is served
            arguments['out_dir'] = arguments.get('out_dir') or str(Path(static['path_group']) / 'predictions')
            static = static_for(hold['root'], live=static)
            static.update(_approved_hold=hold, _defer_publication=True, _serving_scope=scope)
            arguments['at'] = str(issue)
            arguments['dates'] = forecast_stamps(static, issue, arguments.get('days', 1), arguments.get('back', 0))
        return _run(operation, static, arguments, configs=configs, observation=observation,
                    response_path=_response_path)
    except Exception as error:
        if operation != 'service':
            raise
        if durable is not None and hold is not None:
            try:
                from eforecast_online.service import issued_at
                publication.publish(durable, hold['case'], issued_at=issued_at(static, issue),
                    bundle=hold['bundle'], input_version='unavailable:' + hold['attempt_id'],
                    attempt=hold['attempt_id'], summary={'rows': 0, 'values': []}, outcome='failed')
            except Exception:
                log.exception('failed operational attempt could not be recorded')
        return {'state': 'error', 'outcome': getattr(error, 'state', 'failure'),
                'cleanup': observation.get('cleanup', 'unknown'), 'rows': 0,
                'files': [], 'error': f'{type(error).__name__}: {error}', 'log': observation.get('log')}
    finally:
        if hold is not None and durable is not None:
            # Keep an unknown/live worker protected; its recovery owner must settle it first.
            if observation.get('cleanup', 'not_started') in {'not_started', 'complete'}:
                try:
                    publication.finish(durable, hold['case'], hold['forecast_id'],
                                       attempt_id=hold['attempt_id'], hold_token=hold['hold_token'])
                except Exception:
                    log.exception('forecast hold retained: conditional release could not be confirmed')


def main(argv=None):
    import signal
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    from eforecast_online.cli import main as online_main
    def stop(*_):
        raise KeyboardInterrupt('inference host stopped')
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        with hosted():
            if len(argv) == 2 and argv[0] == '--request':
                logging.basicConfig(level=os.environ.get('EFORECAST_INFERENCE_LOG_LEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S')
                request = Path(argv[1])
                try:
                    with request.open('rb') as stream:
                        document = pickle.load(stream)
                    response = request.parent / 'response.pickle'
                    value = dispatch(document['operation'], **{**document['arguments'], '_response_path': str(response)})
                    if not response.is_file():
                        response.write_bytes(pickle.dumps(value, protocol=5))
                    return 0
                except Exception as error:
                    (request.parent / 'error.txt').write_text(f'{type(error).__name__}: {error}')
                    return 1
            return online_main(argv)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
