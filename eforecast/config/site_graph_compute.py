# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Host adapter: multi-site metadata, one worker and guarded publication."""
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
import uuid

from eforecast.bounded_pools import capacity
from eforecast.compute_context import install_resident_backend
from eforecast.common_utils import file_stamps
from eforecast.common_utils.graph_bundle import bundle_root, GENERATIONS
from eforecast.training.command_line.resource_reporting import ResourceUnavailable
from . import graph_publication, model_compute, run_reporting

log = logging.getLogger(__name__)


def _stat(path):
    """A stamp that tells a file whose metadata moved from one that was rewritten (``file_stamps``)."""
    return file_stamps.observe(path)


def initialise(config):
    from eforecast.init.initialize import initializer
    before = _stat(config)
    static = initializer(config)
    if not file_stamps.same(before, _stat(config)):
        raise ResourceUnavailable('site configuration changed during initialization; retry with a stable case')
    static['_graph_config'] = str(Path(config).resolve())
    static['_graph_config_stat'] = before
    return static


def observations(statics, folder=None):
    """Stat-only before/after guard: no arrays or model weights enter the host."""
    paths = set()
    for static in statics.values():
        config = static.get('_graph_config')
        if '_graph_config_stat' in static and not file_stamps.same(static['_graph_config_stat'], _stat(config)):
            raise ResourceUnavailable('site configuration changed after initialization; retry with the current case')
        base = Path(static['path_model'])
        paths.update(p for p in Path(static['path_data']).glob('*') if p.is_file())
        paths.update(base / p for p in ('scalers.pickle', 'imputers.pickle', 'transformers.pickle',
                                       'partition.json', 'global/cv_mask.pickle'))
        paths.update(Path(p) for p in (static.get('_graph_config'), static.get('filename')) if p)
    if folder is not None:
        paths.update(bundle_root(folder) / p for p in ('site_graph.pt', 'meta.json', 'artifact_manifest.json'))
    return {str(path): file_stamps.observe(path) for path in sorted(paths)}


def check_observations(statics, folder, before):
    moved = file_stamps.differing(before, observations(statics, folder))
    if moved:
        raise ResourceUnavailable('site-graph inputs changed; private output was not published: '
                                  + file_stamps.named(moved))


def threads_for(statics, options):
    """The threads a site-graph worker is started with: what was asked, else the cases' own."""
    wanted = int(options.get('threads') or max([int(sd.get('n_cpus') or sd.get('n_jobs') or 1) for sd in statics.values()] + [1]))
    return max(1, min(capacity(), wanted if wanted > 0 else capacity()))


def resources(statics, out, destinations=(), csv=None):
    paths = {Path(sd['path_model']).resolve() / '.stage-artifacts' for sd in statics.values()}
    group = Path(out).resolve()
    if group.name == 'bundle' and group.parent.parent.name == GENERATIONS:
        group = group.parent.parent.parent
    paths.add(group / '.stage-artifacts')
    paths.update(Path(p).resolve().parents[2] / '.stage-artifacts' for p in destinations)
    if csv:
        prefix = Path(csv).resolve()
        paths.add(prefix.parent / ('.graph-output-' + hashlib.sha256(str(prefix).encode()).hexdigest()[:20]))
    return paths


def _run(operation, *, folder, statics=None, options=None, dates=None, data=None, idx=None,
         build_datasets=True, inverse=False, site=None, device='cpu', workers=None, destinations=None):
    options, statics = dict(options or {}), dict(statics or {})
    workers = workers or model_compute.model_workers(device)
    destinations = {key: str(Path(value).resolve()) for key, value in (destinations or {}).items()}
    folder = str(Path(folder).resolve())
    training = operation == 'train'
    requested = device
    threads = threads_for(statics, options)
    card = model_compute.device_for(requested)
    if card is None and str(requested).lower() not in {'cpu', 'none'}:
        log.warning('the site graph asks for device %r and no visible GPU answers it; it runs on the CPU', requested)
    effective = requested if card else 'cpu'
    args = resources(statics, folder, destinations.values(), options.get('csv'))
    with graph_publication.ownership(args) as held:
        before = observations(statics, None if training else folder)
        inputs = {'statics': statics, 'options': options, 'folder': folder, 'dates': dates, 'data': data, 'idx': idx,
                  'build_datasets': build_datasets, 'inverse': inverse, 'site': site, 'observations': before,
                  'destinations': destinations}
        ident = uuid.uuid4().hex
        ctx = run_reporting.context()
        task = 'site-graph:' + hashlib.sha256(folder.encode()).hexdigest() + ':' + operation
        request = workers.request(ctx['run_id'] if ctx else 'site-graph:' + ident, task, ident, threads, device=card)
        handle = None
        try:
            held.check()
            handle = workers.prepare(request, 'eforecast.datasets.site_graph_worker:execute',
                {'operation': operation, 'requested_device': requested, 'effective_device': effective},
                input_files={'graph.pickle': pickle.dumps(inputs, protocol=5)},
                execution={'device': effective, 'threads': threads})
            result = workers.wait(handle, timeout=None)
            if result['state'] != 'success':
                from .compute_workers import failure_reason
                raise ResourceUnavailable(f'site graph {failure_reason(result)}')
            held.check()
            check = lambda: check_observations(statics, None if training else folder, before)
            check()
            if training:
                actual = {key: destinations[key] for key in result['value']['exported']}
                value = graph_publication.publish(workers, handle, result, folder, actual, held, check)
            else:
                root = workers.root(handle)
                held.check()
                check()
                value = {}
                if options.get('csv'):
                    for name in result['value']['sites']:
                        file = Path(f'{options["csv"]}_{name.replace(os.sep, "_")}.csv')
                        file.parent.mkdir(parents=True, exist_ok=True)
                        temporary = file.with_name('.graph-' + ident + '-' + file.name)
                        try:
                            import shutil
                            source = root / 'artifacts' / (hashlib.sha256(name.encode()).hexdigest() + '.csv')
                            shutil.copyfile(source, temporary)
                            from eforecast.common_utils.artifact import digest as file_digest
                            if file_digest(temporary) != result['files'][source.relative_to(root).as_posix()]['sha256']:
                                raise ValueError('forecast CSV changed during handoff')
                            held.check()
                            check()
                            os.replace(temporary, file)
                        finally:
                            temporary.unlink(missing_ok=True)
                else:
                    import joblib
                    value = joblib.load(root / 'artifacts/predictions.pickle')
            workers.discard_result(handle)
            return value
        except BaseException:
            if handle is not None:
                workers.cancel(handle, force=True)
                try:
                    workers.wait(handle, timeout=30)
                except (Exception, KeyboardInterrupt):
                    log.warning('the site-graph worker was told to stop and has not been seen to exit')
            raise


def dispatch(operation, *, folder, statics=None, **kwargs):
    meta = json.loads((bundle_root(folder) / 'meta.json').read_text())
    if operation != 'stack':
        statics = {name: dict(sd) for name, sd in (statics or {}).items()}
        for name in meta['sites']:
            if name not in statics:
                if name not in meta.get('configs', {}):
                    raise ValueError(f'no case file for site {name}')
                sd = initialise(meta['configs'][name])
                statics[name] = sd
            elif name in meta.get('configs', {}):
                statics[name].setdefault('_graph_config', meta['configs'][name])
    kwargs.setdefault('options', {})
    kwargs['options'] = {**kwargs['options'], 'params': meta['params']}
    return _run(operation, folder=folder, statics=statics, **kwargs)


def cli(args):
    if args.recover_publication:
        path = Path(args.recover_publication).resolve()
        if not (path / '.stage-artifacts' / graph_publication.WRITER).exists():
            log.info('no recorded site-graph publication to recover at %s', path)
            return 0
        with graph_publication.ownership([path / '.stage-artifacts']):
            return 0
    if args.predict:
        folder = args.model or args.out
        if not folder:
            raise SystemExit('--predict needs --model (or --out)')
        import pandas as pd
        frames = dispatch('online', folder=folder, dates=pd.date_range(*args.predict), device=args.gpu,
                          inverse=True, options={'csv': args.csv, 'threads': args.threads,
                            'params': json.loads((bundle_root(folder) / 'meta.json').read_text())['params']})
        if not args.csv:
            for name, frame in frames.items():
                print(name)
                print(frame.round(3).to_string())
        return 0
    if len(args.configs) < 2:
        raise SystemExit('a site graph needs at least two cases')
    statics = {}
    for config in args.configs:
        sd = initialise(config)
        if sd['_id'] in statics:
            raise ValueError(f'duplicate site identity {sd["_id"]}: use distinct site cases')
        statics[sd['_id']] = sd
    first = next(iter(statics.values()))
    out = Path(args.out) if args.out else Path(first.get('path_group') or first['path_model']) / 'site_graph' / args.name
    destinations = {}
    if args.export or args.export_root:
        for name, sd in statics.items():
            base = Path(args.export_root) / name if args.export_root else Path(sd['path_model'])
            destinations[hashlib.sha256(name.encode()).hexdigest()] = str((base / 'global/GRAPH/test_0').resolve())
    _run('train', folder=str(out), statics=statics, options=vars(args), device=args.gpu, destinations=destinations)
    log.info('site-graph model %s and %d site exports committed', out, len(destinations))
    return 0


install_resident_backend('site_graph', dispatch)
