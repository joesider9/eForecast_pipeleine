# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Create the training datasets of one project, stage by stage.

Examples (run from the project root)::

    python run_datasets.py --config ../configuration/cases/ptolemaida_pv --dry-run
    python run_datasets.py --config ../configuration/cases/ptolemaida_pv --stage target row
    python run_datasets.py --config ../configuration/cases/ptolemaida_pv --stage nwp --recreate --serial
    python run_datasets.py --config short_term_image --stage all      # legacy config package

``--config`` is a YAML case (single file, parts directory or ``main.yaml``) or
the name of a legacy project package holding ``configuration/config.py``.

Stages, in execution order:

    extract     daily NWP pickles in path_group_nwp (shared by the project group)
    nwp         dataset_nwps.pickle
    images      dataset_images.pickle (satellite images + encoder statistics)
    target      dataset_target_data.csv
    row         dataset_row_data.pickle
    lstm        dataset_lstm_data.pickle (not for a long_term case, which has no sequences)
    split       cv_mask.pickle: train / validation / test dates  (group ``all`` only)
    preprocess  scalers, fitted on the training dates, and then
                the weather inputs scaled once for every reader  (group ``all`` only)
    clustering  clusters and their masks                         (group ``all`` only)
    features    feature selection                                (group ``all`` only)

``datasets`` (the default) runs extract .. lstm, ``all`` adds the four model
stages.  A stage whose output already exists is skipped; ``--recreate``
rebuilds the selected stages (the previous output comes back if the rebuild
fails). ``--serial`` runs data work on one CPU without helper pools, in this
process, so a debugger attaches directly.

The partition is drawn first, before anything is fitted: the scalers of the
preprocess stage, the clusterer and feature selection all see the same
training dates.  ``--stage split --recreate`` draws a **new** partition, which
every artifact already built against the old one refuses to be reused with.
"""
import importlib
import logging
import os
import sys
import time

import numpy as np

sys.path.append(os.getcwd())

from eforecast.init.initialize import initializer
from eforecast.config.entrypoint_parsers import parse_datasets_args
from eforecast.config.capabilities import NWP_EXTRACTORS, NO_SEQUENCES, has_sequences
from eforecast.config.pipeline_stages import (DATASET_STAGES, MODEL_STAGES, DATASET_STAGE_GROUPS,
                                              DATASET_DEFAULT_STAGES)

STAGE_GROUPS = DATASET_STAGE_GROUPS
STAGE_OUTPUT = {'nwp': 'nwp', 'images': 'images', 'target': 'target', 'row': 'row', 'lstm': 'lstm'}

log = logging.getLogger('run_datasets')


# ----------------------------------------------------------------------------- stages

def validate_dataset_request(static_data, stages, check_source=True):
    """Reject unsupported extraction and missing input before recreating any outputs."""
    if 'extract' in stages:
        unsupported = [nwp['model'] for nwp in static_data['NWP'] or [] if nwp['model'] not in NWP_EXTRACTORS]
        if unsupported:
            raise ValueError(f'no NWP extractor for {", ".join(unsupported)}; the extract stage supports ECMWF. '
                             'For pre-extracted providers, select the nwp stage without extract.')
    if check_source and not os.path.isfile(static_data['filename']):
        raise FileNotFoundError(f"Cannot find the main file csv {static_data['filename']}")


def _extract_days(worker, days, n_jobs, serial, give_up=None):
    """The status of each of ``days``, in their order.

    ``give_up`` ends it once that many days in a row could not be downloaded. The days after
    them are not asked for and have no status: the list is then shorter than ``days``.
    """
    from eforecast.bounded_pools import map_bounded
    from tqdm import tqdm
    results = (worker(t) for t in days) if serial else map_bounded(worker, days, n_jobs=n_jobs)
    statuses, refused = [], 0
    try:
        for status in tqdm(results, total=len(days)):
            statuses.append(status)
            refused = refused + 1 if status == 'unfetched' else 0
            if give_up and refused >= give_up:
                break
    finally:
        results.close()         # the readers end here, not whenever the generator is collected
    return statuses


def stage_extract(static_data, dataset, args):
    """Daily NWP pickles of the project group for every day of the time series."""
    validate_dataset_request(static_data, ('extract',), check_source=False)
    from collections import Counter
    from functools import partial
    from eforecast.nwp_extraction import gdrive
    from eforecast.nwp_extraction.ecmwf_extractor import EcmwfExtractor
    variables = [var_data['name'] for var_data in static_data['variables'] if var_data['type'] == 'nwp']
    days = dataset.dates.round('D').unique().sort_values()      # stage_artifacts.weather_days names the same days
    for nwp in static_data['NWP']:
        extractor = EcmwfExtractor(variables, nwp.get('path_nwp_source') or static_data['nwp_folder'],
                                   path_group_nwp=static_data['path_group_nwp'])
        log.info('extract %s: %d days, variables %s, source %s -> %s', nwp['model'], len(days), variables,
                 extractor.path_nwp, static_data['path_group_nwp'])
        worker = partial(extractor.nwps_extract, refit=args.recreate_extraction)
        # The days no local folder holds come from the shared Drive folder when the case names
        # one. They go last and a few at a time, whatever n_jobs is: what is on the disk is
        # read at full width first, and a public folder is not asked by twenty readers at once.
        with gdrive.attached(extractor, nwp.get('path_nwp_gdrive'), days, refit=args.recreate_extraction) as away:
            elsewhere = set(away)
            here = [t for t in days if t not in elsewhere]
            found = dict(zip(here, _extract_days(worker, here, static_data['n_jobs'], args.serial))) if here else {}
            if away:
                found.update(zip(away, _extract_days(worker, away, min(static_data['n_jobs'], gdrive.WORKERS),
                                                     args.serial, give_up=gdrive.GIVE_UP)))
        statuses = [found.get(t, 'unfetched') for t in days]
        counts = Counter(statuses)
        log.info('extract %s: %s', nwp['model'], dict(counts))
        missing = [t.strftime('%Y-%m-%d') for t, status in zip(days, statuses) if status in ('missing', 'error')]
        if missing:
            log.warning('extract %s: %d days without a pickle, e.g. %s', nwp['model'], len(missing), missing[:15])
        unfetched = [t.strftime('%Y-%m-%d') for t, status in zip(days, statuses) if status == 'unfetched']
        if unfetched:
            # Not recorded as days without weather: the shared folder has them. The stage fails,
            # keeps every day it converted, and its next run asks only for these.
            stopped = (f'; it stopped asking after {gdrive.GIVE_UP} days in a row were refused'
                       if any(t not in found for t in away) else '')
            raise gdrive.DriveError(
                f"extract {nwp['model']}: {len(unfetched)} of {len(away)} day(s) were not downloaded from Google "
                f"Drive, e.g. {unfetched[:5]}{stopped}. Why is in the log above, for each day asked. The days "
                f"already converted are kept: run the extract stage again and it fetches only the rest.")


def stage_nwp(static_data, dataset, args):
    dataset.create_nwp_dataset(parallel=not args.serial)


def stage_images(static_data, dataset, args):
    dataset.create_image_dataset(parallel=not args.serial)


def stage_target(static_data, dataset, args):
    dataset.create_target()


def stage_row(static_data, dataset, args):
    dataset.create_row_datasets()


def stage_lstm(static_data, dataset, args):
    dataset.create_lstm_dataset()


def stage_split(static_data, dataset, args):
    """The train / validation / test dates, drawn once and before anything is fitted."""
    from eforecast.datasets.data_preprocessing.data_split import Splitter, partition_file, partition_fingerprint
    splitter = Splitter(static_data, is_online=False, train=True)
    existing = splitter.file_manager.check_if_exists_cv_data()
    if args.recreate and existing is not None and os.path.exists(partition_file(static_data)):
        log.warning('recreate: the partition of %s is redrawn. Everything fitted against the old one -- the '
                    'scalers, the clusters and their models, the feature selectors -- refuses to be reused with '
                    'the new one; rebuild the clustering and feature stages with --recreate too',
                    partition_file(static_data))
    masks = splitter.split(refit=args.recreate)
    total = sum(mask.shape[0] for mask in masks)
    log.info('partition %s: train %d, val %d, test %d samples of %d (%.1f%% held out)',
             partition_fingerprint(static_data, masks)['masks_sha256'],
             *[mask.shape[0] for mask in masks], total,
             100 * (masks[1].shape[0] + masks[2].shape[0]) / max(total, 1))


def stage_preprocess(static_data, dataset, args):
    from eforecast.datasets.data_preprocessing.data_pipeline import DataPipeline
    pipeline = DataPipeline(static_data, recreate=False, online=False, train=True)
    pipeline.fit_pipe(force=args.recreate)
    log.info('scalers: %d in %s', len(pipeline.scaler.scalers), pipeline.scaler.filename)


def stage_clustering(static_data, dataset, args):
    from eforecast.clustering.clustering_manager import ClusterOrganizer
    organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=args.recreate)
    organizer.fit()
    organizer.create_clusters_and_cvs()


def stage_features(static_data, dataset, args):
    from eforecast.feature_selection.feature_selection_fit import FeatureSelector
    selector = FeatureSelector(static_data, recreate=args.recreate, online=False, train=True)
    selector.fit(recreate_lstm=False)
    selector.create_backup()


STAGES = {'extract': stage_extract, 'nwp': stage_nwp, 'images': stage_images, 'target': stage_target,
          'row': stage_row, 'lstm': stage_lstm, 'split': stage_split, 'preprocess': stage_preprocess,
          'clustering': stage_clustering, 'features': stage_features}


def stage_applies(stage, static_data):
    """Reason the stage is not applicable to this project, or None."""
    if stage in {'extract', 'nwp'} and static_data['NWP'] is None:
        return 'project has no NWP models'
    if stage == 'images' and not static_data['use_image']:
        return 'project does not use images'
    if stage == 'lstm' and not has_sequences(static_data):
        return NO_SEQUENCES
    if stage == 'features' and not any(method is not None for method in static_data.get('feature_selection_methods') or []):
        return 'no feature-selection method configured'      # the plan's own rule: a dry run must not promise it
    return None


# ----------------------------------------------------------------------------- helpers

def parse_args(argv=None):
    return parse_datasets_args(argv, description=__doc__)

def expand_stages(requested):
    stages = []
    for item in requested:
        for stage in STAGE_GROUPS.get(item, (item,)):
            if stage not in stages:
                stages.append(stage)
    order = list(STAGES)
    return sorted(stages, key=order.index)


def load_static_data(config, dry_run, args=None):
    if os.path.exists(config):
        if args is not None:
            from eforecast.config.run_inputs import prepare
            return initializer(prepare(config, 'datasets', args), dry_run=dry_run)
        return initializer(config, dry_run=dry_run)
    module_name = config if '.' in config else f'{config}.configuration.config'
    module = importlib.import_module(module_name)
    if args is not None:
        from eforecast.config.run_inputs import prepare_legacy
        return initializer(prepare_legacy(module.config(), module.__file__, 'datasets', args), dry_run=dry_run)
    return initializer(module.config(), dry_run=dry_run)


def setup_logging(static_data, level, dry_run):
    handlers = [logging.StreamHandler(sys.stdout)]
    if not dry_run:
        handlers.append(logging.FileHandler(os.path.join(static_data['path_logs'], 'run_datasets.log'),
                                            mode='a', encoding='utf-8'))
    logging.basicConfig(level=level.upper(), handlers=handlers, force=True,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S')


def log_status(files_manager, title):
    log.info('%s', title)
    for name, (file, size_mb) in files_manager.dataset_status().items():
        state = f'{size_mb:8.1f} MB' if size_mb is not None else '   missing'
        log.info('  %-7s %s  %s', name, state, file)
    from eforecast.datasets import scaled_inputs
    word, why = scaled_inputs.state(files_manager.static_data)
    if word != 'none':
        log.info('  %-7s %s (%s)  %s', 'scaled', word, why, scaled_inputs.folder(files_manager.static_data))


def run_stage(stage, static_data, dataset, args):
    reason = stage_applies(stage, static_data)
    if reason:
        log.info('stage %-10s skipped: %s', stage, reason)
        return
    log.info('stage %-10s start', stage)
    start = time.time()
    # One writer, in place: the stage runs here, in this process, and what it prints is what
    # you see. Nothing is copied aside first and nothing is promoted afterwards.
    if dataset is None and stage in DATASET_STAGES:
        from eforecast.datasets.dataset_creator import DatasetCreator
        dataset = DatasetCreator(static_data, train=True, parallel=not args.serial)
    from eforecast.bounded_pools import cancel_check
    from eforecast.config.run_reporting import check_cancelled
    token = cancel_check.set(check_cancelled)      # a cancelled run stops its reader pools
    try:
        STAGES[stage](static_data, dataset, args)
    finally:
        cancel_check.reset(token)
    elapsed = time.time() - start
    output = STAGE_OUTPUT.get(stage)
    if output is not None:
        from eforecast.datasets.files_manager import FilesManager
        file, size_mb = FilesManager(static_data, train=True).dataset_status()[output]
        state = f'{size_mb:.1f} MB' if size_mb is not None else 'MISSING'
        log.info('stage %-10s done in %6.1fs -> %s (%s)', stage, elapsed, file, state)
    else:
        log.info('stage %-10s done in %6.1fs', stage, elapsed)


# ----------------------------------------------------------------------------- main
def _main(argv=None, report=None):
    args = parse_args(argv)
    # Snapshot identity/path is logged before dataset paths and handlers are created.
    logging.basicConfig(level=args.log_level.upper())
    stages = expand_stages(args.stage)
    static_data = load_static_data(args.config, args.dry_run, args)
    applicable = any(stage_applies(stage, static_data) is None for stage in stages)
    validate_dataset_request(static_data, stages, check_source=not args.dry_run and applicable)
    from eforecast.config.stage_catalogue import resolve, describe, require_ready
    from eforecast.config.run_inputs import request_options
    plan = resolve(static_data, 'datasets', request_options('datasets', args))
    log.info('%s', describe(plan))
    if plan.get('completion_version') == 1:
        from eforecast.config import stage_resume
        approved = stage_resume.accepted()
        resume = stage_resume.decide(static_data, plan, verify=not args.dry_run,
                                     action=stage_resume.requested_action(static_data, plan['options']))
        stage_resume.reconcile(resume, approved)     # the accepted permissions are not replaced here
        log.info('%s', stage_resume.describe(resume))
        stage_resume.pin(plan, resume)   # what may be skipped, and what an explicit rebuild covers
        if not args.dry_run:
            # Datasets that exist were built from other settings than the case has now. This run
            # builds them again, and says which before it does. A run the service started was
            # asked about that in the browser, and stops here if it was not.
            stage_resume.announce_changes(resume, static_data)
    require_ready(plan)
    if not applicable and not args.dry_run:
        report.update(kind='nothing_to_do', detail='no applicable dataset stages',
                      counts={'stages': len(stages), 'applicable': 0})
        return static_data
    if args.serial:
        static_data['n_jobs'] = 1
    setup_logging(static_data, args.log_level, args.dry_run)
    log.info('project %s  type=%s  horizon_type=%s  horizon=%s  ts_resolution=%s',
             static_data['_id'], static_data['type'], static_data['horizon_type'],
             static_data['horizon'], static_data['ts_resolution'])
    log.info('stages: %s%s%s', ' '.join(stages), '  [recreate]' if args.recreate else '',
             '  [serial]' if args.serial else '')

    from eforecast.datasets.files_manager import FilesManager
    files_manager = FilesManager(static_data, is_online=False, train=True)
    if args.dry_run:
        report.update(kind='nothing_to_do', detail='dataset dry run')
        from eforecast.init.initialize import PATH_KEYS
        for key in PATH_KEYS:
            log.info('  %-16s %s', key, static_data[key])
        log.info('  coord %s  area_group %s', static_data['coord'], static_data['area_group'])
        for nwp in static_data['NWP'] or []:
            area = nwp.get('area')
            area = {k: np.asarray(v).round(3).tolist() for k, v in area.items()} if isinstance(area, dict) \
                else np.asarray(area).round(3).tolist()
            log.info('  NWP %s resolution=%s levels=%s area=%s source=%s%s', nwp['model'], nwp['resolution'],
                     nwp['levels'], area, nwp['path_nwp_source'],
                     f" (a day not there: {nwp['path_nwp_gdrive']})" if nwp.get('path_nwp_gdrive') else '')
        log.info('  main file %s', static_data['filename'])
        for stage in stages:
            reason = stage_applies(stage, static_data)
            log.info('  stage %-10s %s', stage, f'skip ({reason})' if reason else 'run')
        log_status(files_manager, 'dataset status:')
        return static_data

    log_status(files_manager, 'dataset status before:')
    completed = set()
    for stage in stages:
        from eforecast.config.dataset_cpus import budget
        from eforecast.config.run_reporting import stage as observe_stage
        # The reader pools and the selectors' estimators take their size from this budget;
        # without it they run on one CPU. It covers the stage boundary as well, so what a
        # stage wrote is checked by the same readers. --serial keeps everything in this
        # process, on one CPU, for a debugger.
        idle = args.serial or stage_applies(stage, static_data) is not None    # no pool to size
        with budget(static_data, stage, serial=idle), \
                observe_stage(plan, stage, static_data, completed) as applicable:
            if applicable:
                private = applicable.static_data
                # --recreate needs nothing here: the stage boundary has already set this
                # stage's outputs aside, and puts them back if the rebuild fails.
                # A stage must parse the source observed by its own receipt. Reusing an earlier
                # creator would bind new source bytes to old cached DataFrames. Model stages
                # operate on published datasets and do not need a source reader at all.
                # The stage opens and parses the source and loads its transforms itself.
                run_stage(stage, private, None, args)
        if stage == 'preprocess' and stage_applies(stage, static_data) is None:
            # The scalers are there, fitted now or reused: the weather is scaled with them once,
            # here, for the clustering and feature stages below and for every trial after them.
            # A copy on disk that already matches the dataset and the scalers is left as it is.
            from eforecast.datasets import scaled_inputs
            scaled_inputs.ensure(static_data)
    report['counts'] = {'stages': len(stages),
                        'applicable': sum(stage_applies(stage, static_data) is None for stage in stages)}
    if report['counts']['applicable'] == 0:
        report.update(kind='nothing_to_do', detail='no applicable dataset stages')
    log_status(files_manager, 'dataset status after:')
    # The content identity of each dataset is computed here, once, so that no trial ever has to
    # read a dataset to find out what it trained on.
    try:
        from eforecast.common_utils import lineage
        lineage.publish_dataset_identities(static_data)
    except Exception as error:                      # noqa: BLE001 -- bookkeeping never fails a stage
        log.warning('dataset identities were not published (%r)', error)
    return static_data


def main(argv=None):
    from eforecast.config.run_reporting import operation
    with operation('datasets') as report:
        return _main(argv, report)


if __name__ == '__main__':
    main()
