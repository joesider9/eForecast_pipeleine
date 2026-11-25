# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Prediction, selection, combination and evaluation of the trained models (the step after training).

Protocol
    The evaluation period (after ``evaluation_start``) is split in **three**, in time order:

    * **selection** chooses the trials and fits the combiners;
    * **calibration** fits the conformal bands and nothing else -- the last
      ``calibration_fraction`` of the part ``selection_fraction`` marks off;
    * **holdout** is touched by nothing but the final report.

    A band fitted on the slice that chose the predictor is narrow for the wrong reason: the
    residuals of a predictor picked *because* it does well on those dates understate the error of
    the same predictor on any other dates, and the interval inherits the understatement. The
    calibration slice is what makes the reported coverage mean something.

    Each boundary is purged of the target-window overlap, as the training partition is: a sample
    whose target reaches across a boundary is dropped from the earlier slice.

Stages (``run_combine.py``)
    predict   every trial of every method predicts the evaluation period once; the prediction is
              cached in its folder (``predictions_eval.pickle``)
    select    per (cluster, method) the ``keep_best_models`` trials with the lowest error on the
              selection slice; written to ``<cluster_dir>/<method>/selected_trials.json``
    combine   per cluster: the kept trials of a method are averaged into one predictor per method,
              the combiners (``combining.methods``: kmeans, bcp, elastic_net) are fitted per horizon
              step on the selection slice; cluster averages across clusters; the "models" combiner
              over the global methods and the cluster averages
    evaluate  MAE, RMSE and bias, normalised by the rated power, per step and overall, on both
              slices, for every predictor of every level -> ``Results/evaluation.csv`` and the
              holdout summary ``Results/summary.csv``; and what the trials' hyper-parameters say
              (``hyper_parameters``) -> ``Results/hyper_parameters.csv`` with every finished trial
              and ``Results/hyper_parameter_report.txt``

Online: ``Ensemble(static_data, online=True).predict()`` runs the kept trials and the fitted
combiners on the online datasets and returns every level; ``inverse`` maps back to power.
"""
import contextlib
import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNetCV

from eforecast.common_utils import artifact, completion, lineage, trial_layout
from eforecast.config.capabilities import CPU_METHODS
from eforecast.combine_predictions.algorithms import bcp_fit, kmeans_fit, kmeans_predict
from eforecast.common_utils.dataset_utils import get_time_level
from eforecast.datasets.data_preprocessing.data_split import resolution_step, target_window
from eforecast.datasets.data_feeder import DataFeeder
from eforecast.datasets.data_preprocessing.data_scaling import Scaler
from eforecast.datasets.row_data_transformation import RowDataTransformer

log = logging.getLogger(__name__)


def inference_adapter():
    from eforecast.compute_context import active_worker, resident_backend
    return resident_backend('inference') if active_worker.get() is None else None

SELECTION_FILE = 'selected_trials.json'
CACHE_FILE = 'predictions_{period}.pickle'
QUANTILE_CACHE_FILE = 'quantiles_{period}.pickle'   # {quantile: frame} of a trial with a quantile head
CACHE_KEY_SUFFIX = '.key.json'                      # what a cached prediction was computed from
LEVELS = ('trial', 'method', 'combiner', 'cluster_average', 'global', 'models')


# ---------------------------------------------------------------------------------- periods

def embargo(earlier, later, window):
    """``earlier`` without the dates whose target window reaches into ``later``.

    The slices are contiguous in time, so one comparison does it: a sample at ``t`` carries labels
    up to ``t + window`` (``obs(t + lag x resolution)``), and anything reaching the next slice is
    the same overlap Step 06 purges at the training boundaries.
    """
    if len(earlier) == 0 or len(later) == 0 or window <= pd.Timedelta(0, 'ns'):
        return earlier
    return earlier[earlier + window < later[0]]


def split_evaluation(dates, fraction, calibration_fraction=0.0, window=None):
    """``(selection, calibration, holdout)`` of the evaluation dates, purged at each boundary.

    ``fraction`` marks off the part that may influence anything; its last ``calibration_fraction``
    is reserved for the conformal bands, so the holdout keeps the size it had. With
    ``calibration_fraction`` at 0 the calibration slice *is* the selection slice, which is what
    this did before the slices were separated, and the caller says so.
    """
    window = pd.Timedelta(0, 'ns') if window is None else window
    dates = pd.DatetimeIndex(get_time_level(dates)).unique().sort_values()
    if len(dates) == 0:
        return dates, dates, dates
    cut = int(round(len(dates) * float(fraction)))
    cut = min(max(cut, 1), len(dates) - 1) if len(dates) > 1 else len(dates)
    fitting, holdout = dates[:cut], dates[cut:]
    share = float(calibration_fraction or 0.0)
    if share <= 0 or len(fitting) < 2:
        return embargo(fitting, holdout, window), embargo(fitting, holdout, window), holdout
    keep = int(round(len(fitting) * (1.0 - share)))
    keep = min(max(keep, 1), len(fitting) - 1)
    selection, calibration = fitting[:keep], fitting[keep:]
    return (embargo(selection, calibration, window), embargo(calibration, holdout, window), holdout)


class Targets:
    """Targets of the evaluation period (scaled and in power) and the two slices."""

    def __init__(self, static_data, online=False):
        self.static_data = static_data
        self.scaler = Scaler(static_data, recreate=False, online=online, train=not online)
        self.row_transformer = RowDataTransformer(static_data)
        self.scale_target_method = static_data['scale_target_method']
        self.rated = static_data['rated']
        self.columns = list(static_data['target_variable']['columns'])
        self.horizon = list(static_data['target_variable']['lags'])
        self.y_scaled = None
        if not online:
            self.y_scaled = DataFeeder(static_data, train=False).feed_target(inverse=False)
            self.y_scaled.index = get_time_level(self.y_scaled.index)
            self.calibration_fraction = float(static_data.get('calibration_fraction', 0.0) or 0.0)
            self.window = target_window(static_data)
            self.selection, self.calibration, self.holdout = split_evaluation(
                self.y_scaled.index, static_data.get('selection_fraction', 0.5),
                calibration_fraction=self.calibration_fraction, window=self.window)
            log.info('evaluation period %s -> %s (target window %s): selection %d rows until %s, '
                     'calibration %d rows from %s, holdout %d rows from %s',
                     self.y_scaled.index.min(), self.y_scaled.index.max(), self.window,
                     len(self.selection), self.selection.max() if len(self.selection) else '-',
                     len(self.calibration), self.calibration.min() if len(self.calibration) else '-',
                     len(self.holdout), self.holdout.min() if len(self.holdout) else '-')
            if self.calibration_fraction <= 0:
                log.warning('methods.calibration_fraction is 0: the conformal bands are fitted on the same '
                            'dates that chose the trials and fitted the combiners, so they are narrower than '
                            'the error they are meant to cover and the holdout will under-cover. Set a '
                            'fraction (0.3 of the selection part is the default) to calibrate independently.')
            self.y = self.inverse(self.y_scaled)

    def inverse(self, frame):
        """Scaled predictions or targets -> power (the target scaler, then the target transformer)."""
        if frame is None or frame.shape[0] == 0:
            return frame
        out = frame.copy()
        out.columns = self.columns[:out.shape[1]] if out.shape[1] <= len(self.columns) else out.columns
        out = self.scaler.inverse_transform_data(out, f'target_{self.scale_target_method}')
        # create_target shifts short-term observations by lag * resolution after
        # transforming them. Undo that transform at the target's time, not the issue.
        offsets = None
        if self.static_data.get('horizon_type') == 'short_term':
            step = resolution_step(self.static_data)
            offsets = {column: lag * step for column, lag in zip(self.columns, self.horizon)}
        return self.row_transformer.transform(out, inverse=True, target_offsets=offsets)


def normalise_columns(pred, columns):
    """Every predictor frame carries the target column names (Step_k), whatever the model named them."""
    pred = pred.copy()
    if pred.shape[1] == len(columns):
        pred.columns = columns
    pred.index = get_time_level(pred.index)
    return pred[~pred.index.duplicated()].sort_index()


# ---------------------------------------------------------------------------------- trials

def trial_discovery_policy():
    """``manifest`` (the default) or ``legacy_weights`` to restore discovery by weights alone."""
    return completion.discovery_policy()


def trial_is_finished(path_weights, method=None):
    """Whether this trial folder may take part; the contract itself lives in ``common_utils``."""
    return completion.is_finished(path_weights, method=method)


def trial_specs(static_data, selected_only=False):
    """One record per **finished** trial: global methods and the cluster methods (GPU and CPU)."""
    specs = []
    skipped = []
    gpu_methods = [m for m, on in static_data['cluster_methods'].items() if on]
    cpu_methods = [m for m, on in (static_data.get('methods_cpu') or {}).items() if on]
    global_methods = [m for m, on in static_data['global_methods'].items() if on]
    groups = [('global', 'global', os.path.join(static_data['path_model'], 'global'), None, global_methods)]
    if static_data['is_Fuzzy']:
        # the folders of this tree, whatever tree the file was written in (trial_layout.cluster_folders)
        clusters = trial_layout.cluster_folders(static_data)
        for cluster_name, cluster_dir in clusters.items():
            clusterer_method = cluster_name.split('_')[0]
            rule = '_'.join(cluster_name.split('_')[1:])
            groups.append((cluster_name, rule, cluster_dir, clusterer_method, gpu_methods + cpu_methods))
    for cluster_name, rule, cluster_dir, clusterer_method, methods in groups:
        for method in methods:
            method_dir = str(trial_layout.method_directory(static_data, cluster_dir, method))
            if not os.path.isdir(method_dir):
                continue
            selected = read_selection(method_dir)
            for trial in sorted(os.listdir(method_dir)):
                path_weights = os.path.join(method_dir, trial)
                if not trial.startswith('test_') or not os.path.exists(os.path.join(path_weights, 'net_weights.pickle')):
                    continue
                usable, reason = trial_is_finished(path_weights, method=method)
                if not usable:
                    skipped.append({'cluster': cluster_name, 'method': method, 'trial': trial, 'reason': reason})
                    continue
                if selected_only and selected is not None and trial not in selected:
                    continue
                if reason.startswith('legacy') or trial_discovery_policy() == 'legacy_weights':
                    log.warning('%s/%s/%s admitted with unverified provenance: %s', cluster_name, method, trial, reason)
                specs.append({'cluster_name': cluster_name, 'rule': rule, 'cluster_dir': cluster_dir,
                              'clusterer_method': clusterer_method, 'method': method, 'kind': 'cpu' if method in CPU_METHODS else 'gpu',
                              'trial': trial, 'path_weights': path_weights, 'selected': selected is None or trial in selected})
    if skipped:
        log.warning('%d trial folder(s) hold weights but nothing that says they finished, so they take no '
                    'part in prediction or selection: %s%s', len(skipped),
                    '; '.join(f"{s['cluster']}/{s['method']}/{s['trial']} ({s['reason']})" for s in skipped[:8]),
                    ' ...' if len(skipped) > 8 else '')
    report_skipped(static_data, skipped)
    return specs


def report_skipped(static_data, skipped):
    """``Results/unfinished_trials.csv``: what was left out, and why, rather than a log line only."""
    try:
        path = os.path.join(static_data['path_model'], 'Results')
        os.makedirs(path, exist_ok=True)
        pd.DataFrame(skipped, columns=['cluster', 'method', 'trial', 'reason']).to_csv(
            os.path.join(path, 'unfinished_trials.csv'), index=False)
    except OSError as error:
        log.debug('cannot write the unfinished-trial table (%r)', error)


def read_selection(method_dir):
    file = os.path.join(method_dir, SELECTION_FILE)
    if not os.path.exists(file):
        return None
    with open(file, encoding='utf-8') as stream:
        return list(json.load(stream).get('trials', []))


class TrialPredictor:
    """Runs one trial on the evaluation (or online) datasets; evaluation predictions are cached in the trial folder."""

    def __init__(self, static_data, online=False):
        self.static_data = static_data
        self.online = online
        self.period = 'online' if online else 'eval'
        self.columns = list(static_data['target_variable']['columns'])
        self.cluster_dates = {}
        self.quantiles = {}                    # path_weights -> {quantile: frame} of the last predict
        self._inputs = None                    # the dependency identity of this case, read once
        self._cluster_dates_ready = False

    def _prepare_clusters(self):
        if self._cluster_dates_ready:
            return
        static_data, online = self.static_data, self.online
        if static_data['is_Fuzzy']:
            from eforecast.clustering.clustering_manager import ClusterOrganizer
            organizer = ClusterOrganizer(static_data, is_online=online, train=False)
            for method in organizer.methods:
                if organizer.makes_clusters(method):
                    _, self.cluster_dates[method] = organizer.compute_activations(method)
        self._cluster_dates_ready = True

    def dates_of(self, spec):
        if spec['clusterer_method'] is None:
            return None
        self._prepare_clusters()
        return self.cluster_dates[spec['clusterer_method']].get(spec['rule'], pd.DatetimeIndex([]))

    def load_model(self, spec):
        if spec['method'] == 'GRAPH':
            return GraphTrial(self.static_data, spec['path_weights'], online=self.online)
        if spec['kind'] == 'cpu':
            from eforecast.shallow_models.foundation_model import FOUNDATION_METHODS, FoundationModel
            from eforecast.shallow_models.shallow_model import ShallowModel
            cls = FoundationModel if spec['method'] in FOUNDATION_METHODS else ShallowModel
            model = cls(self.static_data, spec['path_weights'], train=False, online=self.online)
            if cls is FoundationModel:
                # Where it computes is what the case says now, not what its study was run with:
                # nothing of the model was learned on a device. In a process that was started
                # on no card, any preference is the CPU (execution_policy.torch_device).
                from eforecast.config.capabilities import foundation_device
                model.params['device'] = foundation_device(self.static_data, spec['method'])
        else:
            from eforecast.deep_models.pytorch_2x.network import DeepNetwork
            model = DeepNetwork(self.static_data, spec['path_weights'], train=False, online=self.online)
            from eforecast.compute_context import active_worker
            if active_worker.get() is not None:
                cap = max(1, int(os.environ.get('EFORECAST_INFERENCE_BATCH_SIZE', '256')))
                model.batch_size = max(1, min(cap, int(model.batch_size)))
                model.params['batch_size'] = model.batch_size
        if 'cluster_dir' in model.params:
            model.params['cluster_dir'] = spec['cluster_dir']
        if 'cluster' in model.params:
            model.params['cluster']['cluster_path'] = spec['cluster_dir']
        return model

    def cache_key(self, spec):
        """What this prediction was computed from: the model's own files and the inputs behind them.

        A cache that was written for another model, another dataset, another partition or another
        set of scalers is not this run's answer, however recent its file is.
        """
        manifest = artifact.read(spec['path_weights']) or {}
        model = {name: record.get('sha256') for name, record in (manifest.get('files') or {}).items()
                 if isinstance(record, dict)}
        if not model:   # a model published before manifests: identified as well as it can be, and said so
            model = {'unverified': lineage.file_identity(os.path.join(spec['path_weights'], 'net_weights.pickle'))}
        if self._inputs is None:
            # the inputs, without the budget: asking for more trials does not invalidate a prediction
            self._inputs = lineage.dependencies(self.static_data)['comparability_id']
        from eforecast.common_utils import image_assets
        pixels = None
        if image_assets.trial_uses_pixels(self.static_data, spec['path_weights']):
            if not hasattr(self, '_pixel_inputs'):
                self._pixel_inputs = image_assets.identities(self.static_data)
            pixels = self._pixel_inputs
        return lineage.dependency_id({'version': 1, 'period': self.period, 'columns': list(self.columns),
                                      'model': model, 'inputs': self._inputs, **({'image_pixels': pixels} if pixels is not None else {})})

    def cached(self, path, key):
        """The cached value when it was written for this key, otherwise nothing."""
        try:
            with open(f'{path}{CACHE_KEY_SUFFIX}', encoding='utf-8') as stream:
                stored = json.load(stream)
            if stored.get('key') != key or not os.path.exists(path):
                return None
            return joblib.load(path)
        except (OSError, ValueError) as error:
            log.debug('cache %s is not readable (%r)', path, error)
            return None

    def is_current(self, spec):
        """Whether the cached prediction of a trial was made from its model and the inputs of today,
        without loading it. A graph export publishes its prediction with itself and has no key."""
        if os.path.basename(os.path.dirname(spec['path_weights'])) == 'GRAPH':
            return True
        file = os.path.join(spec['path_weights'], CACHE_FILE.format(period=self.period))
        try:
            with open(f'{file}{CACHE_KEY_SUFFIX}', encoding='utf-8') as stream:
                return json.load(stream).get('key') == self.cache_key(spec)
        except (OSError, ValueError):
            return False

    def publish_cache(self, path, key, value):
        """The value first, then the key: an interrupted write leaves a cache nothing will load."""
        temporary = f'{path}.{os.getpid()}.tmp'
        try:
            joblib.dump(value, temporary)
            os.replace(temporary, path)
            with open(f'{path}{CACHE_KEY_SUFFIX}.tmp', 'w', encoding='utf-8') as stream:
                json.dump({'version': 1, 'key': key, 'written': time.time()}, stream)
            os.replace(f'{path}{CACHE_KEY_SUFFIX}.tmp', f'{path}{CACHE_KEY_SUFFIX}')
        except OSError as error:
            log.warning('cannot publish the prediction cache %s (%r)', path, error)
            for leftover in (temporary, f'{path}{CACHE_KEY_SUFFIX}.tmp'):
                if os.path.exists(leftover):
                    with contextlib.suppress(OSError):
                        os.unlink(leftover)
            from eforecast.common_utils.stage_context import current_stage
            if current_stage.get() is not None:
                raise OSError(f'required stage prediction cache could not be published: {path}') from error

    def drop_cache(self, path):
        for file in (path, f'{path}{CACHE_KEY_SUFFIX}'):
            if os.path.exists(file):
                with contextlib.suppress(OSError):
                    os.unlink(file)

    def predict(self, spec, refit=False):
        adapter = inference_adapter()
        if adapter is not None:
            value = adapter('trial', static=self.static_data, online=self.online, spec=spec, refit=refit)
            self.quantiles[spec['path_weights']] = value['quantiles']
            return value['prediction']
        from eforecast.bounded_pools import cancel_check
        if cancel_check.get() is not None:
            cancel_check.get()()
        from eforecast.compute_context import active_worker
        if active_worker.get() is not None:
            finished, reason = completion.is_finished(spec['path_weights'], method=spec['method'])
            if not finished:
                raise ValueError(f'unusable inference trial: {reason}')
            if artifact.read(spec['path_weights']) is not None:
                valid, reason = artifact.verify_digests(spec['path_weights'])
                if not valid:
                    raise ValueError(f'inference model verification failed: {reason}')
        if spec['method'] == 'GRAPH' and not self.online:
            # the exported frame is the model's own published artifact, not a cache to rewrite
            return self.load_model(spec).predict()
        cache = os.path.join(spec['path_weights'], CACHE_FILE.format(period=self.period))
        qcache = os.path.join(spec['path_weights'], QUANTILE_CACHE_FILE.format(period=self.period))
        key = self.cache_key(spec)
        if not self.online and not refit:
            hit = self.cached(cache, key)
            if hit is not None:
                return hit
            if os.path.exists(cache):
                log.info('%s/%s: the cached prediction was made from other inputs; predicting again',
                         spec['cluster_name'], spec['trial'])
        dates = self.dates_of(spec)
        quantiles = None
        if dates is not None and len(dates) == 0:
            pred = pd.DataFrame(columns=self.columns)
        else:
            model = self.load_model(spec)
            pred = model.predict(cluster_dates=dates)
            pred = normalise_columns(pred.clip(0, np.inf), self.columns)
            raw = getattr(model, 'quantile_predictions', None)
            if raw:
                quantiles = {float(q): normalise_columns(frame.clip(0, np.inf), self.columns) for q, frame in raw.items()}
            # A batch may visit many distinct foundation backbones. Keep at most
            # the current trial's pipeline resident, under the same outer grant.
            from eforecast.compute_context import active_worker
            if active_worker.get() is not None:
                from eforecast.shallow_models.foundation_model import clear_pipelines
                clear_pipelines()
        self.quantiles[spec['path_weights']] = quantiles
        if not self.online:
            self.publish_cache(cache, key, pred)
            if quantiles:
                self.publish_cache(qcache, key, quantiles)
            else:
                self.drop_cache(qcache)
        return pred

    def quantiles_of(self, spec):
        """{quantile: frame} of a trial with a quantile head (from the cache), else None."""
        if spec['path_weights'] in self.quantiles:
            return self.quantiles[spec['path_weights']]
        adapter = inference_adapter()
        if adapter is not None:
            return adapter('quantiles', static=self.static_data, online=self.online, spec=spec)
        qcache = os.path.join(spec['path_weights'], QUANTILE_CACHE_FILE.format(period=self.period))
        return self.cached(qcache, self.cache_key(spec))


def current_predictions(static_data):
    """``spec -> bool``: whether a trial's cached prediction is the answer of today (``is_current``).

    A stage that predicts only the kept trials leaves the caches of the others as they were, and
    a table of every trial must not mix those in. The predictor is made at the first question."""
    made = []

    def current(spec):
        if not made:
            made.append(TrialPredictor(static_data))
        return made[0].is_current(spec)
    return current


class GraphTrial:
    """The cross-site graph model as a trial of one site (``global/GRAPH/test_0``, exported by
    ``run_site_graph.py``).  Offline its predictions are the cached frame; online it rebuilds the online
    datasets of the other sites for the days of this site's online period and runs the saved graph."""

    def __init__(self, static_data, path_weights, online=False):
        verdict, reason = artifact.inspect(path_weights)
        if verdict == 'rejected':
            raise ValueError(f'unusable graph export: {reason}')
        self.static_data = static_data
        self.path_weights = path_weights
        self.online = online
        self.marker = joblib.load(os.path.join(path_weights, 'net_weights.pickle'))
        self.method = 'GRAPH'
        self.quantile_predictions = None

    def predict(self, cluster_dates=None):
        cols = [f'GRAPH_{c}' for c in self.static_data['target_variable']['columns']]
        if not self.online:
            if self.static_data.get('_graph_serving'):
                from eforecast.graph_serving import predict
                return predict(self.static_data, self.marker)
            cache = os.path.join(self.path_weights, CACHE_FILE.format(period='eval'))
            if not os.path.exists(cache):
                raise FileNotFoundError(f'{cache}: run run_site_graph.py --export for this site')
            return joblib.load(cache)
        from eforecast.compute_context import active_worker, resident_backend
        adapter = resident_backend('site_graph')
        if active_worker.get() is None and adapter is not None:
            site = self.marker.get('site', self.static_data['_id'])
            frames = adapter('trial', folder=self.marker['site_graph'], statics={site: self.static_data},
                             dates=cluster_dates, site=site)
            return frames[site]
        from eforecast.deep_models.pytorch_2x.site_graph_net import SiteGraphModel
        from eforecast.datasets.site_graph import site_frames
        model = SiteGraphModel(self.marker['site_graph'])
        site = self.marker.get('site', self.static_data['_id'])
        own = site_frames(self.static_data, online=True, **model.read_options())[0]     # this site's online rows exist already
        if own.shape[0] == 0:
            return pd.DataFrame(columns=cols)
        frames = model.predict_online(own.index, statics={site: self.static_data}, build_datasets=True,
                                      device='auto', log=log.info)
        frame = frames[site]
        frame.columns = cols
        if cluster_dates is not None:
            frame = frame.loc[frame.index.intersection(cluster_dates)]
        return frame


def _own_way(spec, static_data=None):
    """Trials that are not a model loaded and run in a reader: a site graph hands back the frame
    its model exported, and a foundation family that a model image serves starts that runtime.

    A foundation model that runs in this process is loaded and run like any other trial.
    """
    from eforecast.config.capabilities import FOUNDATION_METHODS
    if spec['method'] == 'GRAPH':
        return True
    return spec['method'] in FOUNDATION_METHODS and _served_by_an_image(spec, static_data)


def _served_by_an_image(spec, static_data):
    """Whether a foundation trial is answered by a model image and not by a model loaded here.

    Asked the way the model itself asks when it forecasts. A trial this cannot be said of is
    left to be predicted where it always was, which is also where its error is reported.
    """
    from eforecast import runtime_routing
    try:
        state = joblib.load(os.path.join(spec['path_weights'], 'net_weights.pickle'))
        decision = runtime_routing.route(spec['method'], static_data=static_data, params=state.get('params') or {})
    except Exception:  # noqa: BLE001
        return True
    return decision['where'] == runtime_routing.IMAGE


def _longest_first(spec):
    """Where a trial stands in the readers' queue: the long ones at the front.

    A batch is over when its last trial is, and the long trials of a case are its foundation
    models: Chronos-2 forecast three to seven rows a second where TimesFM forecast twenty and
    a network or a boosted tree thousands. In the order the trials are listed, the Chronos-2
    of the last cluster started when the readers before it were free and ran on alone at the
    end, with the rest of the CPUs idle.
    """
    from eforecast.config.capabilities import FOUNDATION_METHODS
    if spec['method'] in FOUNDATION_METHODS:
        return FOUNDATION_METHODS.index(spec['method'])
    return len(FOUNDATION_METHODS) + (spec['kind'] != 'gpu')


_predictors = {}        # in a reader of the predict stage: the predictor of a case, made once
#: What a batch has to tell whoever started it. A worker's log is removed with its folder when
#: its result is taken, so what it only logged is never read: the worker hands these back
#: (``inference_worker``) and the host repeats them where the stage is followed.
said = []


def _say(message, *arguments):
    log.warning(message, *arguments)
    said.append(message % arguments)


def _predict_one(static_data, spec, refit, context, cluster_dates):
    """One trial predicted in a reader process of the predict stage.

    The reader stands where the worker that started it stands: it is given that worker's context,
    so the trial is checked before it is loaded exactly as in the worker itself, and nothing is
    handed to a backend of the host. One predictor a case serves every trial the reader gets,
    and it is told which samples each cluster holds: the clusterer is asked once, by the process
    that started the readers, not once by each of them.
    """
    from eforecast.compute_context import active_worker
    if context is not None and active_worker.get() is None:
        active_worker.set(context)
    key = static_data['path_model']
    predictor = _predictors.get(key)
    if predictor is None:
        predictor = _predictors[key] = TrialPredictor(static_data)
        if cluster_dates is not None:
            predictor.cluster_dates, predictor._cluster_dates_ready = cluster_dates, True
    return predictor.predict(spec, refit=refit)


def _predict_on_card(static_data, spec, refit, context, cluster_dates):
    """One foundation trial in a reader that was started on a card: ``(None, prediction)``.

    What goes wrong there is handed back as ``(what it was, None)`` and not raised. A card can
    be out of memory, or lost to its driver, while the CPUs are fine: such a trial is predicted
    on the CPUs after all, and only what fails there fails the batch.
    """
    try:
        return None, _predict_one(static_data, spec, refit, context, cluster_dates)
    except Exception as error:  # noqa: BLE001
        return _in_a_line(error), None


def _in_a_line(error):
    """An error as its type and its first sentence: a driver adds a paragraph to "out of memory"."""
    lines = str(error).strip().splitlines()
    first = lines[0].split('. ')[0].strip() if lines else ''
    return f'{type(error).__name__}: {first}'[:240] if first else type(error).__name__


def predict_trials(static_data, refit=False, selected_only=False, cards=None):
    """Stage ``predict``: every trial (or the kept ones) predicts the evaluation period; returns {spec key: frame}.

    The trials are predicted side by side when this process was given CPUs for it. A case has a
    trial for every cluster, method and attempt, and one at a time was 718 of the 759 seconds of
    a combine run with forty-four of them, on a machine with twenty CPUs.

    Each reader works with the threads this process works with, and there are as many readers as
    its CPUs carry at that. The thread count is kept because a network's numbers depend on it:
    three trials of a wind case predicted with one thread and with sixteen differed from the
    same trials with two by 1e-7 (on values up to 0.9) -- inside the tolerance a reproduced
    prediction is held to, and still not the stored numbers. With the same count they are the
    stored numbers, to the bit, however many trials are predicted at once.

    A foundation model that runs in process is one of those trials. It is where a case with one
    for every cluster spends its combine run: Chronos-2 forecast seven rows a second with two
    threads on the weather columns of a solar park, 138,000 rows over seven clusters, which is
    five hours one cluster after another and under one with the clusters side by side. Each
    reader then holds a model of its own (about 2 GB) while it predicts.

    ``cards`` is ``{family: [card, ...]}``: the foundation families of the case that ask for a
    card, and the cards that answer (``model_compute.foundation_cards``, asked by the host). Their
    trials are predicted in readers of their own, one a card, while the others are predicted on
    the CPUs: Chronos-2 forecast three rows a second on a CPU thread of that solar park and 150
    on an RTX 2080 Ti. A trial that cannot be predicted on a card -- no memory left on it, a
    driver that fails -- is tried once more when the cards have done the rest, and is then
    predicted on the CPUs after all.

    Site graphs are predicted here, one after another, as before. So is a foundation model that a
    model image serves, unless its family was offered a card: its container is then started from
    the reader on that card and is given it. Started from here it computed on the CPUs, and TiRex
    took twenty minutes a cluster of a solar park that way.
    """
    adapter = inference_adapter()
    if adapter is not None:
        return adapter('trials', static=static_data, refit=refit, selected_only=selected_only)
    from concurrent.futures import FIRST_COMPLETED, wait
    from itertools import repeat
    from eforecast import execution_policy
    from eforecast.bounded_pools import CardReaders, cancel_check, map_bounded, wide_workers
    from eforecast.compute_context import active_worker
    predictor = TrialPredictor(static_data)
    specs = trial_specs(static_data, selected_only=selected_only)
    policy = execution_policy.current()
    own = int(policy['effective']['threads']) if policy else 0
    readers = wide_workers(own) if own > 1 else 0
    offered = cards or {}
    # A foundation model that an image serves is not loaded in any reader, but its container
    # computes where the process that starts it does. Started from here that is the CPUs, so
    # a family that was offered a card has such a trial started from a reader on that card,
    # which hands the card to the container. On the CPUs it stays here, one at a time.
    apart = {spec['path_weights'] for spec in specs if _own_way(spec, static_data)} if readers else set()
    beside = sorted((spec for spec in specs if spec['path_weights'] not in apart), key=_longest_first) if readers else []
    on_cards = sorted((spec for spec in specs if offered.get(spec['method']) and spec['method'] != 'GRAPH'),
                      key=_longest_first) if readers else []
    beside = [spec for spec in beside if not offered.get(spec['method'])]
    context = active_worker.get()
    ahead = {}
    dates = None
    if (on_cards or len(beside) > 1) and any(spec['clusterer_method'] is not None for spec in (*on_cards, *beside)):
        predictor._prepare_clusters()
        dates = predictor.cluster_dates

    def side_by_side(trials):
        """``trials`` predicted in the readers of this process, each filed under its own trial."""
        # Taken as they finish: in the order they were given, an hour of Chronos-2 holds back
        # every trial behind it, and the readers that are done have nothing to start.
        results = map_bounded(_predict_one, repeat(static_data), trials, repeat(refit), repeat(context),
                              repeat(dates), n_jobs=None, native_threads=own, in_order=False)
        try:
            for position, prediction in results:
                ahead[trials[position]['path_weights']] = prediction
        except Exception as error:  # noqa: BLE001
            position = getattr(error, 'position', None)
            spec = trials[position] if isinstance(position, int) and 0 <= position < len(trials) else None
            log.error('%s: prediction failed: %r',
                      f"{spec['cluster_name']} {spec['method']} {spec['trial']}" if spec else 'a trial', error)
            results.close()
            raise  # a batch that answers for its trials does not go on past one it could not predict

    lane, sent, back, finished = None, {}, [], False
    try:
        if on_cards:
            # The cards the families of these trials named between them, in the order named.
            named = list(dict.fromkeys(card for spec in on_cards for card in offered[spec['method']]))
            log.info('%d of %d trials are foundation models that ask for a card: they are predicted on %d card(s), '
                     'one at a time on each', len(on_cards), len(specs), len(named))
            lane = CardReaders(f'{__name__}:_predict_on_card', named, own)
            sent = {lane.submit(static_data, spec, refit, context, dates): spec for spec in on_cards}
        if len(beside) > 1:
            log.info('%d of %d trials are predicted side by side, %d at a time with %d threads each', len(beside),
                     len(specs), min(len(beside), readers), own)
            side_by_side(beside)
        check = cancel_check.get() or (lambda: None)
        again = set()
        while sent:
            check()
            done, _ = wait(sent, timeout=.2, return_when=FIRST_COMPLETED)
            for future in done:
                spec = sent.pop(future)
                try:
                    failure, prediction = future.result()
                except Exception as error:  # noqa: BLE001 -- the reader went away, with whatever it held
                    failure, prediction = _in_a_line(error), None
                if failure is None:
                    ahead[spec['path_weights']] = prediction
                    continue
                name = f"{spec['cluster_name']} {spec['method']} {spec['trial']}"
                if spec['path_weights'] not in again:
                    # Once more, behind what the cards still have to do. A card that was full a
                    # moment ago -- something else was computing on it -- is often not by then,
                    # and the CPUs are a long way round: TiRex took a hundred seconds for the
                    # seven clusters of a solar park on two cards and over an hour on the CPUs.
                    again.add(spec['path_weights'])
                    try:
                        sent[lane.submit(static_data, spec, refit, context, dates)] = spec
                        _say('%s was not predicted on a card (%s); it is tried on one again after the others',
                             name, failure)
                        continue
                    except Exception:  # noqa: BLE001 -- the readers on the cards are gone
                        pass
                _say('%s was not predicted on a card (%s); it is predicted on the CPUs', name, failure)
                if spec['path_weights'] not in apart:   # an image-served trial is started from here, below
                    back.append(spec)
        finished = True
    finally:
        if lane is not None:
            lane.close(stop=not finished)
    if len(back) > 1:
        side_by_side(sorted(back, key=_longest_first))
    out = {}
    for i, spec in enumerate(specs):
        if spec['path_weights'] in ahead:
            pred = ahead[spec['path_weights']]
        else:
            try:
                pred = predictor.predict(spec, refit=refit)
            except Exception as error:  # noqa: BLE001
                log.error('%s %s %s: prediction failed: %r', spec['cluster_name'], spec['method'], spec['trial'], error)
                if active_worker.get() is not None:
                    raise  # a required admitted batch must not bless inherited caches after failure
                continue
        out[spec['path_weights']] = pred
        log.info('[%d/%d] %s %s %s: %d rows', i + 1, len(specs), spec['cluster_name'], spec['method'], spec['trial'], pred.shape[0])
    return out


# ---------------------------------------------------------------------------------- metrics

def metrics(pred, y, rated, columns=None):
    """MAE, RMSE and bias per column and overall, in percent of the rated power (or of the target)."""
    common = pred.index.intersection(y.index)
    if len(common) == 0:
        return {}
    p = pred.loc[common].values.astype(float)
    t = y.loc[common].values.astype(float)[:, :p.shape[1]]
    norm = float(rated) if rated is not None else np.where(np.abs(t) > 1e-6, np.abs(t), np.nan)
    err = (p - t) / norm
    out = {}
    names = list(columns or pred.columns)
    for k, name in enumerate(names[:err.shape[1]]):
        e = err[:, k]
        out[name] = {'mae': 100 * np.nanmean(np.abs(e)), 'rmse': 100 * np.sqrt(np.nanmean(e ** 2)), 'bias': 100 * np.nanmean(e), 'n': int(np.isfinite(e).sum())}
    out['all'] = {'mae': 100 * np.nanmean(np.abs(err)), 'rmse': 100 * np.sqrt(np.nanmean(err ** 2)), 'bias': 100 * np.nanmean(err), 'n': int(np.isfinite(err).sum())}
    return out


# ---------------------------------------------------------------------------------- selection

def select_trials(static_data, predictions=None, refit=False):
    """Keep trials with lowest MAE on the separate selection slice."""
    adapter = inference_adapter()
    if adapter is not None:
        return adapter('select', static=static_data, predictions=predictions, refit=refit)
    targets = Targets(static_data)
    keep = int(static_data.get('keep_best_models') or 2)
    specs = trial_specs(static_data)
    if predictions is None:
        predictions = predict_trials(static_data, refit=False)
    rows = []
    by_group = {}
    for spec in specs:
        pred = predictions.get(spec['path_weights'])
        if pred is None or pred.shape[0] == 0:
            continue
        m = metrics(targets.inverse(pred.loc[pred.index.intersection(targets.selection)]), targets.y, targets.rated, targets.columns)
        mae = m.get('all', {}).get('mae', np.nan)
        rows.append({'cluster': spec['cluster_name'], 'method': spec['method'], 'trial': spec['trial'], 'mae_selection': mae,
                     'n': m.get('all', {}).get('n', 0), 'method_dir': os.path.dirname(spec['path_weights'])})
        by_group.setdefault((spec['cluster_name'], spec['method'], os.path.dirname(spec['path_weights'])), []).append((mae, spec['trial']))
    table = pd.DataFrame(rows)
    unsupported = []
    for (cluster_name, method, method_dir), scored in by_group.items():
        trials = len(scored)
        scored = sorted((s for s in scored if np.isfinite(s[0])), key=lambda s: s[0])
        kept = [trial for _, trial in scored[:keep]]
        # No finite score is not "every trial was bad": it is a cluster with no rows in the
        # selection slice, so there is nothing to rank its completed trials on. Saving an empty
        # list said nothing and removed the cluster from every later stage in silence.
        reason = None if kept else ('no selection support: the cluster has no rows in the selection '
                                    f'slice, so {trials} completed trial(s) could not be ranked'
                                    if trials else 'no trial produced predictions')
        with open(os.path.join(method_dir, SELECTION_FILE), 'w', encoding='utf-8') as stream:
            json.dump({'trials': kept, 'mae_selection': {t: float(m) for m, t in scored}, 'keep': keep,
                       'reason': reason, 'trials_scored': len(scored), 'trials_found': trials,
                       'selection_until': str(targets.selection.max()) if len(targets.selection) else None},
                      stream, indent=1)
        if reason:
            unsupported.append({'cluster': cluster_name, 'method': method, 'trials_found': trials,
                                'reason': reason})
            log.warning('%s %s is excluded from the ensemble: %s', cluster_name, method, reason)
        else:
            log.info('%s %s: kept %s of %d trials (selection MAE %s)', cluster_name, method, kept, len(scored),
                     [round(float(m), 2) for m, _ in scored[:keep]])
    if unsupported:
        log.warning('%d (cluster, method) unit(s) have no selection support and contribute nothing to the '
                    'ensemble: %s. Their timestamps fall to methods.cluster_coverage_policy.',
                    len(unsupported), ', '.join(f"{u['cluster']}/{u['method']}" for u in unsupported))
    path = os.path.join(static_data['path_model'], 'Results')
    if len(table):
        table['kept'] = [trial in (read_selection(d) or []) for d, trial in zip(table['method_dir'], table['trial'])]
        table = table.drop(columns=['method_dir']).sort_values(['cluster', 'method', 'mae_selection'])
        os.makedirs(path, exist_ok=True)
        table.to_csv(os.path.join(path, 'selection.csv'), index=False, float_format='%.4f')
    if unsupported:
        os.makedirs(path, exist_ok=True)
        pd.DataFrame(unsupported).to_csv(os.path.join(path, 'unsupported_clusters.csv'), index=False)
    return table


# ---------------------------------------------------------------------------------- ensemble

def average_frames(frames):
    """Mean over frames aligned on the union of dates (a date counts the frames that cover it)."""
    frames = [f for f in frames if f is not None and f.shape[0] > 0]
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, axis=0, keys=range(len(frames)))
    return stacked.groupby(level=1).mean().sort_index()


class Ensemble:
    """Method averages per cluster, combiners per cluster and step, cluster averages, models combiner."""

    def __init__(self, static_data, online=False):
        self.static_data = static_data
        self.online = online
        self._targets = None
        self.columns = list(static_data['target_variable']['columns'])
        self.horizon = list(static_data['target_variable']['lags'])
        self.combine_methods = list((static_data.get('combining') or {}).get('methods') or [])
        self.combine_methods = [m for m in self.combine_methods if m in {'kmeans', 'bcp', 'elastic_net'}]
        self.combine_methods = ['mean'] + [m for m in self.combine_methods if m != 'mean']   # the plain mean is always reported
        self.data_type = (static_data.get('combining') or {}).get('data_type') or {}
        self.path_model = static_data['path_model']
        self.problem_type = static_data['type']
        self.predictor = TrialPredictor(static_data, online=online)
        self.inputs = None

    @property
    def targets(self):
        if self._targets is None:
            self._targets = Targets(self.static_data, online=self.online)
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    # ---------------------------------------------------------- level 1: trials -> methods per cluster

    def method_predictions(self, refit=False):
        """{cluster_name: {method: frame}} from the kept trials (mean over the kept trials of a method)."""
        specs = trial_specs(self.static_data, selected_only=True)
        by = {}
        for spec in specs:
            pred = self.predictor.predict(spec, refit=refit)
            by.setdefault(spec['cluster_name'], {}).setdefault(spec['method'], []).append(pred)
        return {cluster: {method: average_frames(frames) for method, frames in methods.items()} for cluster, methods in by.items()}

    def feature_inputs(self):
        if self.inputs is None:
            feeder = DataFeeder(self.static_data, online=self.online, train=False)
            x, _ = feeder.feed_inputs('row_all', merge=self.data_type.get('merge'), compress=self.data_type.get('compress'),
                                      scale_nwp_method=self.data_type.get('scale_nwp_method'),
                                      scale_row_method=self.data_type.get('scale_row_method'),
                                      image_data_type=self.data_type.get('image_data_type'),
                                      scale_img_method=self.data_type.get('scale_img_method'),
                                      feature_selection_method=None,
                                      cluster={'cluster_name': 'global', 'cluster_path': os.path.join(self.path_model, 'global')})
            self.inputs = x['row_all']
            self.inputs.index = get_time_level(self.inputs.index)
            # One row per hour, the one the predictions keep (``normalise_columns``). The rows of
            # a long_term case repeat each hour once per issue day; with the repeats left in, the
            # k-means combiner could not look its features up ("cannot reindex on an axis with
            # duplicate labels") and the combine step failed for any cluster with three models.
            self.inputs = self.inputs[~self.inputs.index.duplicated()]
        return self.inputs

    # ---------------------------------------------------------- combiners

    def combiner_dir(self, scope, combine_method, hor):
        base = os.path.join(scope, 'combine', combine_method) if scope != 'models' else os.path.join(self.path_model, 'combine_models', combine_method)
        return os.path.join(base, f'hour_ahead_{hor}') if self.static_data['horizon_type'] == 'short_term' else base

    def predictors_table(self, frames, hor):
        """One column per predictor for step ``hor``, rows = dates every predictor covers."""
        cols = []
        for name, frame in sorted(frames.items()):
            if frame is None or frame.shape[0] == 0 or frame.shape[1] <= hor:
                continue
            cols.append(frame.iloc[:, hor].rename(name))
        if len(cols) < 2:
            return None
        return pd.concat(cols, axis=1).dropna().clip(0, np.inf)

    def fit_one(self, combine_method, table, y_col, dates, path):
        if combine_method == 'mean':
            return None
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, f'{combine_method}_model.pickle')
        if combine_method == 'bcp':
            model = {'w': bcp_fit(table.values, y_col.values.reshape(-1, 1), table.shape[1])}
        elif combine_method == 'elastic_net':
            model = ElasticNetCV(cv=5, max_iter=200000).fit(table.values, y_col.values)
        elif combine_method == 'kmeans':
            X = self.feature_inputs().reindex(dates).dropna()
            dates = X.index.intersection(table.index)
            if len(dates) < 20:
                return None
            n_clusters = 5 if self.problem_type == 'pv' else 12
            n_clusters = 2 if len(dates) < 50 else min(n_clusters, len(dates) // 10)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            labels = kmeans.fit_predict(X.loc[dates].values)
            model = kmeans_fit(kmeans, labels, table.loc[dates].values, y_col.loc[dates].values.reshape(-1, 1))
        else:
            raise ValueError(f'unknown combine method {combine_method}')
        joblib.dump({'model': model, 'predictors': list(table.columns)}, file)
        return model

    def predict_one(self, combine_method, table, path):
        if table is None:
            return None
        if combine_method == 'mean':
            return table.mean(axis=1)
        file = os.path.join(path, f'{combine_method}_model.pickle')
        if not os.path.exists(file):
            if self.static_data.get('_require_serving_assets'):
                raise ValueError(f'approved serving dependency is missing: {file}')
            return table.mean(axis=1)
        saved = joblib.load(file)
        model, predictors = saved['model'], saved['predictors']
        missing = [p for p in predictors if p not in table.columns]
        if missing:
            if self.static_data.get('_require_serving_assets'):
                raise ValueError(f'approved combiner members missing: {missing}')
            log.warning('%s: predictors %s missing, plain mean used', path, missing)
            return table.mean(axis=1)
        table = table[predictors]
        if combine_method == 'bcp':
            values = (table.values @ model['w'].T).ravel() / np.sum(model['w'])
        elif combine_method == 'elastic_net':
            values = model.predict(table.values)
        else:
            X = self.feature_inputs()
            dates = table.index.intersection(X.index)
            values = kmeans_predict(model, table.loc[dates], X, 1)
            table = table.loc[dates]
        return pd.Series(np.clip(values, 0, None), index=table.index)

    def combine(self, frames, scope, fit=False, fit_dates=None):
        """{combine_method: frame} over the steps for the predictors in ``frames`` (one scope: a cluster or 'models')."""
        out = {}
        for combine_method in self.combine_methods:
            steps = []
            for hor in range(len(self.horizon)):
                table = self.predictors_table(frames, hor)
                if table is None:
                    continue
                path = self.combiner_dir(scope, combine_method, hor)
                if fit:
                    rows = table.index.intersection(fit_dates).intersection(self.targets.y_scaled.index)
                    if len(rows) >= 20:
                        self.fit_one(combine_method, table.loc[rows], self.targets.y_scaled.loc[rows].iloc[:, hor], rows, path)
                series = self.predict_one(combine_method, table, path)
                if series is not None:
                    steps.append(series.rename(self.columns[hor]))
            if steps:
                out[combine_method] = pd.concat(steps, axis=1).sort_index()
        return out

    # ---------------------------------------------------------- the whole ensemble

    def run(self, fit=False, refit_predictions=False):
        """Every level of predictions: returns a nested dict; with ``fit`` the combiners are (re)fitted on the selection slice."""
        adapter = inference_adapter()
        if adapter is not None:
            return adapter('ensemble', static=self.static_data, online=self.online,
                           fit=fit, refit_predictions=refit_predictions)
        fit_dates = self.targets.selection if fit else None
        methods = self.method_predictions(refit=refit_predictions)
        result = {'global': methods.pop('global', {}), 'clusters': {}, 'cluster_average': {}, 'models': {}}
        for cluster_name, frames in methods.items():
            combined = self.combine(frames, self.cluster_dir(cluster_name), fit=fit, fit_dates=fit_dates) if len(frames) > 1 else {}
            result['clusters'][cluster_name] = {**frames, **combined}
        names = sorted({name for frames in result['clusters'].values() for name in frames})
        for name in names:
            result['cluster_average'][name] = average_frames([frames[name] for frames in result['clusters'].values() if name in frames])
        candidates = {f'global_{m}': f for m, f in result['global'].items()}
        candidates.update({f'clusters_{n}': f for n, f in result['cluster_average'].items()})
        if len(candidates) > 1:
            result['models'] = self.combine(candidates, 'models', fit=fit, fit_dates=fit_dates)
        elif candidates:
            result['models'] = {'single': next(iter(candidates.values()))}
        result['models'].update(self.with_global_fallback(result))
        return result

    def with_global_fallback(self, result):
        """``{name + '_global_fallback': frame}`` for the models whose combiner misses timestamps.

        A combiner covers only the rows every candidate covers (``predictors_table`` takes complete
        cases), and a cluster average is missing wherever no *selected* cluster covers the row. A
        cluster whose trials could not be ranked -- no selection support -- drops out entirely, and
        its timestamps go with it. Those rows are predictable: the global model has them.

        The completed series is published as a **separate predictor** rather than filled into the
        combiner, because a number labelled "kmeans" must be kmeans everywhere it is not empty.
        ``methods.cluster_coverage_policy: abstain`` leaves the gaps and reports them.
        """
        if self.static_data.get('cluster_coverage_policy', 'fallback_global') != 'fallback_global':
            return {}
        source = average_frames(list(result['global'].values()))
        if source is None or source.shape[0] == 0:
            return {}
        completed = {}
        for name, frame in result['models'].items():
            if frame is None or frame.shape[0] == 0:
                continue
            missing = source.index.difference(frame.index)
            if not len(missing):
                continue
            filled = pd.concat([frame, source.loc[missing, source.columns[:frame.shape[1]]]
                                .set_axis(frame.columns, axis=1)]).sort_index()
            completed[f'{name}_global_fallback'] = filled
            log.warning('models/%s misses %d timestamp(s) the global model covers; %s_global_fallback '
                        'completes them from the global average (methods.cluster_coverage_policy)',
                        name, len(missing), name)
        return completed

    def cluster_dir(self, cluster_name):
        return trial_layout.cluster_folders(self.static_data)[cluster_name]

    def predict(self):
        """Online: every level from the kept trials and the fitted combiners (scaled); see ``targets.inverse``."""
        return self.run(fit=False, refit_predictions=True)

    # ---------------------------------------------------------- evaluation

    def write_coverage(self, result, trial_predictions, path):
        """How many of the **requested** timestamps each predictor answers, per slice.

        ``evaluation.csv``'s ``n`` counts finite target cells, which is not the same question: a
        predictor that answers nine of ten issue times with sixteen horizons each looks fuller
        than it is. The requested count is the slice itself, so a gap is visible as a gap.
        """
        rows = []
        slices = {'selection': self.targets.selection, 'calibration': self.targets.calibration,
                  'holdout': self.targets.holdout}
        def cover(level, cluster, name, frame):
            if frame is None or frame.shape[0] == 0:
                return
            index = pd.DatetimeIndex(get_time_level(frame.index))
            for slice_name, dates in slices.items():
                requested = len(dates)
                covered = len(index.intersection(dates))
                rows.append({'level': level, 'cluster': cluster, 'predictor': name, 'slice': slice_name,
                             'requested': requested, 'covered': covered, 'missing': requested - covered,
                             'coverage': round(covered / requested, 6) if requested else float('nan')})
        for method, frame in result['global'].items():
            cover('global', 'global', method, frame)
        for cluster_name, frames in result['clusters'].items():
            for name, frame in frames.items():
                cover('combiner' if name in self.combine_methods else 'method', cluster_name, name, frame)
        for name, frame in result['cluster_average'].items():
            cover('cluster_average', 'all', name, frame)
        for name, frame in result['models'].items():
            cover('models', 'all', name, frame)
        frame = pd.DataFrame(rows)
        if not len(frame):
            return frame
        frame.to_csv(os.path.join(path, 'coverage.csv'), index=False)
        holdout = frame[(frame['slice'] == 'holdout') & (frame['missing'] > 0)]
        if len(holdout):
            log.warning('predictors that do not answer every requested holdout timestamp:\n%s',
                        holdout[['level', 'cluster', 'predictor', 'requested', 'covered', 'missing']]
                        .sort_values('missing', ascending=False).head(12).to_string(index=False))
        return frame

    def evaluate(self, result, trial_predictions=None):
        """Long-format table of MAE / RMSE / bias per predictor, level, slice and step; holdout summary."""
        adapter = inference_adapter()
        if adapter is not None:
            return adapter('evaluate', static=self.static_data, result=result, trial_predictions=trial_predictions)
        rows = []
        slices = {'selection': self.targets.selection, 'holdout': self.targets.holdout}
        def add(level, cluster, name, frame):
            if frame is None or frame.shape[0] == 0:
                return
            frame = self.targets.inverse(frame)
            for slice_name, dates in slices.items():
                m = metrics(frame.loc[frame.index.intersection(dates)], self.targets.y, self.targets.rated, self.columns)
                for step, values in m.items():
                    rows.append({'level': level, 'cluster': cluster, 'predictor': name, 'slice': slice_name, 'step': step, **values})
        # every finished trial, once: the kept ones are rows of the table, all of them rows of the
        # hyper-parameter report
        specs = trial_specs(self.static_data) if trial_predictions else []
        for spec in specs:
            if spec['selected']:
                add('trial', spec['cluster_name'], f"{spec['method']}_{spec['trial']}", trial_predictions.get(spec['path_weights']))
        for method, frame in result['global'].items():
            add('global', 'global', method, frame)
        for cluster_name, frames in result['clusters'].items():
            for name, frame in frames.items():
                add('combiner' if name in self.combine_methods else 'method', cluster_name, name, frame)
        for name, frame in result['cluster_average'].items():
            add('cluster_average', 'all', name, frame)
        for name, frame in result['models'].items():
            add('models', 'all', name, frame)
        table = pd.DataFrame(rows)
        path = os.path.join(self.path_model, 'Results'); os.makedirs(path, exist_ok=True)
        table.to_csv(os.path.join(path, 'evaluation.csv'), index=False, float_format='%.4f')
        self.write_coverage(result, trial_predictions, path)
        order = {'models': 0, 'cluster_average': 1, 'global': 2, 'combiner': 3, 'method': 4, 'trial': 5}
        summary = table[(table['slice'] == 'holdout') & (table['step'] == 'all')].copy()
        summary['rank'] = summary['level'].map(order)
        summary = summary.sort_values(['rank', 'mae']).drop(columns='rank')
        summary = summary[['level', 'cluster', 'predictor', 'mae', 'rmse', 'bias', 'n']]
        summary.to_csv(os.path.join(path, 'summary.csv'), index=False, float_format='%.3f')
        whole = summary[summary['level'].isin(['models', 'cluster_average', 'global'])]
        # not "covering the whole period": n counts finite target cells, and coverage.csv is where
        # the question of which requested timestamps a predictor answers is actually answered
        log.info('holdout summary, the levels that span every cluster (percent of rated, all steps; '
                 'see coverage.csv for the timestamps each one answers):\n%s', whole.to_string(index=False))
        if specs:
            # what the trials' hyper-parameters say, with every evaluation; it cannot fail this one
            from eforecast.combine_predictions import hyper_parameters
            hyper_parameters.write(self.static_data, self.targets, specs, trial_predictions, path, metrics=metrics,
                                   current=current_predictions(self.static_data), say=_say)
        return table, summary


    # ---------------------------------------------------------- conformal intervals

    def calibrate(self, result, trial_predictions=None):
        """Fit conformal bands on calibration dates; report coverage on holdout."""
        adapter = inference_adapter()
        if adapter is not None:
            return adapter('calibrate', static=self.static_data, result=result, trial_predictions=trial_predictions)
        from eforecast.combine_predictions.conformal import Calibration, coverage_breakdown, interval_metrics
        coverages = list(self.static_data.get('interval_coverage') or [])
        if not coverages:
            # an empty methods.interval_coverage is a choice, not a missing value: no bands are fitted
            log.info('calibrate: methods.interval_coverage is empty, no prediction intervals are fitted')
            return None, pd.DataFrame()
        calibration = Calibration(coverages, self.columns, self.targets.rated)
        entries = []                                              # (key, level, cluster, predictor, pred, lower, upper)
        for spec in trial_specs(self.static_data, selected_only=True):
            pred = (trial_predictions or {}).get(spec['path_weights'])
            if pred is None or pred.shape[0] == 0:
                continue
            name = f"{spec['method']}_{spec['trial']}"
            entries.append((f"trial/{spec['cluster_name']}/{name}", 'trial', spec['cluster_name'], name, self.targets.inverse(pred), None, None))
            quantiles = self.predictor.quantiles_of(spec)
            if quantiles and len(quantiles) >= 2:
                lo_q, hi_q = min(quantiles), max(quantiles)
                entries.append((f"trial_cqr/{spec['cluster_name']}/{name}", 'trial_cqr', spec['cluster_name'], name, self.targets.inverse(pred),
                                (self.targets.inverse(quantiles[lo_q]), self.targets.inverse(quantiles[hi_q]), (lo_q, hi_q)), None))
        for method, frame in result['global'].items():
            entries.append((f'global/{method}', 'global', 'global', method, self.targets.inverse(frame), None, None))
        for cluster_name, frames in result['clusters'].items():
            for name, frame in frames.items():
                level = 'combiner' if name in self.combine_methods else 'method'
                entries.append((f'clusters/{cluster_name}/{name}', level, cluster_name, name, self.targets.inverse(frame), None, None))
        for name, frame in result['cluster_average'].items():
            entries.append((f'cluster_average/{name}', 'cluster_average', 'all', name, self.targets.inverse(frame), None, None))
        for name, frame in result['models'].items():
            entries.append((f'models/{name}', 'models', 'all', name, self.targets.inverse(frame), None, None))

        rows, breakdown = [], []
        for key, level, cluster, name, pred, cqr, _ in entries:
            if pred is None or pred.shape[0] == 0:
                continue
            if cqr is not None:
                lower_n, upper_n, nominal = cqr
                ok = calibration.add_cqr(key, lower_n, upper_n, nominal, self.targets.y, self.targets.calibration)
            else:
                lower_n = upper_n = None
                ok = calibration.add_split(key, pred, self.targets.y, self.targets.calibration)
            if not ok:
                continue
            for coverage in coverages:
                band = calibration.band(key, coverage, pred, lower_n, upper_n)
                if band is None:
                    continue
                for slice_name, dates in (('selection', self.targets.selection),
                                          ('calibration', self.targets.calibration),
                                          ('holdout', self.targets.holdout)):
                    m = interval_metrics(band[0], band[1], self.targets.y, dates, coverage, self.targets.rated, self.columns)
                    for step, values in m.items():
                        rows.append({'level': level, 'cluster': cluster, 'predictor': name, 'method': calibration.entries[key]['method'],
                                     'coverage_target': coverage, 'slice': slice_name, 'step': step, **values})
                if level in ('models', 'cluster_average', 'global'):
                    # the predictors the holdout summary reports: where the band keeps its promise
                    # and where it does not, by horizon, by season and by regime
                    breakdown += [{'level': level, 'cluster': cluster, 'predictor': name, **entry}
                                  for entry in coverage_breakdown(band[0], band[1], self.targets.y,
                                                                  self.targets.holdout, coverage,
                                                                  self.targets.rated, self.columns)]
        path = os.path.join(self.path_model, 'Results'); os.makedirs(path, exist_ok=True)
        calibration.save(path)
        table = pd.DataFrame(rows)
        if len(table):
            table.to_csv(os.path.join(path, 'intervals.csv'), index=False, float_format='%.4f')
            order = {'models': 0, 'cluster_average': 1, 'global': 2, 'combiner': 3, 'method': 4, 'trial_cqr': 5, 'trial': 6}
            summary = table[(table['slice'] == 'holdout') & (table['step'] == 'all')].copy()
            summary['rank'] = summary['level'].map(order)
            summary = summary.sort_values(['coverage_target', 'rank', 'interval_score']).drop(columns='rank')
            summary = summary[['level', 'cluster', 'predictor', 'method', 'coverage_target', 'coverage', 'width', 'interval_score', 'n']]
            summary.to_csv(os.path.join(path, 'intervals_summary.csv'), index=False, float_format='%.3f')
            whole = summary[summary['level'].isin(['models', 'cluster_average', 'global'])]
            log.info('holdout intervals (coverage achieved vs target, mean width and interval score in percent of rated):\n%s',
                     whole.to_string(index=False))
        else:
            log.warning('no predictor could be calibrated on the %d calibration rows', len(self.targets.calibration))
        if breakdown:
            frame = pd.DataFrame(breakdown)
            frame.to_csv(os.path.join(path, 'interval_coverage.csv'), index=False, float_format='%.4f')
            worst = frame.assign(miss=(frame['coverage'] - frame['coverage_target']).abs())
            worst = worst.sort_values('miss', ascending=False).head(8)
            log.info('measured coverage on the holdout, the eight buckets furthest from their target '
                     '(a single band cannot hold one coverage across seasons and regimes):\n%s',
                     worst[['predictor', 'coverage_target', 'by', 'bucket', 'coverage', 'width', 'n']]
                     .to_string(index=False))
        return calibration, table


def pick(result, path):
    """A prediction frame out of ``Ensemble.run`` by path, e.g. ['models', 'kmeans'],
    ['cluster_average', 'lasso'], ['global', 'MLP'] or ['clusters', 'RBF_rule_0', 'mean']."""
    node = result
    for key in path:
        if key not in node:
            raise KeyError(f'{path}: {key!r} not among {sorted(node)}')
        node = node[key]
    return node


def predict_online(static_data, path=None, intervals=None):
    """Online forecast in power units from the kept trials and the fitted combiners; returns (frame, every level).

    ``path``: ``None`` -> the predictor this case serves (`eforecast.bundle_serving.served_path`:
    `models/kmeans`, or `models/single` when the case trained one method).

    ``intervals``: coverage levels (e.g. [0.8, 0.9]) -> the frame gets ``lower_<c>`` / ``upper_<c>`` columns per
    step from the conformal calibration of that predictor (``Results/calibration.json``, stage ``calibrate``)."""
    adapter = inference_adapter()
    if adapter is not None:
        return adapter('online', static=static_data, path=path, intervals=intervals)
    from eforecast.combine_predictions.conformal import Calibration
    from eforecast.bundle_serving import served_path
    ensemble = Ensemble(static_data, online=True)
    result = ensemble.predict()
    path = tuple(path) if path else served_path(result.get('models') or {})
    frame = ensemble.targets.inverse(pick(result, list(path)))
    if intervals:
        calibration = Calibration.load(os.path.join(static_data['path_model'], 'Results'))
        key = '/'.join(path)
        if calibration is None or key not in calibration.entries:
            log.warning('no conformal calibration for %s: run `run_combine.py --stage calibrate` (intervals skipped)', key)
        else:
            extra = []
            for coverage in intervals:
                band = calibration.band(key, float(coverage), frame)
                if band is None:
                    log.warning('coverage %s not calibrated for %s (available %s)', coverage, key, sorted(calibration.entries[key]['q']))
                    continue
                extra.append(band[0].add_prefix(f'lower_{coverage}_'))
                extra.append(band[1].add_prefix(f'upper_{coverage}_'))
            if extra:
                frame = pd.concat([frame, *extra], axis=1)
    return frame, result
