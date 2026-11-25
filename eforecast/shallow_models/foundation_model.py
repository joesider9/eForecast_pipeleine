# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Zero-shot time-series foundation models as CPU methods of the ensemble (``methods_cpu``).

The past observations of the ``lstm`` data are the context (length = the case's ``global_past_lags``;
a longer history helps a zero-shot model); nothing is trained: ``fit`` only scores the validation and
test slices so the study, the selection and the combiners treat the model like any other trial, and
``predict`` returns the median in the target scale with every quantile kept in
``quantile_predictions`` (so the conformal stage can calibrate it with CQR).

* ``Chronos2`` (Amazon, ``chronos-forecasting`` >= 2): the past calendar as past covariates and, with
  ``covariates`` all, the future NWP columns and the future calendar as known future covariates (their
  past values are unknown to the model and passed as NaN, which Chronos-2 handles).
* ``TimesFM`` (Google, ``timesfm`` >= 3, model 2.5 200M): target history; with ``covariates`` calendar
  and the ``timesfm[xreg]`` extra installed, the calendar columns as dynamic covariates through the
  model's linear XReg step (their pasts are known).  Fast on the CPU.
* ``TiRex`` (NX-AI, ``tirex-ts``): target history only (univariate xLSTM).

``device`` says where a family computes: ``cpu``, or ``auto`` (the default) for the card its worker
was started on, and the CPU when it was started on none. Studies are started on a card by
``cpu_dispatch`` and the trials of the predict stage by ``ensemble.predict_trials``.

Family parameters: ``model_name``, ``covariates`` (all | calendar | none), ``context`` (past steps used,
default all), ``quantile_levels`` ([0.1, 0.5, 0.9]), ``batch_size`` (256), ``device``, ``n_trials`` (1);
``cross_learning`` for Chronos-2, ``xreg_mode`` for TimesFM.
"""
from __future__ import annotations

import os
import time
from contextvars import ContextVar

# TimesFM's XReg step imports JAX, which would otherwise grab most of the GPU memory of the process
# (and starve TiRex or a torch model sharing the card); the regression runs on the CPU anyway.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import pandas as pd

from eforecast.config.capabilities import FOUNDATION_METHODS
from eforecast import execution_policy
from eforecast.compute_context import active_worker, resident_backend
DEFAULT_MODEL = {'Chronos2': 'amazon/chronos-2', 'TimesFM': 'google/timesfm-2.5-200m-pytorch', 'TiRex': 'NX-AI/TiRex'}
_PIPELINES: dict = {}
_CPU_COMPAT = ContextVar('foundation_online_cpu_compat', default=False)


def clear_pipelines():
    """Drop process-local model references; allocation release still waits for exit."""
    _PIPELINES.clear()


def load_pipeline(method: str, model_name: str, device: str = 'cpu', horizon: int = 16, context: int | None = None,
                  batch_size: int = 64, revision: str | None = None):
    """One model per (method, model, device) per process; the weights come from the Hugging Face cache."""
    owner = active_worker.get()
    if owner is None and not _CPU_COMPAT.get():
        raise RuntimeError('load_pipeline runs in a worker process; use FoundationModel.fit/predict')
    device = str(execution_policy.torch_device('cpu' if owner is None else device))
    from eforecast import model_library
    profile = {'Chronos2': 'chronos', 'TimesFM': 'timesfm', 'TiRex': 'tirex'}[method]
    library = {'Chronos2': 'chronos-forecasting', 'TimesFM': 'timesfm', 'TiRex': 'tirex-ts'}[method]
    ref = model_library.resolve(model_name, revision=revision, profile=profile, library=library)
    key = (owner.launch_id, method, ref.identity, device, horizon, context, batch_size) if owner else None
    if key is not None and key in _PIPELINES:
        return _PIPELINES[key]
    if key is not None and _PIPELINES:
        raise RuntimeError('another foundation model needs a worker process of its own')
    import torch
    if method == 'Chronos2':
        from chronos import Chronos2Pipeline
        with model_library.loading(ref):
            model = Chronos2Pipeline.from_pretrained(ref.path, local_files_only=True, device_map=device, dtype=torch.float32)
    elif method == 'TimesFM':
        import timesfm
        model = model_library.timesfm_model(model_name, revision=revision, device=device)
        model.compile(timesfm.ForecastConfig(max_context=max(16, int(context or 512)), max_horizon=max(16, int(horizon)),
                                             normalize_inputs=True, per_core_batch_size=max(1, min(64, batch_size)), use_continuous_quantile_head=True,
                                             fix_quantile_crossing=True, return_backcast=True))
    elif method == 'TiRex':
        model = model_library.tirex_model(model_name, revision=revision, device=device)
    else:
        raise ValueError(f'unknown foundation method {method}')
    if key is not None:
        _PIPELINES[key] = model
    return model


def default_device(method: str) -> str:
    """Where a family computes when nothing says: the card its worker was started on, else the CPU."""
    return str(execution_policy.torch_device('auto'))


def build_items(past_obs, past_calendar, future_data, future_calendar, indices, horizon, covariates='all', context=None,
                obs_names=None, cal_names=None, future_names=None):
    """The Chronos-2 input list for the samples ``indices``.

    ``past_obs`` (n, L, k): column 0 is the target history, the others past covariates; ``past_calendar``
    (n, L, c) past covariates; ``future_calendar`` (n, H, c) future covariates whose past is the past
    calendar when the widths match; ``future_data`` (n, H, m) future covariates with unknown pasts (NaN).
    ``covariates``: 'all' | 'calendar' | 'none'.  ``context``: the last steps of the history (None = all)."""
    n_out = int(horizon)
    obs_names = list(obs_names or [f'obs{j}' for j in range(past_obs.shape[2])])
    cal_names = list(cal_names or [f'cal{j}' for j in range(past_calendar.shape[2] if past_calendar is not None else 0)])
    future_names = list(future_names or [f'nwp{j}' for j in range(future_data.shape[2] if future_data is not None else 0)])

    def horizon_slice(block):                      # the first n_out future rows, padded with the last row when shorter
        if block.shape[0] >= n_out:
            return block[:n_out]
        return np.concatenate([block, np.repeat(block[-1:], n_out - block.shape[0], axis=0)], axis=0)

    items = []
    for i in indices:
        history = past_obs[i, :, 0].astype(np.float32)
        past = {}
        for j in range(1, past_obs.shape[2]):
            past[obs_names[j]] = past_obs[i, :, j].astype(np.float32)
        if past_calendar is not None:
            for j in range(past_calendar.shape[2]):
                past[f'cal_{cal_names[j] if j < len(cal_names) else j}'] = past_calendar[i, :, j].astype(np.float32)
        future = {}
        if covariates in ('all', 'calendar') and future_calendar is not None:
            block = horizon_slice(future_calendar[i])
            same_width = past_calendar is not None and past_calendar.shape[2] == future_calendar.shape[2]
            for j in range(future_calendar.shape[2]):
                name = f'cal_{cal_names[j] if j < len(cal_names) else j}' if same_width else f'fcal_{j}'
                if name not in past:
                    past[name] = np.full(history.shape[0], np.nan, dtype=np.float32)
                future[name] = block[:, j].astype(np.float32)
        if covariates == 'all' and future_data is not None:
            block = horizon_slice(future_data[i])
            for j in range(future_data.shape[2]):
                name = f'nwp_{future_names[j] if j < len(future_names) else j}'
                past[name] = np.full(history.shape[0], np.nan, dtype=np.float32)
                future[name] = block[:, j].astype(np.float32)
        if context:
            c = int(context)
            history = history[-c:]
            past = {k: v[-c:] for k, v in past.items()}
        item = {'target': history}
        if past:
            item['past_covariates'] = past
        if future:
            item['future_covariates'] = future
        items.append(item)
    return items


def routed_forecast(decision, items, dates, n_out, levels, *, scaler_id, identity,
                    adapter_revision, checkpoint, device, parameters=None, timezone='UTC',
                    step_ns=None, static_data=None, task_dir=None):
    """One batch of items through the method's pinned image -- A16-03.

    The arrays cross in the **scaled** space the pipeline works in, and the image is told so:
    it holds no scaler and must not invert. The caller inverts once, afterwards, exactly as it
    does for a native forecast, which is what keeps A15-01's "inversion exactly once" true.

    Returns the same ``[samples, n_out, len(levels)]`` block the native ``_forecast_*`` paths
    return, in the order ``levels`` was given.
    """
    import os as _os
    import shutil
    import tempfile
    from eforecast import runtime_launcher, runtime_routing, runtime_staging

    past_names = sorted({k for item in items for k in (item.get('past_covariates') or {})})
    future_names = sorted({k for item in items for k in (item.get('future_covariates') or {})})
    context = int(len(items[0]['target']))
    samples = len(items)
    history = np.full((samples, context, 1 + len(past_names)), np.nan, dtype='float64')
    future = (np.full((samples, n_out, len(future_names)), np.nan, dtype='float64')
              if future_names else None)
    for i, item in enumerate(items):
        history[i, :, 0] = item['target']
        for j, name in enumerate(past_names):
            values = (item.get('past_covariates') or {}).get(name)
            if values is not None:
                history[i, :, 1 + j] = values
        for j, name in enumerate(future_names):
            values = (item.get('future_covariates') or {}).get(name)
            if values is not None:
                future[i, :, j] = values

    ordered = sorted({float(q) for q in levels})
    step = step_ns if step_ns is not None else runtime_routing.step_nanoseconds(dates, static_data)
    history_index, forecast_index = runtime_routing.window_index(dates, context, n_out, step, timezone)
    masked = not (np.isfinite(history).all() and (future is None or np.isfinite(future).all()))

    # The attempt owns its staging directory, named from its identity alone, so batches of
    # one attempt reuse what is already staged and nothing is addressed by an owner-specific
    # absolute path. A caller may still name one (a drill, a test). A directory we had to
    # invent because the identity could not be used is temporary and is removed here;
    # an attempt's own directory is left for the attempt's cleanup to reap.
    temporary = False
    if task_dir is None:
        try:
            task_dir = str(runtime_staging.attempt_paths(identity)['input'])
        except Exception:                                               # noqa: BLE001
            task_dir = tempfile.mkdtemp(prefix='eforecast-routed-')
            temporary = True
    else:
        task_dir = str(task_dir)
    budget = _os.environ.get('EFORECAST_STAGING_BUDGET_BYTES')
    try:
        request = runtime_launcher.prepare(
            task_dir, method=decision['served'], history=history,
            history_index=history_index, forecast_index=forecast_index,
            targets=['target'], target_units=['global_scaled'], timezone=timezone,
            frequency_ns=int(step), identity=identity,
            runtime={'id': decision['runtime'], 'revision': decision.get('runtime_revision') or (decision['image'] or '').split(':')[-1]},
            component={'adapter_revision': adapter_revision, 'parameters': dict(parameters or {}),
                       'checkpoint': checkpoint, 'artifacts': []},
            device=device, quantiles=tuple(ordered),
            past_covariates=past_names, future_covariates=future_names, future=future,
            target_scale='global_scaled', output_scale='global_scaled',
            scaler_id=scaler_id, missing='mask' if masked else 'forbid',
            budget_bytes=int(budget) if budget else None)
        # A reader on a card (``bounded_pools.CardReaders``) stands where its worker stands, and
        # that worker has no card: the card this process computes on is the one its container
        # is given. A worker that was itself started on a card names it to the launcher.
        policy = execution_policy.current()
        card = policy['effective'].get('device_uuid') if policy and device == 'cuda' else None
        result = runtime_launcher.launch(task_dir, request, reference=decision['reference'],
                                         gpus=f'device={card}' if card else None)
        _, spread = runtime_launcher.collect(task_dir, result)
    finally:
        if temporary:
            shutil.rmtree(task_dir, ignore_errors=True)
    if spread is None:
        raise RuntimeError(f'{decision["method"]}: the image returned no quantiles')
    place = {q: k for k, q in enumerate(ordered)}
    return np.stack([spread[:, :, 0, place[float(q)]] for q in levels], axis=2)


class FoundationModel:
    """The ``ShallowModel`` interface (fit / predict / save / load) for a zero-shot foundation model."""

    def __init__(self, static_data, path_weights, params=None, n_jobs=1, refit=False, train=False, online=False):
        self.best_mae_val = None
        self.best_mae_test = None
        self.sse_val = None
        self.sse_test = None
        self.is_trained = False
        self.refit = refit
        self.static_data = static_data
        self.rated = static_data['rated']
        self.n_jobs = n_jobs
        self.online, self.train = online, train
        self.quantile_predictions = None
        if params is not None:
            self.params = dict(params)
            self.method = self.params['method']
            self.cluster_name = self.params['cluster_name']
        self.path_weights = path_weights
        if not refit:
            try:
                self.load()
            except ImportError:
                pass
        if not hasattr(self, 'params'):
            raise RuntimeError('The foundation model has no params. Provide them for the first run')
        # The coordinator owns metadata only; arrays/scalers load after admission.
        self.data_feeder = None
        self.online, self.train = online, train
        self.model = self.method                                    # nothing fitted: the name stands for the model

    # ------------------------------------------------------------------ data
    def load_data(self, with_target=False):
        if self.data_feeder is None:
            from eforecast.datasets.data_feeder import DataFeeder
            self.data_feeder = DataFeeder(self.static_data, online=self.online, train=self.train)
            # Foundation families consume the materialized LSTM context even
            # when the case's deep architectures only declare row branches.
            self.data_feeder.dataset_tags.add('lstm')
        cluster = {'cluster_name': self.params['cluster_name'], 'cluster_path': self.params['cluster_dir']}
        self.data_feeder.reader_consumer = {'kind': 'foundation', 'method': self.method,
                                           'cross_learning': bool(self.params.get('cross_learning', False))}
        x, mdata = self.data_feeder.feed_inputs('lstm', scale_row_method=self.params.get('scale_row_method'),
                                                feature_selection_method=None, cluster=cluster,
                                                inverse_transform=True, transform_calendar=False)
        X, metadata = x['lstm'], mdata['lstm']
        if not with_target:
            return X, metadata
        y = self.data_feeder.feed_target(inverse=False)
        common = metadata['dates'].intersection(y.index)
        idx = metadata['dates'].get_indexer(common)
        from eforecast.datasets.tensor_views import TensorView
        X = {k: v[idx] for k, v in X.items() if isinstance(v, (np.ndarray, TensorView))}
        metadata = dict(metadata, dates=common)
        return X, y.loc[common], metadata

    def _names(self, metadata):
        obs = metadata.get('past_variables') or metadata.get('lstm_vars') or None
        cal = metadata.get('cal_vars') or None
        future = metadata.get('future_variables') or None
        return obs, cal, future

    def _attempt_identity(self):
        """The contract's identity, taken from the worker this runs in when there is one.

        Routed work is the same attempt as native work, so it carries the worker's run, task
        and attempt rather than inventing its own. The contract is fixed in the model images
        and still names two fields from the time work was reserved: ``allocation_id`` carries
        what names this start of the worker, and ``fence`` is always 1.
        Outside a worker (a direct call, a test) the identity is derived so the request is
        still well formed and traceable.
        """
        import hashlib
        from eforecast.compute_worker import current_worker

        case = str(self.static_data.get('_id') or self.static_data.get('project_name') or 'case')
        context = current_worker()
        if context is not None:
            return {'case_id': case, 'run_id': str(context.run_id),
                    'task_id': str(context.task_id), 'attempt_id': str(context.attempt_id),
                    'allocation_id': str(context.launch_id), 'fence': 1}
        token = hashlib.sha256(
            f'{case}|{self.method}|{self.path_weights}'.encode()).hexdigest()[:16]
        return {'case_id': case, 'run_id': f'unattached-{token}', 'task_id': str(self.method),
                'attempt_id': token, 'allocation_id': f'unattached-{token}', 'fence': 1}

    def _adapter_revision(self):
        """The digest of the adapter module that will serve this request.

        Recorded in the result, so a forecast can be attributed to the adapter code that
        produced it rather than to the image alone.
        """
        import hashlib
        from pathlib import Path
        from eforecast import runtime_launcher
        from eforecast.runtime_adapters import adapter_name

        served = runtime_launcher.CASE_METHODS[self.method]
        module = (Path(__file__).resolve().parents[1] / 'runtime_adapters'
                  / f'{adapter_name(served)}.py')
        return hashlib.sha256(module.read_bytes()).hexdigest()

    def _use_present_threads(self):
        """Tell torch how many threads this work has now.

        A study that runs beside others is started with every thread it may ever be given, and
        is told its share of them in a file that grows as the others finish (``cpu_dispatch``).
        Torch was told the first number when the worker started. Left at that, sixteen studies
        side by side compute with sixteen threads each, and none of them gets anywhere.
        """
        import torch
        threads = execution_policy.network_threads() or max(1, int(self.n_jobs or 1))
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)

    def _forecast(self, X, metadata, indices):
        """(median frame, {quantile: frame}) in the target scale for the samples ``indices``."""
        self._use_present_threads()
        columns = list(self.static_data['target_variable']['columns'])
        n_out = len(columns)
        levels = [float(q) for q in (self.params.get('quantile_levels') or [0.1, 0.5, 0.9])]
        if 0.5 not in levels:
            levels = sorted(levels + [0.5])
        obs_names, cal_names, future_names = self._names(metadata)
        covariates = self.params.get('covariates', 'all')
        device = self.params.get('device') or default_device(self.method)
        model_name = self.params.get('model_name') or DEFAULT_MODEL[self.method]
        batch_size = int(self.params.get('batch_size', 256))
        from eforecast import runtime_routing
        decision = runtime_routing.route(self.method, static_data=self.static_data, params=self.params)
        routed = decision['where'] == runtime_routing.IMAGE
        model = None
        checkpoint = None
        runtime_parameters = dict(self.params.get('runtime_parameters') or {})
        if routed:
            from eforecast.runtime_launcher import resolve_checkpoint
            checkpoint = resolve_checkpoint(decision['served'], model_name, self.params.get('model_revision'))
            if checkpoint:
                self.params['model_revision'] = checkpoint['revision']
            elif decision['served'] == 'TimeGPT':
                runtime_parameters.setdefault('model', model_name)
        if not routed:
            from eforecast import model_library
            with model_library.use_inputs(getattr(self, 'model_library', ())) as used:
                model = load_pipeline(self.method, model_name, device, n_out,
                                      self.params.get('context') or X['past_obs'].shape[1], batch_size=batch_size,
                                      revision=self.params.get('model_revision'))
            self.model_library = used
            if self.model_library:
                self.params['model_revision'] = self.model_library[-1]['source']['revision']
        scaler_id = f"target_{self.static_data['scale_target_method']}"
        # A routed family may take no exogenous inputs at all -- Lag-Llama's checkpoint is
        # univariate and its adapter says so -- and the contract refuses a request carrying
        # any, naming the field. Withholding what it cannot take is the caller's job: the
        # item builder adds the past calendar unconditionally, and `covariates='none'`
        # suppresses only the *future* ones.
        exogenous = True
        if routed:
            from eforecast.runtime_adapters import declared
            exogenous = bool(declared(decision['served']).get('future_covariates'))
        from eforecast.datasets.tensor_views import contains, batch as tensor_batch
        mapped = contains(X)
        def forecast(block, rows, dates=None):
            self._use_present_threads()         # every batch of a mapped dataset asks again
            if routed and not exogenous:
                past = block['past_obs']
                items = build_items(past[:, :, :1] if past.shape[2] > 1 else past,
                                    None, None, None, rows, n_out, covariates='none',
                                    context=self.params.get('context'),
                                    obs_names=obs_names[:1] if obs_names else None)
            else:
                items = build_items(block['past_obs'], block.get('past_calendar'), block.get('future_data'), block.get('future_calendar'), rows, n_out,
                                    covariates=covariates, context=self.params.get('context'),
                                    obs_names=obs_names, cal_names=cal_names, future_names=future_names)
            if routed:
                return routed_forecast(
                    decision, items, dates, n_out, levels, scaler_id=scaler_id,
                    identity=self._attempt_identity(), adapter_revision=self._adapter_revision(),
                    checkpoint=checkpoint,
                    # The image is asked for a kind of device, never a card: ``auto``, an
                    # ordinal or nothing resolves to where this worker computes, as it does
                    # for a family loaded here, and the launcher names the card.
                    device=execution_policy.torch_device(device).type, parameters=runtime_parameters,
                    timezone=self.static_data.get('local_timezone') or 'UTC',
                    static_data=self.static_data)
            if self.method == 'Chronos2':
                return self._forecast_chronos(model, items, n_out, levels, batch_size, bool(self.params.get('cross_learning', False)))
            if self.method == 'TimesFM': return self._forecast_timesfm(model, items, n_out, levels, covariates)
            return self._forecast_tirex(model, items, n_out, levels, batch_size)
        if mapped:
            if self.method == 'TimesFM' or self.params.get('cross_learning'):
                raise ValueError('coupled foundation forecasts require the ordinary complete-context reader')
            # Keep per-model batch grouping, context and ordering. Only the
            # independent Chronos/TiRex item batches are eligible for this port.
            # Item construction duplicates context/covariate buffers and adds
            # padded future arrays. Keep a conservative numeric allowance for
            # these companions before gathering the first leaf.
            shapes = [v.shape for v in X.values() if hasattr(v, 'working_bytes_per_row')]
            steps = max([n_out, *[s[1] for s in shapes]])
            widths = 1 + sum(int(np.prod(s[2:])) for s in shapes)
            item_bytes = 16 * steps * widths * 4
            output_bytes = 2 * len(indices) * n_out * len(levels) * 8
            raw = np.concatenate([forecast(tensor_batch(X, indices[start:start + batch_size],
                                        extra_bytes=output_bytes + len(indices[start:start + batch_size]) * item_bytes), range(len(indices[start:start + batch_size])),
                                        metadata['dates'][indices[start:start + batch_size]])
                                  for start in range(0, len(indices), batch_size)], axis=0)
        else:
            raw = forecast(X, indices, metadata['dates'][indices])
        levels_order = np.argsort(levels)
        levels = [levels[k] for k in levels_order]
        raw = np.sort(raw[:, :, levels_order], axis=2)          # the models' quantiles may cross: keep them ordered
        dates = metadata['dates'][indices]
        scaler_name = f"target_{self.static_data['scale_target_method']}"
        frames = {}
        for j, q in enumerate(levels):
            frame = pd.DataFrame(np.clip(raw[:, :, j], 0, None), index=dates, columns=columns)
            frames[q] = self.data_feeder.pipeline.scaler.transform(frame, scaler_name).clip(lower=0)
        return frames[0.5], frames

    @staticmethod
    def _forecast_chronos(model, items, n_out, levels, batch_size, cross_learning=False):
        quantiles, _ = model.predict_quantiles(items, prediction_length=n_out, quantile_levels=levels, batch_size=batch_size,
                                               cross_learning=cross_learning)
        return np.concatenate([t.detach().cpu().float().reshape(1, n_out, len(levels)) for t in quantiles], axis=0)

    @staticmethod
    def _interpolate_levels(quantiles, model_levels, levels):
        """(n, H, len(levels)) from the model's own quantile grid by linear interpolation between levels."""
        grid = np.asarray(model_levels, dtype=float)
        out = np.empty((quantiles.shape[0], quantiles.shape[1], len(levels)), dtype=np.float32)
        for j, q in enumerate(levels):
            if q <= grid[0]:
                out[:, :, j] = quantiles[:, :, 0]
            elif q >= grid[-1]:
                out[:, :, j] = quantiles[:, :, -1]
            else:
                k = int(np.searchsorted(grid, q))
                w = (q - grid[k - 1]) / (grid[k] - grid[k - 1])
                out[:, :, j] = (1 - w) * quantiles[:, :, k - 1] + w * quantiles[:, :, k]
        return out

    def _forecast_timesfm(self, model, items, n_out, levels, covariates):
        """TimesFM 2.5: quantile head columns 1..9 are the 0.1 .. 0.9 quantiles (column 0 the mean); the
        calendar covariates go through the XReg step when the extra is installed.

        ``load_pipeline`` compiles with ``return_backcast=True``, so a returned block is the
        reconstruction of the context followed by the horizon, not the horizon alone: for a
        64-step context and a 4-step horizon the model answers 36 steps.  The forecast is
        therefore the final ``n_out`` rows.  Reading the leading rows returns the fitted
        history instead of the prediction, and when no backcast is present the block is
        exactly ``n_out`` long, so the same tail slice is correct either way.
        """
        contexts = [np.asarray(it['target'], dtype=np.float32) for it in items]
        grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dynamic = {}
        if covariates in ('all', 'calendar'):
            names = [k for k in items[0].get('future_covariates', {}) if k.startswith('cal_')
                     and not np.isnan(items[0]['past_covariates'][k]).any()]
            dynamic = {k: [np.concatenate([it['past_covariates'][k], it['future_covariates'][k]]).astype(np.float32).tolist()
                           for it in items] for k in names}
        if dynamic:
            try:
                out = model.forecast_with_covariates(inputs=[c.tolist() for c in contexts], dynamic_numerical_covariates=dynamic,
                                                     xreg_mode=self.params.get('xreg_mode', 'xreg + timesfm'), force_on_cpu=True)
                point = np.asarray(out[0], dtype=np.float32)                           # (n, H) point forecast with covariates
                _, quant = model.forecast(horizon=n_out, inputs=contexts)             # the spread from the plain forecast
                quant = np.asarray(quant, dtype=np.float32)[:, -n_out:, 1:10]
                shift = point[:, -n_out:] - quant[:, :, 4]                             # move the quantiles onto the covariate point
                quant = quant + shift[:, :, None]
                return self._interpolate_levels(quant, grid, levels)
            except ImportError as error:
                print(f'TimesFM covariates skipped ({str(error)[:80]}): install timesfm[xreg] for the calendar regression')
        _, quant = model.forecast(horizon=n_out, inputs=contexts)
        quant = np.asarray(quant, dtype=np.float32)[:, -n_out:, 1:10]
        return self._interpolate_levels(quant, grid, levels)

    def _forecast_tirex(self, model, items, n_out, levels, batch_size):
        import torch
        contexts = [torch.as_tensor(np.asarray(it['target'], dtype=np.float32)) for it in items]
        quant, _ = model.forecast(context=contexts, prediction_length=n_out, batch_size=batch_size, output_type='numpy')
        grid = list(getattr(model, 'quantiles', None) or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        return self._interpolate_levels(np.asarray(quant, dtype=np.float32)[:, :n_out, :], grid, levels)

    # ------------------------------------------------------------------ the ShallowModel interface
    def fit(self, cv_masks):
        """Zero-shot: score the validation and test slices (the study value), nothing is trained."""
        if active_worker.get() is None:
            adapter = resident_backend('foundation')
            if adapter is None:
                raise RuntimeError('foundation fitting needs the host compute adapter (run_training)')
            return adapter(self, 'fit', masks=cv_masks)
        X, y, metadata = self.load_data(with_target=True)
        results = {}
        start = time.time()
        for name, mask in (('val', cv_masks[1]), ('test', cv_masks[2])):
            dates = mask.intersection(metadata['dates'])
            if len(dates) < 10:
                self.model = None
                return 1
            idx = metadata['dates'].get_indexer(dates)
            median, _ = self._forecast(X, metadata, idx)
            truth = y.loc[dates].values
            norm = 1 if self.rated is not None else truth
            err = (median.values - truth) / norm
            results[name] = (float(np.mean(np.abs(err))), float(np.sum(np.square(err))))
        self.best_mae_val, self.sse_val = results['val']
        self.best_mae_test, self.sse_test = results['test']
        self.duration = time.time() - start
        self.is_trained = True
        self.save()
        return self.sse_test

    def predict(self, cluster_dates=None):
        if active_worker.get() is None:
            if not self.online:
                adapter = resident_backend('foundation')
                if adapter is None:
                    raise RuntimeError('offline foundations need the host compute adapter (run_combine)')
                return adapter(self, 'predict', dates=cluster_dates)
            # Preserve the lightweight standalone CPU image: this path may neither
            # use CUDA nor retain a process-wide model cache between calls.
            token = _CPU_COMPAT.set(True)
            try:
                return self._predict_local(cluster_dates)
            finally:
                _CPU_COMPAT.reset(token)
        return self._predict_local(cluster_dates)

    def _predict_local(self, cluster_dates=None):
        X, metadata = self.load_data()
        dates = metadata['dates'] if cluster_dates is None else cluster_dates.intersection(metadata['dates'])
        cols = [f'{self.method}_{col}' for col in self.static_data['target_variable']['columns']]
        if len(dates) == 0:
            return pd.DataFrame(columns=cols)
        idx = metadata['dates'].get_indexer(dates)
        median, frames = self._forecast(X, metadata, idx)
        self.quantile_predictions = {q: f.set_axis(cols, axis=1) for q, f in frames.items()}
        return median.set_axis(cols, axis=1)

    def load(self):
        import joblib
        file = os.path.join(self.path_weights, 'net_weights.pickle')
        if not os.path.exists(file):
            raise ImportError(f'Cannot load {file}')
        self.__dict__.update(joblib.load(file))

    def save(self):
        import joblib
        keep = {k: v for k, v in self.__dict__.items() if k not in {'static_data', 'path_weights', 'refit', 'data_feeder', 'quantile_predictions', 'online', 'train'}}
        joblib.dump(keep, os.path.join(self.path_weights, 'net_weights.pickle'))
