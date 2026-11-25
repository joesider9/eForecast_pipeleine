# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Serve Lag-Llama through A16-01's runtime contract -- A16-02.

Lag-Llama gets an image of its own rather than a place in the foundation image, and the
reason is measured rather than assumed: it caps `gluonts[torch]` at 0.14.4, which drags
pandas to 2.1.4 and numpy to 1.26.4, while foundationforecast resolves at gluonts 0.17.0
and pandas 2.3.3.  Asking pip for both is ResolutionImpossible, and asking pipe_env for
either would undo the A13-05 pandas upgrade.  Three incompatible stacks, three homes.

It is also not on PyPI -- only the GitHub repository -- and its weights come from the
ungated Hugging Face repo time-series-foundation-models/Lag-Llama.

The gluonts imports are deferred, so capability() answers preflight without loading torch
or reading a checkpoint.
"""
from __future__ import annotations

import os

import numpy as np
from .checkpoints import pinned_forecast

DISTRIBUTION = 'lag-llama'

METHOD = 'Lag-Llama'
CHECKPOINT = 'time-series-foundation-models/Lag-Llama'
WEIGHTS = 'lag-llama.ckpt'
# the upstream reference uses 100 sample paths; quantiles below are empirical over them
DEFAULT_SAMPLES = 100

# The checkpoint is a pickle, and torch >= 2.6 refuses to unpickle arbitrary classes from
# one unless they are allowlisted.  Upstream predates that change and loads with
# weights_only=False, which executes whatever the file contains.  These are the only two
# globals this checkpoint actually needs -- discovered by loading it, not guessed -- so the
# safe default is kept and exactly these are permitted.
SAFE_GLOBALS = ('gluonts.torch.distributions.studentT.StudentTOutput',
                'gluonts.torch.modules.loss.NegativeLogLikelihood')


class AdapterError(ValueError):
    """The request cannot be served by this image, with the reason a colleague needs."""


def families():
    return [METHOD]


def capability(method):
    if method != METHOD:
        raise AdapterError(f'{method}: this image serves {METHOD} only')
    return {
        'methods': [METHOD],
        'operations': ['predict'],       # the checkpoint is pretrained; finetuning is later work
        'devices': ['cpu', 'cuda'],
        'past_covariates': False,        # univariate checkpoint: it takes lags and time features
        'future_covariates': False,      # claiming NWP covariates here would be untrue
        'multiple_targets': True,        # each target travels as its own series
        'missing_mask': False,
        'quantiles': True,               # empirical over the sampled paths
    }


def _weights(checkpoint=None):
    """The checkpoint file inside the mounted cache, or a remedy naming what is missing."""
    cache = os.environ.get('LAG_LLAMA_WEIGHTS')
    if cache and os.path.isfile(cache) and checkpoint is None:
        return cache
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:                               # pragma: no cover - in-image path
        raise AdapterError('huggingface_hub is not installed in this image') from error
    try:
        return hf_hub_download(repo_id=(checkpoint or {}).get('id', CHECKPOINT), filename=WEIGHTS,
                               revision=(checkpoint or {}).get('revision'))
    except Exception as error:
        raise AdapterError(
            f'{WEIGHTS} is not in this runtime\'s cache and could not be fetched from '
            f'{CHECKPOINT} ({type(error).__name__}). Pre-stage it into the mounted cache, or '
            'set LAG_LLAMA_WEIGHTS to the checkpoint path.') from error


def _allowlist():
    """Permit the checkpoint's own gluonts classes, and nothing else."""
    import importlib
    import torch

    allowed = []
    for path in SAFE_GLOBALS:
        module, attribute = path.rsplit('.', 1)
        try:
            allowed.append(getattr(importlib.import_module(module), attribute))
        except (ImportError, AttributeError) as error:
            raise AdapterError(f'{path} is not in this image\'s gluonts '
                               f'({type(error).__name__}); the checkpoint cannot be read '
                               'without it') from error
    torch.serialization.add_safe_globals(allowed)


def freq_alias(frequency_ns):
    from eforecast.runtime_adapters.foundation import freq_alias as alias
    return alias(frequency_ns)


@pinned_forecast
def forecast(request, arrays):
    """Run one predict request and return (prediction, quantiles) in contract order."""
    import pandas as pd
    import torch
    from gluonts.dataset.pandas import PandasDataset

    try:
        from lag_llama.gluon.estimator import LagLlamaEstimator
    except ImportError as error:                               # pragma: no cover - in-image path
        raise AdapterError('lag_llama is not installed in this image; it is the '
                           'eforecast-lagllama image that serves this method') from error

    data = request['input']
    samples, horizon, targets = data['samples'], data['horizon'], data['targets']
    context = data['context']
    quantiles = request['output']['quantiles']
    parameters = dict(request['component']['parameters'] or {})
    paths = int(parameters.pop('num_samples', DEFAULT_SAMPLES))
    if parameters:
        raise AdapterError(f'component.parameters: {", ".join(sorted(parameters))} is not a '
                           f'{METHOD} adapter parameter (num_samples is)')

    device = torch.device('cuda' if request['device'] == 'cuda' else 'cpu')
    if request['device'] == 'cuda' and not torch.cuda.is_available():
        raise AdapterError('device cuda: this container was not given a GPU')

    path = _weights(request['component']['checkpoint'])
    _allowlist()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as error:
        import re
        unexpected = re.search(r'GLOBAL ([\w.]+) was not an allowed global', str(error))
        if unexpected:
            raise AdapterError(
                f'the checkpoint at {CHECKPOINT} wants to unpickle '
                f'{unexpected.group(1)}, which this adapter does not permit. A checkpoint '
                'that needs a class beyond SAFE_GLOBALS is a different file from the one '
                'this adapter was qualified against -- check which revision was fetched '
                'rather than widening the allowlist.') from error
        raise
    recorded = checkpoint['hyper_parameters']['model_kwargs']
    estimator = LagLlamaEstimator(
        ckpt_path=path, prediction_length=horizon, context_length=context,
        input_size=recorded['input_size'], n_layer=recorded['n_layer'],
        n_embd_per_head=recorded['n_embd_per_head'], n_head=recorded['n_head'],
        scaling=recorded['scaling'], time_feat=recorded['time_feat'],
        rope_scaling=None, batch_size=1, num_parallel_samples=paths, device=device)
    predictor = estimator.create_predictor(estimator.create_transformation(),
                                           estimator.create_lightning_module())

    history, history_index = arrays['history'], arrays['history_index']
    forecast_index = arrays['forecast_index']
    frequency = freq_alias(data['frequency_ns'])

    # one series per (sample, target), in a fixed order, so the predictor's output order is
    # the order we built -- gluonts predicts a stream and does not label what it returns
    order, frames = [], {}
    for sample in range(samples):
        stamps = pd.to_datetime(history_index[sample], unit='ns', utc=True)
        for position, name in enumerate(targets):
            key = f'{sample}|{position}'
            order.append((key, sample, position))
            frames[key] = pd.DataFrame(
                {'target': history[sample, :, position].astype('float32')},
                index=pd.DatetimeIndex(stamps).tz_convert(None))
    dataset = PandasDataset(frames, target='target', freq=frequency)

    prediction = np.empty((samples, horizon, len(targets)), dtype='float64')
    spread = (np.empty((samples, horizon, len(targets), len(quantiles)), dtype='float64')
              if quantiles else None)
    produced = list(predictor.predict(dataset, num_samples=paths))
    if len(produced) != len(order):
        raise AdapterError(f'{METHOD}: {len(produced)} forecasts for {len(order)} series')

    for (key, sample, position), outcome in zip(order, produced):
        drawn = np.asarray(outcome.samples, dtype='float64')
        if drawn.shape[-1] != horizon:
            raise AdapterError(f'{METHOD}: returned {drawn.shape[-1]} steps for {key}, not {horizon}')
        # The forecast must start where the request's forecast index starts: a model that
        # returned reconstructed history would line up in shape but not in time, which is
        # exactly the TimesFM backcast defect A13-01 found.
        #
        # Compared at the frequency's own resolution, not exactly: gluonts indexes by period,
        # so a start that is not on a period boundary comes back snapped to one.  That is
        # rounding, not a backcast -- a backcast is out by the whole context window, which is
        # orders of magnitude more than one step.
        start = pd.Timestamp(outcome.start_date.to_timestamp(), tz='UTC')
        expected = pd.Timestamp(forecast_index[sample][0], unit='ns', tz='UTC')
        step = pd.Timedelta(data['frequency_ns'], unit='ns')
        if abs(start - expected) >= step:
            raise AdapterError(f'{METHOD}: the forecast for {key} starts at {start} but the '
                               f'request asked for {expected}, a gap of {abs(start - expected)} '
                               f'against a {step} step; this is not the window that was asked '
                               'for, and a gap of the whole context is a backcast')
        prediction[sample, :, position] = drawn.mean(axis=0)
        for index, level in enumerate(quantiles):
            spread[sample, :, position, index] = np.quantile(drawn, level, axis=0)
    return prediction, spread
