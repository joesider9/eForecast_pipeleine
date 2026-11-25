# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Serve the FoundationForecast families through A16-01's runtime contract.

One image, one adapter, thirteen families.  The package already resolves their mutual pins,
so TimeGPT, Timestral, T0, Chronos, TimesFM, TiRex, Toto, Moirai, Sundial, TabPFN, Tafsut,
FlowState and PatchTST-FM share one environment instead of needing thirteen -- which is why
A16-02 builds this image rather than one image per family.

The family classes are resolved by name at forecast time, not imported at module scope, so
capability() answers preflight and doctor without loading torch, reaching Hugging Face or
touching a checkpoint.  A family the installed version does not export is reported as a
missing runtime, not an AttributeError.
"""
from __future__ import annotations

import re

import numpy as np
from .checkpoints import pinned_forecast

DISTRIBUTION = 'foundationforecast'

NS = 1_000_000_000

# method name -> the class foundationforecast.models exports, its default checkpoint, and
# whether that default accepts known-future covariates (the package's "native exog" list).
FAMILIES = {
    'Chronos':     {'cls': 'Chronos', 'repo': 'amazon/chronos-2', 'future': True},
    'TimesFM':     {'cls': 'TimesFM', 'repo': 'google/timesfm-2.5-200m-pytorch', 'future': False},
    'TiRex':       {'cls': 'TiRex', 'repo': 'NX-AI/TiRex', 'future': False},
    # TiRex-2 is a checkpoint of the same class, not a class of its own: the installed
    # foundationforecast exports TiRex only, and native exog needs a TiRex-2 repo_id
    'TiRex2':      {'cls': 'TiRex', 'repo': 'NX-AI/TiRex-2', 'future': True},
    'Timestral':   {'cls': 'Timestral', 'repo': 'mistralai/Timestral-209M', 'future': True},
    # T0's weights are Apache-2.0 but the repo is gated: the terms must be accepted on
    # Hugging Face once, and a token presented on every pull
    'T0':          {'cls': 'T0', 'repo': 'theforecastingcompany/t0-alpha', 'future': True,
                    'credential': 'hf_token', 'environment': 'HF_TOKEN'},
    'TimeGPT':     {'cls': 'TimeGPT', 'repo': None, 'future': True, 'model': 'timegpt-1',
                    'credential': 'nixtla_api_key', 'environment': 'NIXTLA_API_KEY'},
    'Toto':        {'cls': 'Toto', 'repo': 'Datadog/Toto-Open-Base-1.0', 'future': False},
    'Moirai':      {'cls': 'Moirai', 'repo': 'Salesforce/moirai-2.0-R-small', 'future': False,
                    'licence': 'CC-BY-NC-4.0: lab research only, not revenue-generating use'},
    'Sundial':     {'cls': 'Sundial', 'repo': 'thuml/sundial-base-128m', 'future': False},
    'TabPFN':      {'cls': 'TabPFN', 'repo': None, 'future': False,
                    'credential': 'tabpfn_token', 'environment': 'TABPFN_TOKEN',
                    'licence': 'TabPFN non-commercial: lab research only'},
    'Tafsut':      {'cls': 'Tafsut', 'repo': 'Tafsut-FM/tafsut-univariate-base', 'future': False},
    'FlowState':   {'cls': 'FlowState', 'repo': 'ibm-research/flowstate', 'future': False},
    'PatchTSTFM':  {'cls': 'PatchTSTFM', 'repo': 'ibm-research/patchtst-fm-r1', 'future': False,
                    'licence': 'CC-BY-NC-SA-4.0 for r1 checkpoints: lab research only'},
}

# context ceilings are a property of a checkpoint rather than of the adapter, so the adapter
# declares none and lets the family refuse what it cannot take
CAPABILITY_LIMITS = {}


class AdapterError(ValueError):
    """The request cannot be served by this image, with the reason a colleague needs."""


def families():
    """The method names this adapter knows, whether or not the image can serve them."""
    return sorted(FAMILIES)


def capability(method):
    """What this adapter supports for one method, in the shape validate_capability reads."""
    family = FAMILIES.get(method)
    if family is None:
        raise AdapterError(f'{method}: not a FoundationForecast family; this image serves {", ".join(families())}')
    exogenous = bool(family['future'])
    declared = {
        'methods': [method],
        'operations': ['predict'],          # these are pretrained checkpoints; fit is A16 later work
        'devices': ['cpu', 'cuda'],
        # The package carries a covariate's history as a column of `df` and its future in
        # `X_df`, so past covariates are supported exactly where future ones are -- a family
        # with no native exog takes neither.
        'past_covariates': exogenous,
        'future_covariates': exogenous,
        'multiple_targets': True,           # each target travels as its own series
        # A16-03: the pipeline's own item builder fills the history of a future-only
        # covariate with NaN, and the transport refuses non-finite values. So a gap crosses
        # the boundary as a finite filler plus a boolean mask, and is restored to NaN here,
        # which is what the package expects.
        'missing_mask': exogenous,
        'quantiles': True,
    }
    declared.update(CAPABILITY_LIMITS.get(method, {}))
    return declared


def freq_alias(frequency_ns):
    """A pandas offset alias for the contract's nanosecond step."""
    if frequency_ns % (86_400 * NS) == 0:
        return f'{frequency_ns // (86_400 * NS)}D'
    if frequency_ns % (3_600 * NS) == 0:
        return f'{frequency_ns // (3_600 * NS)}h'
    if frequency_ns % (60 * NS) == 0:
        return f'{frequency_ns // (60 * NS)}min'
    if frequency_ns % NS == 0:
        return f'{frequency_ns // NS}s'
    raise AdapterError(f'input.frequency_ns: {frequency_ns} is not a whole number of seconds')


def _series_id(sample, target):
    return f'{sample}|{target}'


def _resolve(method, device, parameters, checkpoint=None):
    """The constructed family object, or a remedy naming what the image is missing.

    `checkpoint` is the id the request asked for and it wins over the family default.  It
    has to: a result that records one checkpoint while a different one produced the numbers
    is a lineage claim that is simply false, and the family defaults are chosen for
    convenience, not for what a case decided to keep.
    """
    try:
        from foundationforecast import models as catalogue
    except ImportError as error:                                  # pragma: no cover - in-image path
        raise AdapterError('foundationforecast is not installed in this image; '
                           'it is the eforecast-foundation image that serves these families') from error
    import inspect

    family = FAMILIES[method]
    cls = getattr(catalogue, family['cls'], None)
    if cls is None:
        available = sorted(n for n in dir(catalogue) if n[:1].isupper())
        raise AdapterError(f'{method}: this image\'s foundationforecast does not export '
                           f'{family["cls"]}; it exports {", ".join(available)}')
    kwargs = dict(parameters or {})
    kwargs.setdefault('alias', method)
    # a hosted family names a model, a weight-based one names a repository
    selector = 'model' if family.get('model') else 'repo_id'
    chosen = checkpoint or kwargs.get(selector) or family.get('model') or family.get('repo')
    if chosen:
        if selector in kwargs and kwargs[selector] != chosen:
            raise AdapterError(f'{method}: {selector} contradicts the recorded checkpoint')
        kwargs[selector] = chosen
    # pass device only where the family takes it; the rest follow torch's own placement
    accepted = inspect.signature(cls.__init__).parameters
    if 'device' in accepted or 'device_map' in accepted:
        placement = 'device' if 'device' in accepted else 'device_map'
        equivalent = (device, 'cuda:0', 0) if device == 'cuda' else (device,)
        if placement in kwargs and kwargs[placement] not in equivalent:
            raise AdapterError(f'{method}: {placement} contradicts the assigned device')
        kwargs[placement] = device
    unknown = [k for k in kwargs if k not in accepted and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values())]
    if unknown:
        raise AdapterError(f'{method}: {", ".join(sorted(unknown))} is not a parameter of '
                           f'{family["cls"]} in this image')
    return cls(**kwargs)


def _quantile_columns(frame, alias, quantiles):
    """Map each requested level onto the column the package actually produced."""
    pattern = re.compile(rf'^{re.escape(alias)}-q-(\d+(?:\.\d+)?)$')
    produced = {}
    for column in frame.columns:
        found = pattern.match(str(column))
        if found:
            produced[round(float(found.group(1)) / 100, 6)] = column
    missing = [q for q in quantiles if round(q, 6) not in produced]
    if missing:
        raise AdapterError(f'output.quantiles: {alias} returned '
                           f'{sorted(produced) or "no quantile columns"}, not {missing}')
    return [produced[round(q, 6)] for q in quantiles]


@pinned_forecast
def forecast(request, arrays):
    """Run one predict request and return (prediction, quantiles) in contract order.

    `arrays` holds the already-verified request arrays keyed as the contract names them.
    The returned prediction is [samples, horizon, targets]; quantiles add a trailing level
    axis, or are None when no level was asked for.
    """
    import pandas as pd
    from foundationforecast import FoundationForecast

    # Timestamps travel as UTC nanoseconds and are rebuilt timezone-NAIVE on purpose.  The
    # instants are identical -- the contract has no local time in it -- and the packages
    # require naive: Chronos-2's exogenous path does `df['ds'].to_numpy().view("int64")`,
    # which raises "Cannot change data-type for array of references" on a tz-aware column,
    # and gluonts indexes by naive period.  A16-03 found this the moment a covariate request
    # reached a real model.

    data = request['input']
    method, device = request['method'], request['device']
    samples, horizon = data['samples'], data['horizon']
    targets, future_names = data['targets'], data['future_covariates']
    quantiles = request['output']['quantiles']

    past_names = data['past_covariates']
    masked = data['missing'] == 'mask'
    history, history_index = arrays['history'], arrays['history_index']
    forecast_index = arrays['forecast_index']

    def restore(values, mask):
        """A masked payload back to the NaN the package reads as "unknown"."""
        out = values.astype('float64', copy=True)
        if mask is not None:
            out[~mask.astype(bool)] = np.nan
        return out

    history_mask = arrays.get('history_mask') if masked else None
    future_mask = arrays.get('future_mask') if masked else None

    # one series per (sample, target): the families are univariate, so a multi-target request
    # becomes several series rather than an unsupported multivariate call.  A covariate's
    # history rides alongside as a column, which is where the package reads it.
    rows = []
    for sample in range(samples):
        stamps = pd.to_datetime(history_index[sample], unit='ns')
        for position, target in enumerate(targets):
            block = pd.DataFrame({'unique_id': _series_id(sample, position), 'ds': stamps,
                                  'y': history[sample, :, position].astype('float64')})
            for column, name in enumerate(past_names):
                index = len(targets) + column
                block[name] = restore(history[sample, :, index],
                                      None if history_mask is None else history_mask[sample, :, index])
            rows.append(block)
    frame = pd.concat(rows, ignore_index=True)

    exogenous = None
    if future_names:
        future = arrays['future']
        blocks = []
        for sample in range(samples):
            stamps = pd.to_datetime(forecast_index[sample], unit='ns')
            for position in range(len(targets)):
                block = pd.DataFrame({'unique_id': _series_id(sample, position), 'ds': stamps})
                for column, name in enumerate(future_names):
                    block[name] = restore(future[sample, :, column],
                                          None if future_mask is None else future_mask[sample, :, column])
                blocks.append(block)
        exogenous = pd.concat(blocks, ignore_index=True)

    engine = FoundationForecast(models=[_resolve(
        method, device, request['component']['parameters'],
        checkpoint=(request['component']['checkpoint'] or {}).get('id'))])
    arguments = {'df': frame, 'h': horizon, 'freq': freq_alias(data['frequency_ns'])}
    if quantiles:
        arguments['quantiles'] = list(quantiles)
    if exogenous is not None:
        arguments['X_df'] = exogenous
    produced = engine.forecast(**arguments)

    if method not in produced.columns:
        offered = [c for c in produced.columns if c not in ('unique_id', 'ds')]
        raise AdapterError(f'{method}: the forecast frame carries {offered}, not a {method} column')
    levels = _quantile_columns(produced, method, quantiles) if quantiles else []

    # reassemble on (unique_id, ds) rather than on row order: the package is free to return
    # the panel in any order, and silently trusting it would mix samples together
    produced = produced.set_index(['unique_id', 'ds'])
    prediction = np.empty((samples, horizon, len(targets)), dtype='float64')
    spread = np.empty((samples, horizon, len(targets), len(levels)), dtype='float64') if levels else None
    for sample in range(samples):
        stamps = pd.to_datetime(forecast_index[sample], unit='ns')
        for position in range(len(targets)):
            key = _series_id(sample, position)
            try:
                block = produced.loc[key].reindex(stamps)
            except KeyError as error:
                raise AdapterError(f'{method}: no forecast returned for series {key}') from error
            if block[method].isna().any():
                raise AdapterError(f'{method}: the forecast does not cover every requested timestamp '
                                   f'for series {key}; the index the request asked for is authoritative')
            prediction[sample, :, position] = block[method].to_numpy(dtype='float64')
            for index, column in enumerate(levels):
                spread[sample, :, position, index] = block[column].to_numpy(dtype='float64')
    return prediction, spread
