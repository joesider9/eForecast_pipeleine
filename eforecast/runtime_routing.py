# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Where a method's model work runs, and the timestamps the contract needs -- A16-03.

Two jobs, both small and both easy to get quietly wrong.

**The route.** A family the host cannot import at all (TimeGPT, Timestral, T0, Lag-Llama)
can only run in its pinned image. A family the host *can* import (Chronos2, TimesFM, TiRex)
keeps running natively unless the case asks otherwise through `methods.runtime_ids`, because
A16-02's capability is opt-in and silently moving existing work into a container would change
what produced every retained result.

**The window.** The contract carries UTC nanosecond indices; `build_items` carries no
timestamps at all. They are reconstructed here from the sample dates, and the convention is
not a guess: `derived_facts` builds `target_columns = [f"Step_{i}" for i in range(steps)]`
with `target_lags = list(range(steps))`, so `Step_0` is the sample's **own** timestamp. The
forecast therefore begins at the sample date and the context is the steps immediately before
it. Getting that wrong by one step would shift hour-of-day, which for a PV case is the
difference between noon and midnight.
"""
from __future__ import annotations

import numpy as np

#: The series step per declared resolution, in seconds.  ``schema.lag_step`` expresses a step
#: in the horizon's own unit -- it maps "D" to 1 rather than to 24 hours -- which is right for
#: counting target columns and wrong for building timestamps, so this table is separate.
RESOLUTION_SECONDS = {'1min': 60, '5min': 300, '15min': 900, 'H': 3600, 'D': 86400}

NATIVE, IMAGE = 'native', 'image'

NS = 1_000_000_000


class RoutingError(RuntimeError):
    """Nothing available can serve this method, with the reason a colleague needs."""


def step_nanoseconds(dates=None, static_data=None):
    """Use the declared sampling grid; infer from dates only for an undeclared legacy grid."""
    resolution = (static_data or {}).get('ts_resolution')
    if resolution is not None:
        # A cluster/validation batch is a sparse selection, not a sampling grid.
        # Its most frequent gap may be hours or days larger than the original series step.
        try:
            import re
            from pandas.tseries.frequencies import to_offset
            normalized = resolution
            if isinstance(resolution, str):
                legacy = re.fullmatch(r'(\d+(?:\.\d+)?)?(H|T|S|L|U|N)', resolution.strip())
                if legacy:
                    aliases = {'H': 'h', 'T': 'min', 'S': 's', 'L': 'ms', 'U': 'us', 'N': 'ns'}
                    normalized = (legacy[1] or '1') + aliases[legacy[2]]
            step = to_offset(normalized).nanos
            if step <= 0:
                raise ValueError('non-positive frequency')
            return int(step)
        except (TypeError, ValueError) as error:
            raise RoutingError(f'the series step cannot be established from {resolution!r}') from error
    if dates is not None and len(dates) >= 2:
        values = np.asarray(dates, dtype='datetime64[ns]').astype('int64')
        gaps = np.diff(np.unique(values))
        gaps = gaps[gaps > 0]
        if len(gaps):
            unique, counts = np.unique(gaps, return_counts=True)
            return int(unique[counts.argmax()])
    resolution = (static_data or {}).get('ts_resolution')
    seconds = RESOLUTION_SECONDS.get(resolution)
    if seconds is None:
        raise RoutingError(
            f'the series step cannot be established: the dates do not show one and '
            f'ts_resolution is {resolution!r}, not one of {", ".join(sorted(RESOLUTION_SECONDS))}')
    return seconds * NS


def window_index(dates, context, horizon, step_ns, timezone='UTC'):
    """(history_index, forecast_index) as UTC nanoseconds, one row per sample.

    The forecast starts at the sample's own date (``Step_0``) and the history is the
    ``context`` steps immediately before it.
    """
    import pandas as pd
    try:
        index = pd.DatetimeIndex(dates)
        if index.tz is None:
            index = index.tz_localize(timezone, ambiguous='raise', nonexistent='raise')
        origin = index.tz_convert('UTC').as_unit('ns').asi8
    except Exception as error:
        raise RoutingError(f'cannot establish UTC forecast dates: {error}') from error
    if index.hasnans or step_ns <= 0:
        raise RoutingError('forecast dates and frequency must be valid')
    if origin.ndim != 1:
        raise RoutingError('dates must be one timestamp per sample')
    step = int(step_ns)
    forward = np.arange(int(horizon), dtype='int64') * step
    backward = (np.arange(int(context), dtype='int64') - int(context)) * step
    return origin[:, None] + backward[None, :], origin[:, None] + forward[None, :]


def _native_available(method):
    from eforecast.config import readiness
    module, _ = readiness.FAMILY_DISTRIBUTIONS.get(method, (None, None))
    return bool(module) and readiness._available(module)


def route(method, *, static_data=None, params=None, require_coherent=True):
    """Where this method's model work should run, and why.

    `require_coherent` checks the image against the selected release when one is selected.
    A stale image is refused rather than used, because the symptom of using one is a wrong
    number rather than an error (A16-11).
    """
    from eforecast import image_runtime, runtime_launcher
    from eforecast.runtime_adapters import adapter_name

    served = runtime_launcher.CASE_METHODS.get(method)
    from eforecast.runtime_registry import validate_binding, resolve, require_available, runtime_for
    static = static_data or {}
    selected = (static.get('runtime_ids') or {}).get(method, (static.get('runtime_ids') or {}).get('default'))
    if selected in (None, '') and params:
        selected = params.get('runtime_id')

    binding = static.get('_runtime_binding')
    if binding is not None or selected not in (None, ''):
        try:
            binding = validate_binding(binding) if binding is not None else resolve(
                {'runtime_ids': {method: selected}})
            key = runtime_for(binding, method)['id']
            definition = binding['definitions'][key]
            if definition['kind'] == 'container':
                require_available(binding)
                adapter = adapter_name(served)
                if require_coherent and static.get('_runtime_binding') is None:
                    unmet = _incoherent(adapter)
                    if unmet:
                        raise RoutingError(f'{method}: selected runtime is {unmet}')
                return {'where': IMAGE, 'method': method, 'served': served, 'runtime': key,
                        'reference': definition['locator'], 'image': image_runtime.image_id(definition['locator']),
                        'runtime_revision': runtime_for(binding, method)['revision'],
                        'reason': 'the accepted runtime binding selects this image'}
            selected = NATIVE
        except ValueError as error:
            raise RoutingError(str(error)) from error

    if served is None:
        if selected not in (None, '', NATIVE):
            raise RoutingError(f'{method}: no model image serves it, so runtime {selected!r} '
                               f'cannot be selected for it')
        return {'where': NATIVE, 'method': method, 'served': method, 'runtime': NATIVE,
                'reference': None, 'image': None,
                'reason': 'not a family any model image serves'}

    native = _native_available(method)
    if selected in (NATIVE,):
        if not native:
            raise RoutingError(f'{method}: native execution was selected but the host cannot '
                               f'import it; it is served by {runtime_launcher.image_for(served)}')
        return {'where': NATIVE, 'method': method, 'served': served, 'runtime': NATIVE,
                'reference': None, 'image': None,
                'reason': 'the case selected native execution'}

    wants_image = selected not in (None, '')
    if wants_image and selected not in (NATIVE, served, adapter_name(served)):
        raise RoutingError(f'{method}: runtime {selected!r} is not one that serves it; '
                           f'{adapter_name(served)!r} and {NATIVE!r} are')
    if native and not wants_image:
        # Opt-in: existing work keeps running where it has always run.
        return {'where': NATIVE, 'method': method, 'served': served, 'runtime': NATIVE,
                'reference': None, 'image': None,
                'reason': 'the host can import it and the case selected no runtime'}

    reference = runtime_launcher.image_for(served)
    # `served` is the adapter's name for the method (Chronos2 -> Chronos); the runtime that
    # holds it is the adapter itself (foundation), which is what a release describes
    adapter = adapter_name(served)
    image = image_runtime.image_id(reference)
    if image is None:
        raise RoutingError(
            f'{method}: it is served by {reference}, which is not present on this host. '
            f'{"The host cannot import it either" if not native else "Select native to use the host package"}.')
    if require_coherent:
        unmet = _incoherent(adapter)
        if unmet:
            raise RoutingError(f'{method}: {reference} is {unmet}. A stale image produces a '
                               f'wrong number rather than an error, so it is not used.')
    # `served` is the name the adapter dispatches on (the case says Chronos2, the package
    # says Chronos); the request carries that, and the case's own name stays in the identity
    return {'where': IMAGE, 'method': method, 'served': served, 'runtime': adapter,
            'reference': image, 'image': image,
            'reason': 'the case selected this runtime' if wants_image else
                      'the host cannot import it, so its image is the only route'}


def _incoherent(adapter):
    """Why the release says this runtime is not coherent, or None."""
    from eforecast import image_release

    release = image_release.selected()
    if release is None:
        return None                     # nothing selected: A16-11's doctor reports that
    try:
        report = image_release.check(release, runtimes=[adapter])
    except Exception as error:
        return f'unverifiable against release {release} ({type(error).__name__})'
    entry = (report.get('runtimes') or {}).get(adapter) or {}
    if entry.get('status') == image_release.COHERENT:
        return None
    reasons = ', '.join(sorted({item['reason'] for item in entry.get('reasons', [])}))
    return f'{entry.get("status", "unknown")} against release {release} ({reasons})'
