# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The time step a case runs at, and the pandas offset alias that means it.

Every call site used to write ``ts_resolution.lower()``, and for a real reason: pandas >= 2.2
raises ``ValueError`` on ``'H'``. pandas 3 then canonicalised the other half of the alias table
the other way -- sub-daily aliases lowercase (``'h'``, ``'min'``), daily and above uppercase
(``'D'``, ``'W'``, ``'ME'``). So the very transformation that rescues ``'H'`` turns the equally
supported ``'D'`` into ``'d'``, which pandas 3 deprecates and pandas 4 removes outright. One of
the resolutions this pipeline accepts was quietly on that path, in seven places.

Resolve through the supported set instead of transforming the string. ``'me'`` is the warning
worth remembering: lowercasing month-end does not deprecate it, it makes it invalid immediately.

This module imports nothing on purpose, so ``eforecast.config`` -- which keeps pandas out of its
import graph -- and the dataset code can both use it for free.
"""

#: What a configuration may say, mapped to the one spelling the rest of the code uses. Shared
#: with ``eforecast.config.schema``'s validator so the two cannot drift apart.
SPELLINGS = {'h': 'H', 'd': 'D', '1MIN': '1min', '5MIN': '5min', '15MIN': '15min', 'min': '1min', 'T': '1min',
             '5T': '5min', '15T': '15min'}

#: The pandas offset alias for each supported resolution. A test checks these keys against the
#: schema's own ``RESOLUTION_STEP``, so a resolution cannot be added without an alias for it.
PANDAS_FREQ = {'1min': '1min', '5min': '5min', '15min': '15min', 'H': 'h', 'D': 'D'}


def canonical(ts_resolution):
    """The pipeline's spelling of ``ts_resolution``, accepting the documented variants."""
    value = str(ts_resolution) if ts_resolution else 'H'
    return SPELLINGS.get(value, value)


def pandas_freq(ts_resolution):
    """The pandas offset alias for a case's ``ts_resolution``.

    Raises ``ValueError`` rather than passing an unknown string through to pandas. Letting the
    raw value past is how ``'d'`` reached ``date_range`` in the first place, where it worked, and
    warned, and would have stopped working at a pandas upgrade.
    """
    value = canonical(ts_resolution)
    try:
        return PANDAS_FREQ[value]
    except KeyError:
        raise ValueError(f'ts_resolution {ts_resolution!r} is not one of '
                         f'{", ".join(PANDAS_FREQ)}') from None
