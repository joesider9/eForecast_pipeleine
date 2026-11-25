"""Adapters that serve one model family inside its own image, through A16-01's contract.

A16-02.  Each adapter exposes the same three things -- families(), capability(method) and
forecast(request, arrays) -- so the host does not need to know which image answered, and
an image that cannot serve a method says so instead of failing in the middle of a run.

Nothing here imports torch, pandas or a model package at module scope: the host asks what
an image supports far more often than it asks it to forecast, and a colleague opening the
UI must not pull thirteen checkpoints.
"""
from __future__ import annotations

from importlib import import_module
import importlib.metadata

# adapter module -> the methods it claims; kept here so dispatch needs no heavy import
CATALOGUE = {
    'foundation': ('Chronos', 'TimesFM', 'TiRex', 'TiRex2', 'Timestral', 'T0', 'TimeGPT',
                   'Toto', 'Moirai', 'Sundial', 'TabPFN', 'Tafsut', 'FlowState', 'PatchTSTFM'),
    'lag_llama': ('Lag-Llama',),
}


# the package each adapter needs; checked by metadata so asking what an image serves never
# imports torch, reaches Hugging Face or loads a checkpoint
DISTRIBUTIONS = {'foundation': 'foundationforecast', 'lag_llama': 'lag-llama'}


class UnknownMethod(ValueError):
    """No adapter in this image claims the requested method."""


def methods():
    """Every method any adapter in this build claims, whether or not it is installed."""
    return sorted(name for names in CATALOGUE.values() for name in names)


def installed(name):
    """Whether this image carries the package the named adapter needs."""
    try:
        importlib.metadata.version(DISTRIBUTIONS[name])
    except (importlib.metadata.PackageNotFoundError, KeyError):
        return False
    return True


def served():
    """The methods this image can actually serve.

    Distinct from methods(): every build carries every adapter module, because they are a few
    kilobytes of pure Python, but only one of them has its model package installed. An image
    that claimed the other one would send the host a request it cannot answer.
    """
    return sorted(name for adapter, names in CATALOGUE.items() if installed(adapter)
                  for name in names)


def adapter_name(method):
    for name, names in CATALOGUE.items():
        if method in names:
            return name
    raise UnknownMethod(f'{method}: no adapter serves it; this build knows {", ".join(methods())}')


def declared(method):
    """What an adapter says it supports, read on the host without the model package.

    `adapter()` refuses a method the *current* image cannot serve, which is right for
    dispatch and wrong for planning: the host carries every adapter module but no model
    package, and it still has to know what a request may contain before it builds one. The
    capability declaration needs nothing but numpy, so it can be read here.
    """
    name = adapter_name(method)
    try:
        return import_module(f'{__name__}.{name}').capability(method)
    except ImportError as error:
        raise UnknownMethod(f'{method}: the {name} adapter is not in this build '
                            f'({error})') from error


def adapter(method):
    """The adapter module for a method, imported only when it is actually needed.

    The import succeeding proves nothing about this image: every adapter module needs only
    numpy at module scope, so the Lag-Llama adapter imports cleanly inside the foundation
    image, which cannot serve it. The model package is what decides, so it is checked here
    -- otherwise an image would declare a capability it would fail to honour, and the host
    would discover that only after staging a request.
    """
    name = adapter_name(method)
    if not installed(name):
        raise UnknownMethod(
            f'{method}: this image does not carry {DISTRIBUTIONS[name]}, so it cannot serve '
            f'it. This image serves {", ".join(served()) or "no method"}.')
    try:
        return import_module(f'{__name__}.{name}')
    except ImportError as error:
        raise UnknownMethod(f'{method}: the {name} adapter is not present in this image '
                            f'({error})') from error
