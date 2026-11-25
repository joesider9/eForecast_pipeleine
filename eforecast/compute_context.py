# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Neutral host injection and worker scope; safe in the CPU inference image."""
from contextlib import contextmanager
from contextvars import ContextVar

active_worker = ContextVar('eforecast_active_worker', default=None)
fuzzy_backend = ContextVar('eforecast_fuzzy_backend', default=None)
dataset_backend = ContextVar('eforecast_dataset_backend', default=None)
inference_sites = ContextVar('eforecast_inference_sites', default=None)
inference_input_records = ContextVar('eforecast_inference_input_records', default=None)
_resident_backends = {}
_scoped_backends = ContextVar('eforecast_scoped_backends', default={})


def install_resident_backend(kind, callback):
    """Host entry points install adapters; the CPU inference image imports no host code."""
    _resident_backends[kind] = callback


def resident_backend(kind):
    return _scoped_backends.get().get(kind, _resident_backends.get(kind))


@contextmanager
def using_resident_backend(kind, callback):
    token = _scoped_backends.set({**_scoped_backends.get(), kind: callback})
    try:
        yield
    finally:
        _scoped_backends.reset(token)


@contextmanager
def using_dataset_backend(backend):
    token = dataset_backend.set(backend)
    try:
        yield
    finally:
        dataset_backend.reset(token)


@contextmanager
def using_fuzzy_backend(backend):
    token = fuzzy_backend.set(backend)
    try:
        yield
    finally:
        fuzzy_backend.reset(token)
