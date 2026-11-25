# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Enforce the checkpoint named by a one-request model worker.

Third-party families often omit ``revision`` when calling the Hub. Install the
pin before importing those families, including the Hub's implementation modules
from which libraries import helpers directly. No model bytes are fetched here.
The process serves one request; the lock also serializes in-process callers.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import re
import threading
from unittest.mock import patch

_LOCK = threading.RLock()
_PIN = ContextVar('runtime_checkpoint', default=None)


@contextmanager
def pin(checkpoint):
    if checkpoint is None:
        yield
        return
    repository, revision = checkpoint['id'], checkpoint['revision']
    if not re.fullmatch('[0-9a-f]{40}', revision) or revision == '0' * 40:
        raise ValueError('isolated Hub checkpoints need a real resolved commit, not a placeholder')
    import huggingface_hub as hub
    from huggingface_hub import file_download, _snapshot_download
    from contextlib import ExitStack

    def pinned(original):
        signature = inspect.signature(original)

        @wraps(original)
        def call(*args, **kwargs):
            current = _PIN.get()
            if current is None:
                return original(*args, **kwargs)
            repository, revision = current['id'], current['revision']
            # Hub decorators accept and normalize deprecated keyword arguments that
            # their public signature no longer lists (e.g. transformers' proxies).
            # Inspect positional arguments only; let the original API handle kwargs.
            bound = signature.bind_partial(*args)
            repo = bound.arguments.get('repo_id', kwargs.get('repo_id'))
            if repo != repository:
                raise ValueError(f'checkpoint dependency {repo!r} is not the recorded repository {repository!r}')
            requested = bound.arguments.get('revision', kwargs.get('revision'))
            if requested not in (None, 'main', revision):
                raise ValueError('model loader tried to override the recorded checkpoint revision')
            if 'revision' in bound.arguments:
                bound.arguments['revision'] = revision
            else:
                kwargs = dict(kwargs, revision=revision)
            return original(*bound.args, **kwargs)
        return call

    with _LOCK, ExitStack() as stack:
        stack.callback(_PIN.reset, _PIN.set(checkpoint))
        download = pinned(hub.hf_hub_download)
        snapshot = pinned(hub.snapshot_download)
        for module, name, replacement in (
                (hub, 'hf_hub_download', download), (file_download, 'hf_hub_download', download),
                (_snapshot_download, 'hf_hub_download', download),
                (hub, 'snapshot_download', snapshot), (_snapshot_download, 'snapshot_download', snapshot)):
            stack.enter_context(patch.object(module, name, replacement))
        stack.enter_context(patch.object(hub.HfApi, 'model_info', pinned(hub.HfApi.model_info)))
        yield


def pinned_forecast(function):
    @wraps(function)
    def forecast(request, arrays):
        with pin(request['component']['checkpoint']):
            return function(request, arrays)
    return forecast
