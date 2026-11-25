# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""What differs between two case documents, as the settings a person would name.

A fingerprint says *that* a configuration changed. This says *what*: one entry per setting,
with the value it had and the value it has, at the path the editor shows it under::

    data.variables[Cloud].lags        [0, 1, 2, 3, 4] -> [0, 1, 2, 3, 4, 5]
    methods.method_params.TRANSFORMER.model_name.choices   ['iTransformer', 'PatchTST'] -> ['iTransformer']

Nothing here decides anything. What a change costs is decided by the receipts and the trial
records (``stage_resume``, ``revisions``); these entries are what makes that decision readable.
"""
from __future__ import annotations

from typing import Any

#: The key that names an entry of a list of mappings, in the order it is looked for. The
#: variables of a case are named; its weather providers are told apart by their model.
NAMES = ('name', 'model')

ABSENT = object()


def _named(items):
    """``{name: item}`` when a list holds mappings that one of ``NAMES`` tells apart, else None."""
    if not items or not all(isinstance(item, dict) for item in items):
        return None
    for key in NAMES:
        names = [item.get(key) for item in items]
        if all(isinstance(name, (str, int)) and not isinstance(name, bool) for name in names) \
                and len(set(names)) == len(names):
            return {str(name): item for name, item in zip(names, items)}
    return None


def _join(path, key):
    return f'{path}.{key}' if path else str(key)


def differences(before: Any, after: Any, path: str = '') -> list[dict[str, Any]]:
    """Every setting that is not the same in both, in the order ``after`` holds them.

    Mappings are compared key by key. A list of named mappings -- the variables -- is compared
    entry by entry under ``[name]``, so that adding a lag to one variable is one line and not
    the whole list twice; an entry only one side has is reported whole, as added or removed.
    Any other list is one value: its order means something, and its parts do not have names.
    """
    if before is ABSENT:
        return [{'path': path, 'change': 'added', 'after': after}]
    if after is ABSENT:
        return [{'path': path, 'change': 'removed', 'before': before}]
    if isinstance(before, dict) and isinstance(after, dict):
        out = []
        for key in [*after, *(key for key in before if key not in after)]:
            out += differences(before.get(key, ABSENT), after.get(key, ABSENT), _join(path, key))
        return out
    if isinstance(before, list) and isinstance(after, list):
        old, new = _named(before), _named(after)
        if old is not None and new is not None:
            out = []
            for name in [*new, *(name for name in old if name not in new)]:
                out += differences(old.get(name, ABSENT), new.get(name, ABSENT), f'{path}[{name}]')
            if not out and list(old) != list(new):
                out.append({'path': path, 'change': 'reordered', 'before': list(old), 'after': list(new)})
            return out
    if before == after and type(before) is type(after):
        return []
    # ``1`` and ``1.0``, ``True`` and ``1``: equal to Python and not the same setting in a file.
    if before == after and not isinstance(before, bool) and not isinstance(after, bool) \
            and isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return []
    return [{'path': path, 'change': 'changed', 'before': before, 'after': after}]


def as_read(document: Any) -> dict[str, Any] | None:
    """A case document as the schema reads it, every default written out; None when it is not a case.

    Two documents that are read alike are one configuration, however they are spelled. An
    editor that writes down the defaults of the sections it showed has changed no setting; a
    key that is left out has the value its default gives it, and a change of that key starts
    from that value and not from nothing.
    """
    if not isinstance(document, dict):
        return None
    try:
        import copy
        from .schema import ExperimentConfig
        return ExperimentConfig.model_validate(copy.deepcopy(document)).model_dump(mode='json')
    except Exception:                                           # noqa: BLE001 -- a draft with an error is not a case yet
        return None


def settings_changed(before: Any, after: Any) -> list[dict[str, Any]]:
    """What differs between two case documents as they are read, not as they are spelled.

    ``differences`` of the two with their defaults written out: a default that was merely
    written down is no change, and a key that was left out and is now given another value is
    a change *from its default*. When either is not a case -- a draft with an error in it, or
    nothing at all -- the documents are compared as they are written.
    """
    old, new = as_read(before), as_read(after)
    if old is None or new is None:
        return differences(before or {}, after or {})
    return differences(old, new)


def brief(value: Any, limit: int = 80) -> str:
    """A value on one line, cut where it stops being readable."""
    import json
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[:limit - 1] + '…'


def describe(change: dict[str, Any]) -> str:
    """One change as a line: the path, and what became of its value."""
    if change['change'] == 'added':
        return f"{change['path']}: added ({brief(change['after'])})"
    if change['change'] == 'removed':
        return f"{change['path']}: removed (was {brief(change['before'])})"
    return f"{change['path']}: {brief(change['before'])} -> {brief(change['after'])}"
