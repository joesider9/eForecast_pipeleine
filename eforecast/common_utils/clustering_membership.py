# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Rule membership shared by subdatasets, fuzzy diagnostics and the linear proxy.

A finite activation equal to the threshold is active. A row with no active rule is
assigned to its strongest finite rule; a row with no finite rule is invalid.
This module needs neither a training framework nor a clustering library.
"""
import numbers

import numpy as np


def validate_rule_threshold(threshold):
    """Validate the supported finite, positive rule-activation cutoff explicitly.

    ``None`` is the placeholder a non-fuzzy project carries, so the type is checked here:
    this function exists for its message, and ``np.isfinite(None)`` raises an unreadable
    numpy ``TypeError`` instead.  ``numbers.Real`` accepts NumPy scalars; ``bool`` is a
    subclass of ``int`` and is not a threshold.
    """
    if not isinstance(threshold, numbers.Real) or isinstance(threshold, bool):
        raise ValueError(f'the rule activation threshold must be a number between 0 and 1, '
                         f'not {threshold!r}; a non-fuzzy project has no threshold')
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('the rule activation threshold must be finite and between 0 and 1')


def rule_membership(activations, threshold):
    validate_rule_threshold(threshold)
    values = np.asarray(activations, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('rule activations must have shape (samples, rules), with at least one rule')
    return np.isfinite(values) & (values >= threshold)


def ensure_rule_coverage(activations, threshold):
    """Return ``(values, assigned_count)`` without modifying the supplied activations."""
    values = np.asarray(activations, dtype=float)
    active = rule_membership(values, threshold)
    finite = np.isfinite(values)
    invalid = np.flatnonzero(~finite.any(axis=1))
    if len(invalid):
        raise ValueError(f'no finite rule activation for {len(invalid)} sample(s); '
                         f'first row positions: {invalid[:5].tolist()}')
    rows = np.flatnonzero(~active.any(axis=1))
    if len(rows):
        values = values.copy()
        strongest = np.where(finite[rows], values[rows], -np.inf).argmax(axis=1)
        values[rows, strongest] = threshold
    return values, int(len(rows))
