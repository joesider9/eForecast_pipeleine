# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Conformal prediction intervals for the predictors of the ensemble.

Two recipes, both fitted on the *selection* slice of the evaluation period and reported on the
untouched *holdout* slice, per horizon step and per coverage level:

* ``split``: for a point forecast, the residual score ``|y - yhat|``; the band is
  ``yhat -/+ q`` with ``q`` the finite-sample-corrected empirical quantile of the scores at the
  coverage level.  Works for every predictor (trials, methods, combiners, cluster averages, models).
* ``cqr`` (conformalised quantile regression, Romano et al. 2019): for a trial with a quantile head,
  the score ``max(q_lo - y, y - q_hi)`` of the nominal band and the band ``[q_lo - Q, q_hi + Q]``;
  the learned quantiles carry the heteroscedasticity, the correction restores the coverage.

Everything is in power units.  ``Calibration`` holds ``{predictor: {coverage: {step: q}}}`` (plus
the pair of nominal quantiles used by ``cqr``) and is written as JSON next to the results, so the
online forecast can attach bands to whatever predictor it returns.
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CALIBRATION_FILE = 'calibration.json'
# Below this many calibration points a band is not refused by the arithmetic but is not worth
# trusting: the quantile of a handful of residuals moves with every one of them.
MIN_CALIBRATION_ROWS = int(os.environ.get('EFORECAST_MIN_CALIBRATION_ROWS', 20))


def min_calibration_samples(coverage: float) -> int:
    """The fewest scores for which the corrected quantile is a bound rather than the largest score.

    The split-conformal quantile is the k-th smallest score with ``k = ceil((n + 1) * coverage)``.
    When ``k > n`` there is no such score and the recipe has nothing to offer: ``k <= n`` requires
    ``n >= coverage / (1 - coverage)`` -- 9 points for 80 %, 19 for 95 %, 99 for 99 %.
    """
    coverage = float(coverage)
    if not 0 < coverage < 1:
        return 0
    # from the predicate itself rather than from ceil(c / (1 - c)): in floating point that is
    # 4.000000000000001 for a coverage of 0.8, and one point too many for every round coverage
    n = max(1, int(math.floor(coverage / (1.0 - coverage))))
    while math.ceil((n + 1) * coverage) > n:
        n += 1
    return n


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """The finite-sample-corrected empirical quantile of the scores, or NaN when there are too few.

    Capping ``k`` at ``n`` -- returning the largest score as if it were the bound -- is what this
    used to do, and it is the one answer that cannot be right: the widest band these points can
    make is not a ``coverage`` band, it is simply the worst residual seen. A band that cannot be
    calibrated is better left unfitted and reported as such.
    """
    scores = np.asarray(scores, dtype=float)
    scores = np.sort(scores[np.isfinite(scores)])
    n = scores.size
    k = math.ceil((n + 1) * float(coverage))
    if n == 0 or k > n:
        return float('nan')
    return float(scores[max(1, k) - 1])


def _aligned(pred: pd.DataFrame, y: pd.DataFrame, dates: pd.DatetimeIndex | None) -> tuple[np.ndarray, np.ndarray]:
    common = pred.index.intersection(y.index)
    if dates is not None:
        common = common.intersection(dates)
    p = pred.loc[common].values.astype(float)
    t = y.loc[common].values.astype(float)[:, :p.shape[1]]
    return p, t


def fit_split(pred: pd.DataFrame, y: pd.DataFrame, dates, coverages) -> dict[float, list[float]]:
    """{coverage: [q per step]} of the symmetric residual band around ``pred``."""
    p, t = _aligned(pred, y, dates)
    scores = np.abs(t - p)
    return _quantiles(scores, coverages)


def fit_cqr(lower: pd.DataFrame, upper: pd.DataFrame, y: pd.DataFrame, dates, coverages) -> dict[float, list[float]]:
    """{coverage: [Q per step]} of the conformalised quantile band ``[lower - Q, upper + Q]``."""
    lo, t = _aligned(lower, y, dates)
    hi, _ = _aligned(upper, y, dates)
    if lo.shape != hi.shape:
        return {}
    scores = np.maximum(lo - t, t - hi)
    return _quantiles(scores, coverages)


def _quantiles(scores: np.ndarray, coverages) -> dict[float, list[float]]:
    """``{coverage: [q per step]}``, leaving out a coverage this many points cannot support.

    Two floors, and they answer different questions. ``min_calibration_samples`` is arithmetic:
    below it the corrected quantile does not exist and the old code returned the largest residual
    as if it did. ``MIN_CALIBRATION_ROWS`` is judgement: a 4-point 80 % band is a valid bound and
    a useless one. Both are per coverage rather than one threshold for all of them, because 300
    points are plenty for an 80 % band and not enough for a 99 % one, where the old blanket
    ``n < 20`` gave a predictor either every band or none.
    """
    n = scores.shape[0]
    out = {}
    for coverage in coverages:
        needed = max(min_calibration_samples(coverage), MIN_CALIBRATION_ROWS)
        if n < needed:
            log.info('no %.0f%% band: %d calibration points, %d needed (%d for the corrected quantile to be a '
                     'bound at all, %d to be worth trusting)', 100 * float(coverage), n, needed,
                     min_calibration_samples(coverage), MIN_CALIBRATION_ROWS)
            continue
        values = [conformal_quantile(scores[:, k], coverage) for k in range(scores.shape[1])]
        if any(not math.isfinite(v) for v in values):
            continue
        out[float(coverage)] = values
    return out


def band_split(pred: pd.DataFrame, q: list[float], rated=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    qv = np.asarray(q, dtype=float)[:pred.shape[1]]
    lower = (pred - qv).clip(lower=0)
    upper = pred + qv
    if rated is not None:
        upper = upper.clip(upper=float(rated))
    return lower, upper


def band_cqr(lower: pd.DataFrame, upper: pd.DataFrame, q: list[float], rated=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    qv = np.asarray(q, dtype=float)[:lower.shape[1]]
    lo = (lower - qv).clip(lower=0)
    hi = upper + qv
    if rated is not None:
        hi = hi.clip(upper=float(rated))
    return lo, hi


def interval_metrics(lower: pd.DataFrame, upper: pd.DataFrame, y: pd.DataFrame, dates, coverage: float, rated=None,
                     columns=None) -> dict[str, dict[str, float]]:
    """Coverage (fraction inside), mean width and the interval (Winkler) score per step and overall;
    width and score in percent of the rated power (or of the target)."""
    lo, t = _aligned(lower, y, dates)
    hi, _ = _aligned(upper, y, dates)
    if lo.shape[0] == 0 or lo.shape != hi.shape:
        return {}
    alpha = 1.0 - float(coverage)
    norm = float(rated) if rated is not None else np.where(np.abs(t) > 1e-6, np.abs(t), np.nan)
    inside = (t >= lo) & (t <= hi)
    width = (hi - lo) / norm
    score = width + (2 / alpha) * np.maximum(lo - t, 0) / norm + (2 / alpha) * np.maximum(t - hi, 0) / norm
    names = list(columns or lower.columns)
    out = {}
    for k, name in enumerate(names[:lo.shape[1]]):
        out[name] = {'coverage': float(np.nanmean(inside[:, k])), 'width': 100 * float(np.nanmean(width[:, k])),
                     'interval_score': 100 * float(np.nanmean(score[:, k])), 'n': int(inside.shape[0])}
    out['all'] = {'coverage': float(np.nanmean(inside)), 'width': 100 * float(np.nanmean(width)),
                  'interval_score': 100 * float(np.nanmean(score)), 'n': int(inside.size)}
    return out


def coverage_breakdown(lower: pd.DataFrame, upper: pd.DataFrame, y: pd.DataFrame, dates, coverage: float,
                       rated=None, columns=None) -> list[dict[str, Any]]:
    """Measured coverage by horizon step, by month and by regime, on the dates given.

    One number for the whole slice hides the two ways a conformal band fails a time series. A band
    is a single correction for every hour of every season, so it over-covers the quiet periods and
    under-covers the volatile ones; and exchangeability, which the split-conformal guarantee needs,
    is exactly what a seasonal series does not have. Reporting the breakdown does not repair that
    -- nothing in a single global correction can -- but it says where the band is not keeping its
    promise instead of averaging it away.
    """
    lo, t = _aligned(lower, y, dates)
    hi, _ = _aligned(upper, y, dates)
    if lo.shape[0] == 0 or lo.shape != hi.shape:
        return []
    common = pred_index(lower, y, dates)
    inside = (t >= lo) & (t <= hi)
    norm = float(rated) if rated is not None else np.where(np.abs(t) > 1e-6, np.abs(t), np.nan)
    width = (hi - lo) / norm
    names = list(columns or lower.columns)[:lo.shape[1]]
    rows = []

    def row(by, bucket, mask_rows, mask_cols=None):
        block = inside[mask_rows][:, mask_cols] if mask_cols is not None else inside[mask_rows]
        wide = width[mask_rows][:, mask_cols] if mask_cols is not None else width[mask_rows]
        if block.size == 0:
            return
        rows.append({'by': by, 'bucket': bucket, 'coverage_target': float(coverage),
                     'coverage': float(np.nanmean(block)), 'width': 100 * float(np.nanmean(wide)),
                     'n': int(block.size)})

    everything = np.ones(inside.shape[0], dtype=bool)
    for k, name in enumerate(names):
        row('step', str(name), everything, [k])
    for month in sorted(set(common.month)):
        row('month', f'{month:02d}', np.asarray(common.month == month))
    level = np.nanmean(t, axis=1)
    finite = level[np.isfinite(level)]
    if finite.size >= 3:
        low, high = np.quantile(finite, [1 / 3, 2 / 3])
        for label, mask in (('low', level <= low), ('medium', (level > low) & (level <= high)),
                            ('high', level > high)):
            row('regime', label, mask)
    return rows


def pred_index(pred: pd.DataFrame, y: pd.DataFrame, dates) -> pd.DatetimeIndex:
    """The dates ``_aligned`` kept, in the same order."""
    common = pred.index.intersection(y.index)
    if dates is not None:
        common = common.intersection(dates)
    return pd.DatetimeIndex(common)


class Calibration:
    """The fitted corrections: ``{predictor: {'method': 'split' | 'cqr', 'quantiles': [lo, hi] | None,
    'q': {coverage: [per step]}}}``; predictors are keyed by their path in the ensemble result
    (``models/kmeans``, ``cluster_average/lasso``, ``global/MLP``, ``clusters/RBF_rule_0/mean``,
    ``trial/<cluster>/<method>_<trial>``)."""

    def __init__(self, coverages, columns, rated=None):
        self.coverages = [float(c) for c in coverages]
        self.columns = list(columns)
        self.rated = rated
        self.entries: dict[str, dict[str, Any]] = {}

    def add_split(self, key: str, pred: pd.DataFrame, y: pd.DataFrame, dates) -> bool:
        q = fit_split(pred, y, dates, self.coverages)
        if q:
            self.entries[key] = {'method': 'split', 'quantiles': None, 'q': q}
        return bool(q)

    def add_cqr(self, key: str, lower: pd.DataFrame, upper: pd.DataFrame, nominal: tuple[float, float], y: pd.DataFrame, dates) -> bool:
        q = fit_cqr(lower, upper, y, dates, self.coverages)
        if q:
            self.entries[key] = {'method': 'cqr', 'quantiles': [float(nominal[0]), float(nominal[1])], 'q': q}
        return bool(q)

    def band(self, key: str, coverage: float, pred: pd.DataFrame, lower: pd.DataFrame | None = None, upper: pd.DataFrame | None = None):
        """(lower, upper) frames for a predictor at a coverage level, or None when not calibrated."""
        entry = self.entries.get(key)
        if entry is None or float(coverage) not in entry['q']:
            return None
        q = entry['q'][float(coverage)]
        if entry['method'] == 'cqr':
            if lower is None or upper is None:
                return None
            return band_cqr(lower, upper, q, self.rated)
        return band_split(pred, q, self.rated)

    def to_json(self) -> dict[str, Any]:
        return {'coverages': self.coverages, 'columns': self.columns, 'rated': self.rated,
                'predictors': {k: {'method': v['method'], 'quantiles': v['quantiles'],
                                   'q': {str(c): [float(x) for x in qs] for c, qs in v['q'].items()}} for k, v in self.entries.items()}}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> 'Calibration':
        cal = cls(data['coverages'], data['columns'], data.get('rated'))
        for key, entry in data.get('predictors', {}).items():
            cal.entries[key] = {'method': entry['method'], 'quantiles': entry.get('quantiles'),
                                'q': {float(c): list(qs) for c, qs in entry['q'].items()}}
        return cal

    def save(self, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, CALIBRATION_FILE)
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(self.to_json(), stream, indent=1)
        return path

    @classmethod
    def load(cls, folder: str) -> 'Calibration | None':
        path = os.path.join(folder, CALIBRATION_FILE)
        if not os.path.exists(path):
            return None
        with open(path, encoding='utf-8') as stream:
            return cls.from_json(json.load(stream))
