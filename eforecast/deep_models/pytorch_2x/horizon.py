# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Horizon-dependent loss weights for short_term networks.

The forecast error grows with the step, so a plain MSE over the steps is dominated by the late
ones and the trunk is trained mostly for them.  A weight per step, inversely proportional to a
reference error of that step, lets every step pull equally.  ``horizon_weights`` in the method
parameters selects the reference:

* ``None`` / ``'none'``: plain MSE (default)
* ``'persistence'``: the error of the persistence forecast per step, computed once from the
  training targets (rows that follow each other in time), the natural yardstick of a horizon
* ``'model'``: the network's own per-step RMSE on the training batches, an exponential moving
  average refreshed after every epoch (starts equal)
* a list of ``n_out`` numbers: fixed weights

Weights are normalised to mean one, so the loss keeps its scale; ``horizon_weight_floor`` (0.05)
bounds how small a reference error may get relative to the largest, which caps the ratio between
the heaviest and the lightest step at 400.
"""
import numpy as np
import pandas as pd
import torch

EMA = 0.7


def weights_from_errors(errors, floor=0.05):
    errors = np.asarray(errors, dtype=np.float64)
    errors = np.maximum(errors, floor * np.nanmax(errors) if np.nanmax(errors) > 0 else 1.0)
    weights = 1.0 / errors ** 2
    return weights * weights.size / weights.sum()


def persistence_errors(y):
    """Per-step MAE of the persistence forecast (last observed value = step 0 of the previous row).

    ``y``: DataFrame (dates x steps).  Rows are used only when the previous row is one time step
    earlier.  Returns None when fewer than 50 such rows exist or ``y`` is not a DataFrame."""
    if not isinstance(y, pd.DataFrame) or y.shape[1] < 2 or not isinstance(y.index, pd.DatetimeIndex):
        return None
    dt = y.index.to_series().diff()
    step = dt.mode().iloc[0] if dt.notna().any() else None
    if step is None:
        return None
    consecutive = (dt == step).values
    last_obs = y.iloc[:, 0].shift(1).values
    errors = np.abs(y.values[consecutive] - last_obs[consecutive, None])
    if errors.shape[0] < 50:
        return None
    return np.nanmean(errors, axis=0)


class HorizonWeights:
    """The current per-step weights of a training run and their update rule."""

    def __init__(self, mode, n_out, floor=0.05):
        self.n_out = int(n_out)
        self.floor = floor
        self.fixed = None
        if isinstance(mode, (list, tuple, np.ndarray)):
            self.fixed = np.asarray(mode, dtype=np.float64)
            self.mode = 'fixed'
        else:
            self.mode = (mode or 'none') if isinstance(mode, str) else 'none'
        if self.mode not in {'none', 'persistence', 'model', 'fixed'}:
            raise ValueError(f"horizon_weights {mode!r}: none | persistence | model | list of {self.n_out} weights")
        self.reference = None      # per-step reference error (EMA for 'model')
        self.current = None

    @property
    def active(self):
        return self.mode != 'none' and self.n_out > 1

    def initial(self, y_train):
        """Weights known before training: persistence from the targets, or the fixed list."""
        if self.mode == 'fixed':
            self.current = self.fixed * self.fixed.size / self.fixed.sum()
        elif self.mode == 'persistence':
            errors = persistence_errors(y_train)
            if errors is None:
                print('horizon weights: persistence needs consecutive target rows; falling back to the model error')
                self.mode = 'model'
            else:
                self.reference = errors
                self.current = weights_from_errors(errors, self.floor)
        return self.current

    def update(self, step_rmse):
        """After an epoch ('model' mode): refresh the running per-step error and the weights."""
        if self.mode != 'model' or step_rmse is None:
            return self.current
        step_rmse = np.asarray(step_rmse, dtype=np.float64)
        if step_rmse.shape[0] != self.n_out or not np.all(np.isfinite(step_rmse)):
            return self.current
        self.reference = step_rmse if self.reference is None else EMA * self.reference + (1 - EMA) * step_rmse
        self.current = weights_from_errors(self.reference, self.floor)
        return self.current

    def describe(self):
        if self.current is None:
            return f'horizon weights: {self.mode} (not set yet)'
        w = np.round(self.current, 2)
        return f'horizon weights ({self.mode}): min {w.min():.2f} max {w.max():.2f} -> {w.tolist()}'


class HorizonWeightedMSE(torch.nn.Module):
    """MSE over the steps with one weight per step (mean one); plain MSE until weights are set."""

    def __init__(self):
        super(HorizonWeightedMSE, self).__init__()
        self.weights = None

    def set_weights(self, weights):
        self.weights = None if weights is None else torch.as_tensor(np.asarray(weights), dtype=torch.float32)

    def forward(self, output, y):
        se = (output - y) ** 2
        if self.weights is None or se.dim() < 2 or se.shape[1] != self.weights.numel():
            return se.mean()
        return (se.mean(0) * self.weights.to(se.device)).mean()


class HorizonWeightedPinball(torch.nn.Module):
    """Pinball (quantile) loss of a quantile head, one weight per step (mean one).

    ``output`` is ``(batch, n_quantiles, n_out)`` from ``QuantileHead``, ``y`` is ``(batch, n_out)``.
    For quantile q and error e = y - q_hat the loss is max(q e, (q - 1) e); it is averaged over the
    batch and the quantiles, weighted per step like ``HorizonWeightedMSE`` and averaged over the steps.
    Minimising it makes every quantile output a calibrated estimate of that quantile of the target."""

    def __init__(self, quantiles):
        super(HorizonWeightedPinball, self).__init__()
        q = torch.as_tensor(sorted(float(v) for v in quantiles), dtype=torch.float32)
        self.register_buffer('q', q.view(1, -1, 1))
        self.weights = None

    def set_weights(self, weights):
        self.weights = None if weights is None else torch.as_tensor(np.asarray(weights), dtype=torch.float32)

    def forward(self, output, y):
        if output.dim() == 2:                       # a plain head under the pinball loss: the median only
            output = output.unsqueeze(1)
            q = torch.full((1, 1, 1), 0.5, device=output.device)
        else:
            q = self.q.to(output.device)
        error = y.unsqueeze(1) - output
        pinball = torch.maximum(q * error, (q - 1) * error).mean(dim=(0, 1))     # per step
        if self.weights is None or pinball.numel() != self.weights.numel():
            return pinball.mean()
        return (pinball * self.weights.to(pinball.device)).mean()


class MedianMetric(torch.nn.Module):
    """A metric of the deterministic training loop (MAE, SSE) applied to the median of a quantile head,
    so early stopping, checkpoints and the study value keep their meaning."""

    def __init__(self, metric, quantiles):
        super(MedianMetric, self).__init__()
        self.metric = metric
        qs = sorted(float(v) for v in quantiles)
        self.index = int(np.argmin([abs(v - 0.5) for v in qs]))

    def forward(self, output, y):
        if output.dim() == 3:
            output = output[:, self.index]
        return self.metric(output, y)
