# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Legacy variable presets represented as typed VariableConfig instances."""

from typing import Any

from .schema import lag_step, HorizonType, VariableConfig


def _variable(name: str, **kwargs: Any) -> VariableConfig:
    return VariableConfig(name=name, **kwargs)


def _variables(
    problem_type: str,
    horizon: int,
    horizon_type: HorizonType | str,
    ts_resolution: str,
    timezone: str = "UTC",
) -> list[VariableConfig]:
    horizon_type = HorizonType(horizon_type)
    if problem_type == "pv":
        result = [
            _variable(
                "ir039_ir108_vis006", type="image", source="eumetview",
                lags=[-0.5, -1, -1.5, -2, -3.5],
                transformer=["area_adjust", "resize"],
                transformer_params={"resize": [224, 224], "area_adjust": [250]},
                bands=[],
            ),
            _variable(
                "rgb_snow", type="image", source="eumetview",
                lags=[-0.5, -1, -1.5, -2, -3.5],
                transformer=["area_adjust", "resize"],
                transformer_params={"resize": [224, 224], "area_adjust": [250]},
                bands=[],
            ),
            _variable(
                "RBG", type="image", source="eumetdac",
                lags=[-0.5, -1, -1.5, -2, -3.5],
                transformer=["fillnan", "alignment", "resize", "inverse", "crop",
                             "normalize", "img_clear_sky"],
                transformer_params={
                    "resize": [211, 310], "fillnan": -1,
                    "alignment": [(62, 11), (0, 211), (220, 200), (310, 0)],
                    "crop": [[11, 121], [49, 199]], "normalize": [13, 22, 18],
                },
                bands=["channel_3", "channel_2", "channel_1"],
            ),
            _variable('Cloud', nwp_provider='ecmwf', lags=[-1, 0, 1]),
            _variable('High_Cloud', nwp_provider='ecmwf', lags=[-1, 0, 1]),
            _variable('Medium_Cloud', nwp_provider='ecmwf', lags=[-1, 0, 1]),
            _variable('Low_Cloud', nwp_provider='ecmwf', lags=[-1, 0, 1]),
            _variable('Flux', nwp_provider='ecmwf', lags=[-1, 0, 1]),
            _variable('Temperature', nwp_provider='ecmwf', lags=[-1, 0, 1]),
            _variable("hour", type="calendar", source="index", lags=[0]),
            _variable("month", type="calendar", source="index", lags=[0]),
        ]
        if horizon > 0:
            result.extend([
                _variable("Obs", type="timeseries", source="target",
                          lags=[-1, -2, -3, -4, -8, -10, -12], timezone=timezone),
            ])
        return result
    if problem_type == "wind":
        result = [
            _variable("WS", nwp_provider="ALL"),
            _variable("WD", nwp_provider="ecmwf"),
            _variable("Uwind", nwp_provider="ecmwf", lags=[-1, 0, 1]),
            _variable("Vwind", nwp_provider="ecmwf", lags=[-1, 0, 1]),
            _variable("Uwind10", nwp_provider="ecmwf", lags=[-1, 0, 1]),
            _variable("Vwind10", nwp_provider="ecmwf", lags=[-1, 0, 1]),
            _variable("WindGust", nwp_provider="ecmwf", lags=[-1, 0, 1]),
            _variable("hour", type="calendar", source="index", timezone=timezone),
            _variable("month", type="calendar", source="index"),
            _variable("horizon", type="calendar", source="index", timezone=timezone),
        ]
        if horizon > 0:
            result.append(_variable("Obs", type="timeseries", source="target",
                                    lags=3, timezone=timezone))
        return result
    if problem_type == "FA":
        lags1 = [-int(i) for i in range(1, 7)]
        lags2 = (
            [-int(i) for i in range(7, 11)] + [-int(i) for i in range(14, 16)]
            + [-int(i) for i in range(21, 23)]
        )
        lags_pred = [int(i) for i in range(2)]
        return [
            _variable("Final/Ζητούμενο", type="timeseries", source="target",
                      lags=lags2, timezone=timezone),
            _variable("Athens_24", type="timeseries", source="target", lags=lags1,
                      timezone=timezone, use_diff_between_lags=True),
            _variable("temp_max", type="timeseries", source="target",
                      lags=lags_pred + lags1, timezone=timezone),
            _variable("temp_min", type="timeseries", source="target",
                      lags=lags_pred, timezone=timezone),
            _variable("temp_mean", type="timeseries", source="target",
                      lags=lags_pred, timezone=timezone),
            _variable("rh", type="timeseries", source="target",
                      lags=lags_pred, timezone=timezone),
            _variable("precip", type="timeseries", source="target",
                      lags=lags_pred, timezone=timezone),
            _variable("hdd_h", type="timeseries", source="target",
                      lags=lags_pred, timezone=timezone),
            _variable("hdd_h2", type="timeseries", source="target",
                      lags=lags_pred, timezone=timezone),
            _variable("Temp_max", type="timeseries", source="nwp_dataset",
                      lags=[1, 0, -1, -2, -3], timezone=timezone),
            _variable("Temp_min", type="timeseries", source="nwp_dataset",
                      lags=[1, 0, -1, -2, -3], timezone=timezone),
            _variable("Temperature", nwp_provider="ALL"),
            _variable("Cloud", nwp_provider="ALL"),
            _variable("WS", nwp_provider="ALL"),
            _variable("WD", nwp_provider="ALL"),
            _variable("dayweek", type="calendar", source="index",
                      lags=lags_pred, timezone=timezone),
            _variable("sp_index", type="calendar", source="index",
                      lags=lags_pred, timezone=timezone),
            _variable("month", type="calendar", source="index",
                      lags=lags_pred, timezone=timezone),
        ]
    if problem_type == "load":
        if horizon > 0:
            lags = (
                [-int(i) for i in range(1, 13)] + [-int(i) for i in range(22, 28)]
                + [-int(i) for i in range(47, 53)] + [-int(i) for i in range(166, 176)]
                + [-192]
            )
        elif horizon_type == HorizonType.DAY_AHEAD:
            lags = (
                [-int(i) for i in range(48, 60)] + [-int(i) for i in range(72, 77)]
                + [-int(i) for i in range(96, 100)] + [-int(i) for i in range(120, 122)]
                + [-int(i) for i in range(144, 146)] + [-int(i) for i in range(166, 176)]
                + [-int(i) for i in range(190, 192)] + [-216]
            )
        else:
            lags = (
                [-int(i) for i in range(24, 36)] + [-int(i) for i in range(48, 54)]
                + [-int(i) for i in range(72, 77)] + [-int(i) for i in range(96, 100)]
                + [-int(i) for i in range(120, 122)] + [-int(i) for i in range(144, 146)]
                + [-int(i) for i in range(166, 176)] + [-int(i) for i in range(190, 192)]
                + [-216]
            )
        lags_days = [-int(24 * i) for i in range(0, 8)]
        return [
            _variable("load", type="timeseries", source="target", lags=lags, timezone=timezone),
            _variable("Temp_max", type="timeseries", source="nwp_dataset",
                      lags=lags_days, timezone=timezone),
            _variable("Temp", type="timeseries", source="nwp_dataset",
                      lags=[0, -1, -2, -3, -24, -48, -168], timezone=timezone),
            _variable("Temperature", nwp_provider="ALL"),
            _variable("Cloud", nwp_provider="ALL"),
            _variable("WS", nwp_provider="ALL"),
            _variable("WD", nwp_provider="ALL"),
            _variable("dayweek", type="calendar", source="index", timezone=timezone),
            _variable("sp_index", type="calendar", source="index",
                      lags=lags_days, timezone=timezone),
            _variable("hour", type="calendar", source="index", timezone=timezone),
            _variable("month", type="calendar", source="index", timezone=timezone),
        ]
    raise NotImplementedError(f"Define variables for type {problem_type}")


def variables(
    problem_type: str,
    horizon: int,
    horizon_type: HorizonType | str,
    ts_resolution: str,
    timezone: str = "UTC",
) -> list[VariableConfig]:
    """Return the same expanded variable dictionaries as legacy ``variables``."""
    result = _variables(problem_type, horizon, horizon_type, ts_resolution, timezone)
    resolution = lag_step(ts_resolution)
    horizon_type = HorizonType(horizon_type)
    for variable in result:
        lags = variable.lags
        if lags is None or lags == 0:
            variable.lags = (
                [0] if horizon_type != HorizonType.SHORT_TERM
                else [int(i) for i in range(int(horizon / resolution))]
            )
        elif isinstance(lags, int):
            variable.lags = [-int(i) for i in range(int(lags / resolution))]
    return result


def rbf_variables(problem_type: str) -> list[list[str]]:
    return {
        "pv": [["hour", "month", "Flux_0_ecmwf_center", "Cloud_0_ecmwf_center"]],
        "wind": [["Uwind_0_ecmwf_center", "Vwind_0_ecmwf_center",
                  "Uwind10_0_ecmwf_center", "Vwind10_0_ecmwf_center"]],
        "load": [["sp_index", "hour", "month", "Temp"]],
        "FA": [["sp_index", "month", "temp_max", "dayweek"]],
    }.get(problem_type, [[]])
