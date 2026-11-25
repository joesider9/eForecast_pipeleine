# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""What ``rbf_var_imp`` entries resolve to, for the clustering editor.

``methods.method_params.clustering.rbf_var_imp`` is a list of rule sets, **one fuzzy network per
entry**, and each inner entry names variables of the ``row_all`` dataset in one of four ways
(:func:`eforecast.common_utils.dataset_utils.get_fuzzy_variables` and
``tf_rbf_clusterer.fit``):

===============  ========================================================================
``hour``         one of the five calendar names: expanded to ``hour_sin`` and ``hour_cos``
``Flux_0_..._c`` an exact ``row_all`` column: that one column
``Flux``         any other text: every column containing it, case-insensitively, as a group
``Flux$center``  two parts: every column containing both
===============  ========================================================================

The trap is the third kind.  Nothing tells you how many columns a pattern caught until training
fails with ``Cannot find variables associated with X`` -- and a pattern that catches *too much*
never fails at all.  :func:`resolve` answers that question for the editor.

The column universe comes from the best source available:

``dataset``   the built row dataset (``dataset_row_data.pickle``): exact column names
``declared``  the variables of the case: calendar names and the ``<var>_<lag>_<model>`` prefixes
              of the NWP statistics, whose statistic suffix (``_center``, ``_mean``, ...) the
              compressor appends when the dataset is built

Nothing here imports torch or builds a dataset.
"""
from __future__ import annotations

import os
from typing import Any

# tf_rbf_clusterer.fit expands exactly these to <name>_sin and <name>_cos
CALENDAR_NAMES: tuple[str, ...] = ("month", "day", "dayweek", "hour", "minute")


def _project(document: dict[str, Any]) -> dict[str, Any]:
    return document.get("project") or {}


def _variables(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [v for v in ((document.get("data") or {}).get("variables") or []) if isinstance(v, dict)]


def declared_columns(document: dict[str, Any]) -> dict[str, list[str]]:
    """The column names and prefixes the case declares, grouped by where they come from."""
    calendar: list[str] = []
    observations: list[str] = []
    nwp: list[str] = []
    models = [m.get("model") for m in (_project(document).get("nwp_models") or []) if isinstance(m, dict)]
    models = [m for m in models if m]
    for variable in _variables(document):
        name, kind = variable.get("name"), variable.get("type")
        if not name:
            continue
        if kind == "calendar":
            calendar.extend([f"{name}_sin", f"{name}_cos"] if name in CALENDAR_NAMES else [name])
        elif kind == "nwp":
            lags = variable.get("lags")
            lags = lags if isinstance(lags, list) else [0 if lags is None else lags]
            provider = variable.get("nwp_provider")
            providers = models if provider in (None, "ALL") else [provider]
            for lag in lags:
                for model in providers or [None]:
                    nwp.append(f"{name}_{lag}_{model}" if model else f"{name}_{lag}")
        else:
            # a timeseries column is named <name>_lag_<lag> (dataset_creator.get_data)
            lags = variable.get("lags")
            if isinstance(lags, list) and lags:
                observations.extend(f"{name}_lag_{lag}" for lag in lags)
            elif isinstance(lags, int):
                observations.extend(f"{name}_lag_{-lag}" for lag in range(lags)) if lags > 1 else \
                    observations.append(f"{name}_lag_0")
            else:
                observations.append(f"{name}_lag_0")
    return {"calendar": calendar, "observations": observations, "nwp": nwp}


def dataset_columns(document: dict[str, Any]) -> list[str] | None:
    """Columns of the built row dataset, or ``None`` when it has not been created yet."""
    try:
        import joblib

        from eforecast.config.loader import to_static_data
        from eforecast.config.schema import ExperimentConfig
        from eforecast.init.initialize import create_paths

        static_data = create_paths(to_static_data(ExperimentConfig.model_validate(document)), dry_run=True)
        path = os.path.join(static_data["path_data"], "dataset_row_data.pickle")
        if not os.path.isfile(path):
            return None
        stored = joblib.load(path)
    except Exception:                                   # an incomplete case, or no pipeline packages
        return None
    columns: list[str] = []
    for frame in (stored or {}).values():
        if frame is not None and hasattr(frame, "columns"):
            columns.extend(str(c) for c in frame.columns)
    return columns or None


def _match(entry: str, columns: list[str]) -> list[str]:
    """What ``get_fuzzy_variables`` would select for ``entry``."""
    if "$" in entry:
        first, _, second = entry.partition("$")
        return [c for c in columns if first.lower() in c.lower() and second.lower() in c.lower()]
    return [c for c in columns if entry.lower() in c.lower()]


def _match_series(entry: str, prefixes: list[str]) -> list[str]:
    """NWP series an entry points at, before the compressor has appended its statistic.

    Both directions count.  ``Flux`` is contained in the series ``Flux_0_ecmwf`` and will catch all
    of its statistics; ``Flux_0_ecmwf_center`` instead *starts with* that series and names one of
    them.  Neither is a column yet, which is the whole reason this case needs reporting.
    """
    head = entry.partition("$")[0] if "$" in entry else entry
    return [p for p in prefixes
            if head.lower() in p.lower() or head.lower().startswith(p.lower())]


def resolve_entry(entry: Any, columns: list[str], declared: dict[str, list[str]],
                  calendar_declared: set[str], exact: bool) -> dict[str, Any]:
    """One entry resolved: its kind, what it matches, and whether that looks intended."""
    text = "" if entry is None else str(entry).strip()
    if not text:
        return {"entry": entry, "kind": "empty", "matched": [], "count": 0,
                "level": "error", "note": "empty entry"}

    if text in CALENDAR_NAMES:
        matched = [f"{text}_sin", f"{text}_cos"]
        if text not in calendar_declared:
            return {"entry": text, "kind": "calendar", "matched": matched, "count": 2, "level": "error",
                    "note": f"the clusterer would use {text}_sin and {text}_cos, but {text!r} is not a "
                            f"calendar variable of this case"}
        return {"entry": text, "kind": "calendar", "matched": matched, "count": 2, "level": "ok",
                "note": f"a calendar variable: the clusterer uses {text}_sin and {text}_cos"}

    if text in columns:
        return {"entry": text, "kind": "column", "matched": [text], "count": 1, "level": "ok",
                "note": "one exact column"}

    matched = _match(text, columns)
    if matched:
        level = "ok" if len(matched) == 1 else "warn"
        note = ("one column, matched as a pattern" if len(matched) == 1
                else f"a group of {len(matched)} columns")
        return {"entry": text, "kind": "group", "matched": matched, "count": len(matched),
                "level": level, "note": note}

    # no column yet: an NWP prefix still tells us it will resolve once the datasets are built
    prefixes = _match_series(text, declared.get("nwp") or [])
    if prefixes and not exact:
        note = (f"matches {len(prefixes)} NWP series; the statistic suffix (_center, _mean, ...) "
                f"is added when the datasets are built")
        if "$" in text:
            # the part after $ almost always names the statistic, which does not exist yet
            note += "; the part after $ can only be checked then"
        return {"entry": text, "kind": "nwp_prefix", "matched": prefixes, "count": len(prefixes),
                "level": "warn", "note": note}

    return {"entry": text, "kind": "unknown", "matched": [], "count": 0, "level": "error",
            "note": "no column matches; training would stop at "
                    f"'Cannot find variables associated with {text}'"}


def resolve(document: dict[str, Any], rule_sets: Any = None) -> dict[str, Any]:
    """Resolve every entry of every rule set of ``rbf_var_imp``.

    ``rule_sets`` is the value being edited; when it is omitted the one stored in the document is
    used.  A flat list (the ``Distributed`` form) is read as a single rule set.
    """
    if rule_sets is None:
        families = ((document.get("methods") or {}).get("method_params") or {})
        rule_sets = (families.get("clustering") or {}).get("rbf_var_imp")
    if rule_sets is None:
        rule_sets = []
    if isinstance(rule_sets, list) and rule_sets and not any(isinstance(item, list) for item in rule_sets):
        rule_sets = [rule_sets]                          # the flat Distributed form
    if not isinstance(rule_sets, list):
        rule_sets = []

    declared = declared_columns(document)
    built = dataset_columns(document)
    exact = built is not None
    columns = built if exact else sorted({*declared["calendar"], *declared["observations"]})
    calendar_declared = {v.get("name") for v in _variables(document) if v.get("type") == "calendar"}

    resolved = []
    for index, rule_set in enumerate(rule_sets):
        entries = rule_set if isinstance(rule_set, list) else [rule_set]
        items = [resolve_entry(entry, columns, declared, calendar_declared, exact) for entry in entries]
        resolved.append({
            "index": index,
            "entries": items,
            "columns": sum(item["count"] for item in items),
            "level": ("error" if any(i["level"] == "error" for i in items)
                      else "warn" if any(i["level"] == "warn" for i in items)
                      else "ok"),
        })
    return {
        "source": "dataset" if exact else "declared",
        "column_count": len(columns),
        "rule_sets": resolved,
        "suggestions": {
            "calendar": sorted(n for n in CALENDAR_NAMES if n in calendar_declared),
            "columns": sorted(set(columns))[:2000],
            "nwp": sorted({v["name"] for v in _variables(document)
                           if v.get("type") == "nwp" and v.get("name")}),
            "nwp_series": sorted(set(declared["nwp"]))[:2000],
        },
        "note": ("resolved against the built row dataset" if exact else
                 "no row dataset yet: resolved against the variables the case declares, so NWP "
                 "statistics are matched by their series prefix"),
    }
