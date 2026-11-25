# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The experiment graph behind the web editor: normalised templates, data shapes, and the dry build
that runs a network on dummy tensors to report the shape after every layer.

Nothing here touches the datasets: the shapes come from the case (horizon, lags, variables, NWP
levels), which is what the graph editor needs to say "64 grid tokens of 96 features enter the
merge" before any data exists.  Torch and the builders are imported only inside ``dry_build``.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from eforecast.config.schema import NetworkExperimentConfig
from eforecast.config.service import ServiceError, derived_facts, inside_roots

REPO_ROOT = Path(__file__).resolve().parents[2]
CALENDAR = {"hour": 2, "month": 2, "dayweek": 2, "minute": 2, "day": 2, "sp_index": 1}   # sin/cos pairs
DEFAULT_LAYER_PARAMS = {"act_func": "gelu", "filters": 16, "batch_norm": False, "use_nwp": True, "attention": "cbam",
                        "conv_dim": 2, "heads": 4, "model_name": "Informer", "pretrained": "none", "unet_type": "Unet",
                        "encoder_name": "resnet18"}


# ---------------------------------------------------------------------------- templates

def normalize_experiments(experiments: dict[str, Any] | None) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Every template in the editor form: {tag: {branch: [{type, name, params, enabled}]}}.

    Every layer parameter, ``size`` included, sits under ``params`` as ``{tag: fixed|choice|range,
    value: ...}``; the older shapes are folded into it by ``NetworkLayerConfig``.
    """
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for tag, spec in (experiments or {}).items():
        try:
            model = NetworkExperimentConfig.model_validate(spec)
        except ValidationError as exc:
            raise ServiceError(f"experiment {tag!r}: {exc.errors()[0]['msg']}") from exc
        out[tag] = {branch: [layer.model_dump() for layer in layers] for branch, layers in model.root.items()}
    return out


def module_templates(path: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """The ``experiments`` dict of a Python module (an ``experiments_module`` of a case), normalised."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = inside_roots(candidate, roots=None) if not str(candidate.resolve()).startswith(str(REPO_ROOT)) else candidate.resolve()
    if not resolved.is_file():
        raise ServiceError(f"{path} is not a file")
    namespace = runpy.run_path(str(resolved))
    experiments = namespace.get("experiments")
    if not isinstance(experiments, dict):
        raise ServiceError(f"{path} defines no `experiments` dict")
    return normalize_experiments(experiments)


def default_case_path() -> Path:
    raw = os.environ.get("EFORECAST_CASE") or str(REPO_ROOT / "configuration" / "cases" / "default.yaml")
    return Path(raw) if Path(raw).is_absolute() else REPO_ROOT / raw


def preset_templates(source: str, path: str | None = None) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Ready-made templates: ``repo`` = ``configuration/experiments.py``; ``default`` = a case YAML (``path``,
    else ``EFORECAST_CASE`` or ``configuration/cases/default.yaml``): its module templates overridden by its inline ones."""
    if source == "repo":
        return module_templates("configuration/experiments.py")
    if source == "default":
        from eforecast.config.config_files import load_document
        if path:
            candidate = Path(path).expanduser()
            candidate = candidate if candidate.is_absolute() else REPO_ROOT / candidate
            path = candidate.resolve() if str(candidate.resolve()).startswith(str(REPO_ROOT)) else inside_roots(candidate)
        else:
            path = default_case_path()
        if not path.exists():
            raise ServiceError(f"no case file at {path} (pick one on the Project tab, set EFORECAST_CASE or link configuration/cases/default.yaml)")
        document = load_document(path)
        out: dict[str, Any] = {}
        module = document.get("experiments_module")
        if module:
            try:
                out.update(module_templates(module if Path(module).is_absolute() else str((path.parent / module) if (path.parent / module).exists() else module)))
            except (ServiceError, Exception):  # noqa: BLE001 - a broken module must not hide the inline templates
                pass
        out.update(normalize_experiments(document.get("experiments") or {}))
        return out
    raise ServiceError(f"unknown preset source {source!r}: repo or default")


# ---------------------------------------------------------------------------- shapes from the case

def data_shapes(document: dict[str, Any]) -> dict[str, Any]:
    """Dummy input shapes per data tag (batch excluded) derived from the case, with notes on the guesses."""
    data = document.get("data") or {}
    project = document.get("project") or {}
    facts = derived_facts(document)
    steps = max(1, facts["steps"])
    variables = data.get("variables") or []
    nwp_vars = [v for v in variables if v.get("type") == "nwp"]
    obs_vars = [v for v in variables if v.get("type") == "timeseries"]
    cal_vars = [v for v in variables if v.get("type") == "calendar"]
    n_lags = lambda v: (len(v["lags"]) if isinstance(v.get("lags"), list) else (int(v["lags"]) if isinstance(v.get("lags"), int) and v["lags"] > 0 else steps))
    # NWP variables are hourly whatever the series resolution: their lag count is what the case writes
    nwp_lags = max([n_lags(v) for v in nwp_vars] + [1])
    levels = next((int(m.get("levels") or 8) for m in (project.get("nwp_models") or []) if m.get("levels")), 8)
    grid = 2 * levels + 1
    n_obs = sum(n_lags(v) for v in obs_vars)
    n_cal = sum(CALENDAR.get(v.get("name"), 1) for v in cal_vars)
    n_nwp_row = len(nwp_vars) * nwp_lags

    def rows_of(value: Any, default: int) -> int:   # the LSTM dataset rule: a list of lags, an int count, else the horizon steps
        if isinstance(value, list):
            return len(value) or default
        if isinstance(value, int) and value > 0:
            return value
        return default
    past = rows_of(data.get("global_past_lags"), steps)
    future = rows_of(data.get("global_future_lags"), steps)
    shapes: dict[str, Any] = {}
    notes: list[str] = []
    if nwp_vars:
        shapes["nwp"] = [nwp_lags, len(nwp_vars), grid, grid]
        notes.append(f"nwp: {len(nwp_vars)} variables x {nwp_lags} hourly lags on a {grid}x{grid} grid ({levels} levels around the site); head: {steps} outputs")
    else:
        notes.append("nwp: no NWP variable in the case; a 6 x 2 x 17 x 17 placeholder is used")
        shapes["nwp"] = [steps, 2, 17, 17]
    shapes["row_all"] = [max(1, n_obs + n_cal + n_nwp_row)]
    shapes["row_obs"] = [max(1, n_obs)]
    shapes["row_calendar"] = [max(1, n_cal)]
    shapes["row_nwp"] = [max(1, n_nwp_row)]
    shapes["row_obs_calendar"] = [max(1, n_obs + n_cal)]
    shapes["row_obs_nwp"] = [max(1, n_obs + n_nwp_row)]
    shapes["lstm"] = {"past_obs": [past, max(1, len(obs_vars))], "past_calendar": [past, max(1, n_cal)],
                      "future_calendar": [future, max(1, n_cal)], "future_data": [future, max(1, len(nwp_vars))]}
    notes.append(f"lstm: {past} past rows (global_past_lags) and {future} future rows (global_future_lags) at the series resolution; "
                 f"row features are counted from the variables (obs lags {n_obs}, calendar {n_cal}, nwp statistics {n_nwp_row}); the datasets may differ slightly")
    return {"shapes": shapes, "steps": steps, "notes": notes}


# ---------------------------------------------------------------------------- dry build

def _resolve_params(document: dict[str, Any], method: str | None, tag: str) -> dict[str, Any]:
    """Layer parameters for a probe: fixed values of the method family (first choice, low end of a range)."""
    params = dict(DEFAULT_LAYER_PARAMS)
    family = ((document.get("methods") or {}).get("method_params") or {}).get(method or "", {}) or {}
    for key, value in family.items():
        if isinstance(value, dict) and set(value) == {"choices"} and value["choices"]:
            params[key] = value["choices"][0]
        elif isinstance(value, dict) and set(value) == {"range"} and value["range"]:
            params[key] = value["range"][0]
        elif isinstance(value, list) and value and key not in ("experiment_tag",):
            params[key] = value[0] if key in {"attention", "scheduler", "merge", "compress", "act_func", "optimizer"} else value
        elif not isinstance(value, (dict, list)):
            params[key] = value
    params["experiment_tag"] = tag
    return params


def unknown_parameters(experiment: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Parameter names the catalogue does not list for their layer type.

    Not an error: the catalogue can lag the code, and the builder simply passes anything it is
    given.  It is almost always a typo, so the dry build says so.
    """
    from eforecast.config.layer_catalogue import LAYERS, param_names

    warnings: list[str] = []
    for branch, layers in experiment.items():
        for index, layer in enumerate(layers):
            layer_type = layer.get("type")
            if layer_type not in LAYERS:
                continue
            known = set(param_names(layer_type))
            for name, param in (layer.get("params") or {}).items():
                if name in known:
                    continue
                # ('concatenate', []) and ('concatenate', {1}) alike: the legacy tuples always
                # carried a size for layers that take none, empty or not, and the conversion kept
                # whatever was in them.  Nobody typed it, so it is not worth a warning.  A size on
                # a layer that does take parameters is still only exempt when it carries nothing.
                if name == "size" and (not known or param.get("value") in (None, [], {}, "")):
                    continue
                warnings.append(f"{branch}[{index}] {layer_type}: {name!r} is not a parameter of this layer"
                                f" (it takes {', '.join(sorted(known)) or 'none'})")
    return warnings


#: What a value of each catalogue type is, in the words a finding uses.
TAKES = {"int": "a number", "float": "a number", "bool": "yes or no", "str": "text", "list": "branch names"}


def _is_a(value: Any, kind: str) -> bool:
    if kind in ("int", "float"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "bool":
        return isinstance(value, bool) or (isinstance(value, (int, float)) and value in (0, 1))
    if kind == "str":
        return isinstance(value, str)
    if kind == "list":
        return isinstance(value, str) or (isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value))
    return True


def _values(param: Any) -> list[Any]:
    """Every value a stored parameter can take: the fixed one, each choice, the bounds of a range.

    The value of a ``list`` parameter is a list itself, so a fixed one is one value, not several.
    """
    if not isinstance(param, dict) or "tag" not in param:
        return [param]
    value = param.get("value")
    if param.get("tag") in ("choice", "range") and isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def mistyped_parameters(experiment: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Layer parameters whose value is not of the kind the catalogue declares, and branch names a graph does not have.

    One entry a value: ``where`` (``output[0] cross_attention``), ``name``, ``value``, ``takes``
    and what follows from it (``effect``, ``remedy``). Nothing is refused: the builder hands a
    layer what it is given. An empty value is a parameter that is not set and is passed over,
    and so is ``size``, which has forms of its own (``'linear'``, the ``[]`` of a layer without one).

    The real case: all three cases of the first user carried ``attention_query: 64`` on the
    ``cross_attention`` of ``mlp1``. The graph editor had started the parameter at 64 when it
    was added, the layer asked which branches 64 names, and every trial that drew ``mlp1``
    ended in ``'int' object is not iterable`` -- 22 of the 127 MLP trials of one run.
    """
    from eforecast.config.layer_catalogue import LAYERS, layer_params

    branches = [branch for branch in experiment if branch not in ("output", "hidden_layer")]
    found: list[dict[str, Any]] = []
    for branch, layers in experiment.items():
        for index, layer in enumerate(layers or []):
            layer_type = layer.get("type")
            if layer_type not in LAYERS or layer.get("enabled") is False:
                continue
            declared = {entry["name"]: entry["type"] for entry in layer_params(layer_type)}
            where = f"{branch}[{index}] {layer_type}"
            for name, param in (layer.get("params") or {}).items():
                kind = declared.get(name)
                if name == "size" or kind not in TAKES:
                    continue
                for value in _values(param):
                    if value is None or (isinstance(value, str) and not value.strip()) or value == []:
                        continue
                    if not _is_a(value, kind):
                        entry = {"where": where, "name": name, "value": value, "takes": TAKES[kind]}
                        if kind == "list":
                            # Not "remove it": a change to any graph changes what every family of
                            # the case is, and a case that is trained is then trained again.
                            entry.update(effect="the layer reads it as not set, so the flat branches are the queries",
                                         remedy="Nothing has to be done for the layer to work. To clear this, remove "
                                                f"{name} from the layer or name branches of the graph"
                                                f" ({', '.join(branches) or 'it has none'}) -- before the case is trained: "
                                                "a change to a graph makes the models of a case those of another "
                                                "configuration, and they are trained again.")
                        elif kind == "bool":
                            entry.update(effect="a value that is not empty counts as yes, whatever it says",
                                         remedy="Set it to true or false.")
                        else:
                            entry.update(effect="the layer is handed the value as it stands, and one that computes with it "
                                                "fails as the network is built; the dry build of the graph shows whether it does",
                                         remedy=f"Enter {TAKES[kind]}, or empty the value to leave {name} unset.")
                        found.append(entry)
                    elif kind == "list":
                        names = [value] if isinstance(value, str) else list(value)
                        known = [n for n in names if any(n == b or n == b.split("/")[-1] for b in branches)]
                        strangers = [n for n in names if n not in known]
                        if strangers:
                            found.append({
                                "where": where, "name": name, "value": value, "takes": TAKES[kind],
                                "strangers": strangers,
                                "effect": (f"only {', '.join(known)} {'is' if len(known) == 1 else 'are'} used" if known else
                                           "no branch is then the query, and a learned query pools every branch"),
                                "remedy": f"Name branches of the graph ({', '.join(branches) or 'it has none'}), "
                                          f"or remove {name} to let the flat branches be the queries."})
    return found


def describe_mistyped(entry: dict[str, Any]) -> str:
    """One finding of ``mistyped_parameters`` as a sentence, the way ``unknown_parameters`` words its own."""
    if entry.get("strangers"):
        return (f"{entry['where']}: {entry['name']} names {', '.join(map(str, entry['strangers']))}, "
                f"and the graph has no such branch; {entry['effect']}")
    return (f"{entry['where']}: {entry['name']} is {entry['value']!r}, and the layer takes {entry['takes']} there; "
            f"{entry['effect']}")


def _count(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def compute_preview(payload, context):
    """Common-worker target. The worker process owns every temporary model/cache."""
    return dry_build(payload['document'], payload['tag'], payload.get('method'))


def dry_build(document: dict[str, Any], tag: str, method: str | None = None) -> dict[str, Any]:
    """Build the network of ``document.experiments[tag]`` on dummy tensors; shapes and parameter counts per layer."""
    import torch
    from eforecast.common_utils.train_utils import fix_convolutional_names
    from eforecast.config.loader import _inline_experiment_layer_params, _inline_experiments
    from eforecast.deep_models.pytorch_2x.builders import build_graph, build_layer, branches_for_merge

    experiments = normalize_experiments(document.get("experiments") or {})
    if tag not in experiments:
        raise ServiceError(f"experiment {tag!r} is not defined inline in the case")
    models = {tag: NetworkExperimentConfig.model_validate(experiments[tag])}
    legacy = _inline_experiments(models)[tag]
    layer_params = _inline_experiment_layer_params(models, resolve=True)   # no trial here: first choice, low bound
    project = document.get("project") or {}
    params = _resolve_params(document, method, tag)
    params["experiment_layer_params"] = layer_params
    info = data_shapes(document)
    params["n_out"] = info["steps"]
    params["is_lstm_output"] = False
    experiment = fix_convolutional_names(project.get("horizon_type"), {k: list(v) for k, v in legacy.items()}, params.get("conv_dim", 2))
    adapted = branches_for_merge(experiment)

    result: dict[str, Any] = {"tag": tag, "method": method, "data_shapes": info, "branches": {}, "merge": None, "head": None,
                              "total_params": None, "ok": True,
                              "warnings": unknown_parameters(experiments[tag])
                              + [describe_mistyped(entry) for entry in mistyped_parameters(experiments[tag])]}
    x: dict[str, Any] = {}
    for branch in [b for b in experiment if b not in ("output", "hidden_layer")]:
        base = branch.split("/")[-1]
        shape = info["shapes"].get(base)
        if shape is None:
            result["branches"][branch] = {"error": f"no data shape known for branch {branch!r}", "layers": []}
            result["ok"] = False
            continue
        if isinstance(shape, dict):
            x[branch] = {k: torch.zeros(4, *v) for k, v in shape.items()}
            entry = {"input": {k: list(v) for k, v in shape.items()}, "layers": []}
            probe_shape = {k: [10, *v] for k, v in shape.items()}
        else:
            x[branch] = torch.zeros(4, *shape)
            entry = {"input": list(shape), "layers": []}
            probe_shape = [10, *shape]
        layers = adapted[branch]
        # build layer by layer so an error points at the layer that raised it
        try:
            built = build_layer(probe_shape, layers, branch, dict(params), branch_layer_params=layer_params.get(tag, {}).get(base))
            for i, layer in enumerate(built.layers_built):
                shape_out = getattr(layer, "output_shape", None)
                entry["layers"].append({"index": i, "type": adapted[branch][i][0], "configured": experiment[branch][i][0],
                                        "output_shape": [int(s) for s in shape_out][1:] if shape_out is not None else None,
                                        "params": _count(layer)})
            entry["output"] = entry["layers"][-1]["output_shape"] if entry["layers"] else entry["input"]
        except Exception as error:  # noqa: BLE001
            failing = None
            for n in range(1, len(layers) + 1):
                try:
                    build_layer(probe_shape, layers[:n], branch, dict(params), branch_layer_params=layer_params.get(tag, {}).get(base))
                except Exception:  # noqa: BLE001
                    failing = n - 1
                    break
            entry["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            entry["error_layer"] = failing
            result["ok"] = False
        result["branches"][branch] = entry
    if not result["ok"]:
        return result
    try:
        net = build_graph(x, {k: list(v) for k, v in experiment.items()}, params, is_fuzzy=False, device="cpu")
        merge_block = net.model_layers_built["hidden_output"]
        merge_layers = []
        for i, layer in enumerate(merge_block.layers_built):
            shape_out = getattr(layer, "output_shape", None)
            item = {"index": i, "type": experiment["output"][i][0], "output_shape": [int(s) for s in shape_out][1:] if shape_out is not None else None,
                    "params": _count(layer)}
            if hasattr(layer, "kv_names"):
                item.update({"query_branches": list(getattr(layer, "query_names", [])) or ["(learned query)"],
                             "kv_branches": list(layer.kv_names),
                             "tokens": int(sum(layer.views[n][0] for n in layer.kv_names)), "d_model": int(layer.d_model)})
            merge_layers.append(item)
        head = net.model_layers_built["output"]
        with torch.no_grad():
            out = net(x)
        result["merge"] = {"layers": merge_layers, "output": merge_layers[-1]["output_shape"] if merge_layers else None}
        result["head"] = {"type": type(head).__name__, "n_out": int(out.shape[-1]), "params": _count(head),
                          "quantiles": list(getattr(head, "quantiles", None) or []) or None,
                          "experts": int(head.experts) if hasattr(head, "experts") else None}
        result["total_params"] = _count(net)
    except Exception as error:  # noqa: BLE001
        result["ok"] = False
        result["merge"] = {"error": f"{type(error).__name__}: {str(error)[:300]}", "layers": []}
    return result
