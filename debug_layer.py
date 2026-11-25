# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Starting point for developing a layer of ``eforecast/deep_models/pytorch_2x/layers.py``.

Builds one branch (or a whole template) on dummy tensors whose shapes come from a case, prints the
shape and parameter count after every layer, runs a forward pass, checks for NaN, runs a backward
pass and reports the parameters that received no gradient.  No dataset is needed, so the loop
"edit the class, run this, read the shapes" takes seconds.  Put a breakpoint in your layer's
``forward`` and run it under the PyCharm debugger with the same arguments.

    # one branch, layers as name[:size] separated by commas (size as in the templates)
    python debug_layer.py --branch nwp --layers "conv:3,conv:3,tokens,dense:64"
    python debug_layer.py --branch lstm --layers "lstm_att:2,dense:32" --method LSTM
    # the branch merged with a flat one, through a merge layer
    python debug_layer.py --branch nwp --layers "conv:3,tokens,dense:64" --with "row_calendar=dense:32" --merge cross_attention:64
    # a whole template of the case, as the web UI's dry build
    python debug_layer.py --tag lstm1 --method LSTM
    # layer parameters (the per-layer ``params`` of the template) and method parameters
    python debug_layer.py --branch nwp --layers "grid_transformer:64" --layer-params '{"depth": 1, "heads": 2}' --params '{"act_func": "gelu"}'

The case decides the shapes (``--config``, default ``$EFORECAST_CASE`` or configuration/cases/default.yaml);
``--shape 6,4,17,17`` overrides them for the branch under test.

Adding a layer
--------------
1. A class in layers.py with the builder's contract::

       class my_layer(nn.Module):
           def __init__(self, shape, params, size, name_scope, layer_id, train=True):
               super().__init__()
               self.name = f'{name_scope}_my_layer_{layer_id}'
               ...                                   # modules from ``shape`` (batch first, batch = 10 in the probe)
               self.apply(_init_weights)
               with torch.no_grad():                 # the builder reads this to feed the next layer
                   self.output_shape = list(self.forward(torch.zeros(10, *shape[1:])).shape)

           def forward(self, x):
               ...

   ``shape`` is a list ``[batch, ...]`` (or a dict for the ``lstm`` branch), ``size`` the template value
   (``dense:64`` -> 64), ``params`` the method parameters merged with the layer's own ``params``.
2. Register it in ``layers_func()`` (layers.py) under the name used in the templates.
3. Describe it in ``eforecast/config/layer_catalogue.py`` (inputs, size, params, description); the test
   ``tests/test_graph_service.py::test_catalogue_covers_registry`` fails until it is there, and the web UI's
   layer guide and "+ layer" menu read it.
4. Run this script on it, then a template through ``--tag``, then ``debug_experiment.py`` on the
   project's datasets, then a real trial::

       python run_training.py --config <case> --prepare-only
       python eforecast/training/command_line/trial_runner.py --path-weights <cluster_dir>/<method>/test_0 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_layers(text: str) -> list[tuple[str, object]]:
    """``"conv:3,tokens,dense:64"`` -> ``[('conv', 3), ('tokens', []), ('dense', 64)]``."""
    layers = []
    for item in [t.strip() for t in text.split(",") if t.strip()]:
        name, _, size = item.partition(":")
        if size == "":
            layers.append((name, []))
        else:
            try:
                layers.append((name, json.loads(size)))
            except ValueError:
                layers.append((name, size))
    return layers


def fmt(shape) -> str:
    if isinstance(shape, dict):
        return "{" + ", ".join(f"{k}: {fmt(v)}" for k, v in shape.items()) + "}"
    return "(" + " x ".join(str(int(s)) for s in list(shape)[1:]) + ")"


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.environ.get("EFORECAST_CASE", "configuration/cases/default.yaml"), help="YAML case that decides the dummy shapes; default: $EFORECAST_CASE or configuration/cases/default.yaml")
    parser.add_argument("--branch", default=None, help="data branch to build: nwp, lstm, row_all, row_calendar, ...")
    parser.add_argument("--layers", default=None, help='layers of the branch, "name[:size],..."')
    parser.add_argument("--with", dest="extra", action="append", default=[], help='another branch "name=layers", repeatable')
    parser.add_argument("--merge", default=None, help='merge layers, e.g. "cross_attention:64,dense:64" (default concatenate)')
    parser.add_argument("--tag", default=None, help="build a whole template of the case instead")
    parser.add_argument("--method", default="MLP", help="method family whose parameters are used (default MLP)")
    parser.add_argument("--params", default="{}", help="JSON overriding the method parameters")
    parser.add_argument("--layer-params", default="{}", help="JSON: the per-layer params of every layer of --layers")
    parser.add_argument("--shape", default=None, help="comma separated shape (without batch) for --branch, overriding the case")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-backward", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from eforecast.config.config_files import load_document
    from eforecast.config import graph_service
    from eforecast.common_utils.train_utils import fix_convolutional_names
    from eforecast.deep_models.pytorch_2x.builders import build_graph, build_layer, branches_for_merge

    document = load_document(args.config) if Path(args.config).exists() else {}
    if not document:
        print(f"warning: no case at {args.config}; using placeholder shapes")
    info = graph_service.data_shapes(document)
    print("data shapes from the case:", json.dumps(info["shapes"]), f"| head outputs: {info['steps']}")

    if args.tag:
        t0 = time.time()
        report = graph_service.dry_build(document, args.tag, args.method)
        for branch, entry in report["branches"].items():
            print(f"\n[{branch}] input {entry.get('input')}")
            for layer in entry["layers"]:
                print(f"   {layer['index']:2d} {layer['type']:<24} -> {layer['output_shape']}  {layer['params']:>10,d} params")
            if entry.get("error"):
                print(f"   ERROR at layer {entry.get('error_layer')}: {entry['error']}")
        if report.get("merge"):
            print("\n[merge]")
            for layer in report["merge"].get("layers", []):
                extra = f"  tokens {layer['tokens']} d_model {layer['d_model']} query {layer['query_branches']} keys {layer['kv_branches']}" if "tokens" in layer else ""
                print(f"   {layer['index']:2d} {layer['type']:<24} -> {layer['output_shape']}  {layer['params']:>10,d} params{extra}")
            if report["merge"].get("error"):
                print(f"   ERROR: {report['merge']['error']}")
        print(f"\nhead: {report.get('head')}  total params: {report.get('total_params')}  ok={report['ok']}  ({time.time() - t0:.1f}s)")
        return 0 if report["ok"] else 1

    if not (args.branch and args.layers):
        parser.error("either --tag, or --branch with --layers")

    params = dict(graph_service.DEFAULT_LAYER_PARAMS)
    params.update(graph_service._resolve_params(document, args.method, "debug"))
    params.update(json.loads(args.params))
    params["n_out"] = info["steps"]
    params["is_lstm_output"] = False
    layer_params = json.loads(args.layer_params)

    experiment = {args.branch: parse_layers(args.layers)}
    for item in args.extra:
        name, _, spec = item.partition("=")
        experiment[name.strip()] = parse_layers(spec)
    experiment["output"] = parse_layers(args.merge) if args.merge else [("concatenate", []), ("dense", 32)]
    # the template names: ``conv`` becomes conv_2d, or time_distr_conv_2d under short_term
    horizon_type = (document.get("project") or {}).get("horizon_type")
    experiment = fix_convolutional_names(horizon_type, experiment, params.get("conv_dim", 2))
    adapted = branches_for_merge(experiment)

    def dummy(branch):
        base = branch.split("/")[-1]
        shape = [int(s) for s in args.shape.split(",")] if (args.shape and branch == args.branch) else info["shapes"].get(base)
        if shape is None:
            raise SystemExit(f"no shape known for branch {branch}; pass --shape")
        # random inputs: with zeros every weight that multiplies the input would show an empty gradient
        if isinstance(shape, dict):
            return {k: torch.randn(args.batch, *v, device=args.device) for k, v in shape.items()}, {k: [10, *v] for k, v in shape.items()}
        return torch.randn(args.batch, *shape, device=args.device), [10, *shape]

    x, probes = {}, {}
    for branch in experiment:
        if branch != "output":
            x[branch], probes[branch] = dummy(branch)

    # 1. the branch under test, layer by layer
    print(f"\n[{args.branch}] input {fmt(probes[args.branch])}")
    branch_layer_params = [{"type": name, "name": None, "params": layer_params} for name, _ in adapted[args.branch]] if layer_params else None
    built = build_layer(probes[args.branch], adapted[args.branch], args.branch, dict(params), branch_layer_params=branch_layer_params)
    for i, layer in enumerate(built.layers_built):
        shape = getattr(layer, "output_shape", None)
        print(f"   {i:2d} {adapted[args.branch][i][0]:<24} -> {fmt(shape) if shape is not None else '?'}  {count(layer):>10,d} params  {type(layer).__name__}")
    built = built.to(args.device)
    out = built(x[args.branch])
    print(f"   forward: {fmt(out.shape)}  nan={bool(torch.isnan(out).any())}  |mean| {out.abs().mean():.4f}")

    # 2. the graph with the merge and the head
    print("\n[graph]")
    net = build_graph(x, {k: list(v) for k, v in experiment.items()}, params, is_fuzzy=False, device=args.device)
    y = net(x)
    print(f"   output {fmt(y.shape)} (head of {params['n_out']} steps)  nan={bool(torch.isnan(y).any())}  total params {count(net):,d}")
    if not args.no_backward:
        loss = (y ** 2).mean()
        loss.backward()
        missing = [n for n, p in net.named_parameters() if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
        print(f"   backward ok; parameters without gradient: {len(missing)}" + (f" -> {missing[:8]}{' ...' if len(missing) > 8 else ''}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
