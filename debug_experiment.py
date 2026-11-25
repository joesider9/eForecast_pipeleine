# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Test a layer inside a real experiment of a project, on the project's datasets.

Where ``debug_layer.py`` uses dummy tensors, this script goes through the pipeline's own path: the
case -> ``static_data`` -> the search space of (cluster, method) -> one sampled trial with the
chosen experiment tag -> ``load_data_deep_models`` (the scaled, merged, feature-selected inputs of
the trial) -> one real batch -> ``build_graph`` -> forward, loss, backward.  With ``--epochs N`` it
runs the network's own training loop for N epochs on the same data.

    # the template mlp1 of the MLP family on the global data, one real batch through the graph
    python debug_experiment.py --tag mlp1 --method MLP
    # replace the nwp branch of cnn1 with the layers under development, per-layer params, other batch size
    python debug_experiment.py --tag cnn1 --method CNN --layers "nwp=conv:3,conv:3,tokens,dense:64" \
        --layer-params 'nwp={"pool": 2}' --set batch_size=64
    # a cluster instead of the global model, then five epochs of the real loop on the GPU
    python debug_experiment.py --tag lstm1 --method LSTM --cluster RBF_rule_0 --epochs 5 --gpu 0
    # time 20 training batches (forward + backward) to size the batch for the GPU
    python debug_experiment.py --tag cnn1 --method CNN --batches 20 --gpu 0

Needs the datasets and the clustering of the case (``run_datasets.py --stage all``): the inputs are
read from ``<path_model>/DATA``, the CV masks from ``<cluster_dir>/cv_mask.pickle``.  Everything is
written under ``<cluster_dir>/<method>/debug_<tag>`` (``--path`` to change), never into a trial
folder of a study.  Fuzzy global networks (``is_fuzzy``) are outside this script: debug them with
``run_training.py --prepare-only`` and ``trial_runner.py``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_layers(text):
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


def parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def describe(x, indent="   "):
    """Shapes of what load_data_deep_models returns: frames, arrays, dicts of them."""
    import numpy as np
    import pandas as pd
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {describe(v, indent)}" for k, v in x.items()) + "}"
    if isinstance(x, pd.DataFrame):
        return f"frame {x.shape[0]} x {x.shape[1]}"
    if isinstance(x, np.ndarray):
        return f"array {tuple(x.shape)}"
    if hasattr(x, "shape"):
        return f"tensor {tuple(int(s) for s in x.shape)}"
    return type(x).__name__


def count(module):
    return sum(p.numel() for p in module.parameters())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.environ.get("EFORECAST_CASE", "configuration/cases/default.yaml"), help="YAML case whose datasets, clusters and families are used; default: $EFORECAST_CASE or configuration/cases/default.yaml")
    parser.add_argument("--tag", required=True, help="experiment tag of the case (must be listed in the family's experiment_tag)")
    parser.add_argument("--method", required=True, help="method family: CNN, LSTM, MLP, TRANSFORMER, ...")
    parser.add_argument("--cluster", default="global", help="global (default) or a cluster name of clusters.pickle")
    parser.add_argument("--layers", action="append", default=[], help='replace a branch: "branch=name[:size],..." (repeatable)')
    parser.add_argument("--layer-params", action="append", default=[], help='per-layer params of a replaced branch: "branch=JSON" (same dict for every layer)')
    parser.add_argument("--set", action="append", default=[], help='override a trial parameter: key=value (JSON value), e.g. batch_size=64')
    parser.add_argument("--path", default=None, help="folder for the debug model (default <cluster_dir>/<method>/debug_<tag>)")
    parser.add_argument("--gpu", default="cpu", help="GPU id or cpu (default)")
    parser.add_argument("--batches", type=int, default=1, help="training batches to time in the one-batch mode")
    parser.add_argument("--epochs", type=int, default=0, help="run the real training loop for N epochs instead")
    parser.add_argument("--seed", type=int, default=1, help="seed of the trial sampling and the weights")
    args = parser.parse_args(argv)

    import joblib
    import numpy as np
    import optuna
    import torch
    from eforecast.init.initialize import initializer
    from eforecast.training.command_line.optuna_search import suggest_params
    from eforecast.training.command_line.search_space import SearchSpace

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    static_data = initializer(args.config)
    clusters = {"global": os.path.join(static_data["path_model"], "global")}
    clusters_file = os.path.join(static_data["path_model"], "clusters.pickle")
    if os.path.exists(clusters_file):
        clusters.update({str(k): str(v) for k, v in joblib.load(clusters_file).items()})
    if args.cluster not in clusters:
        raise SystemExit(f"unknown cluster {args.cluster!r}; known: {', '.join(clusters)}")
    cluster_dir = clusters[args.cluster]
    if args.method not in static_data:
        raise SystemExit(f"{args.method} is not a method family of the case")
    if args.tag not in static_data["experiments"]:
        raise SystemExit(f"experiment {args.tag!r} is not defined; known: {', '.join(static_data['experiments'])}")
    tags = static_data[args.method]["experiment_tag"]
    if args.tag not in (tags if isinstance(tags, (set, list, tuple)) else {tags}):
        print(f"note: {args.tag} is not in {args.method}.experiment_tag {sorted(tags) if isinstance(tags, set) else tags}; using it anyway")
        static_data[args.method]["experiment_tag"] = {args.tag}

    # ---- one trial of the (cluster, method) search space, with the requested tag
    objective = SearchSpace(static_data, args.cluster, cluster_dir, args.method)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=args.seed))
    if "experiment_tag" in objective.space:
        study.enqueue_trial({"experiment_tag": args.tag})
    trial = study.ask()
    params, structure, tag = suggest_params(trial, objective)
    if tag != args.tag:
        tag = args.tag
    experiment_params = objective.init_experiment_params(0, params, tag, structure)
    experiment_params["experiment"] = objective.select_structure(structure, tag, static_data["experiments"][tag])
    for item in args.layers:
        branch, _, spec = item.partition("=")
        experiment_params["experiment"][branch.strip()] = parse_layers(spec)
        if branch.strip() not in experiment_params["data_types"]:
            print(f"warning: branch {branch!r} is not an input of {tag}; its data types are copied from row_all")
            experiment_params["data_types"][branch.strip()] = dict(experiment_params["data_types"].get("row_all", {}))
    for item in args.layer_params:
        branch, _, spec = item.partition("=")
        layer_params = json.loads(spec)
        experiment_params.setdefault("experiment_layer_params", {})[branch.strip()] = [
            {"type": name, "name": None, "params": dict(layer_params)} for name, _ in experiment_params["experiment"][branch.strip()]]
    for item in args.set:
        key, _, value = item.partition("=")
        experiment_params[key.strip()] = parse_value(value)
    experiment_params["inputs"] = [key for key in experiment_params["experiment"] if "output" not in key]
    path_weights = args.path or os.path.join(cluster_dir, args.method, f"debug_{tag}")
    os.makedirs(path_weights, exist_ok=True)
    print(f"trial parameters ({args.method}, {args.cluster}, {tag}): "
          + ", ".join(f"{k}={v}" for k, v in experiment_params.items() if k not in {"experiment", "data_types", "cluster", "experiment_layer_params"} and not isinstance(v, (dict, list))))
    print("experiment:", json.dumps(experiment_params["experiment"]))
    print("data types:", json.dumps(experiment_params["data_types"]))

    from eforecast.deep_models.pytorch_2x.network import DeepNetwork
    model = DeepNetwork(static_data, path_weights, copy.deepcopy(experiment_params), train=True, refit=True)
    cv_masks = joblib.load(os.path.join(cluster_dir, "cv_mask.pickle"))
    gpu = "cpu" if args.gpu == "cpu" else int(args.gpu)

    if args.epochs > 0:                                   # the real loop
        model.epochs = model.params["max_iterations"] = args.epochs
        t0 = time.time()
        model.fit(cv_masks, gpu_id=gpu)
        print(f"\n{args.epochs} epoch(s) in {time.time() - t0:.0f}s: best mae val {model.best_mae_val:.4f}, test {model.best_mae_test:.4f}; "
              f"results.csv and weights in {path_weights}")
        return 0

    # ---- the data of the trial, as fit() loads it
    from eforecast.common_utils.dataset_utils import get_slice_for_nets, load_data_deep_models
    from eforecast.common_utils.train_utils import fix_convolutional_names
    from eforecast.deep_models.pytorch_2x.builders import build_graph
    from eforecast.deep_models.pytorch_2x.trainer import feed_dataset

    device = torch.device("cpu" if gpu == "cpu" or not torch.cuda.is_available() else f"cuda:{gpu}")
    model.params["device"] = device
    model.params["experiment"] = fix_convolutional_names(static_data["horizon_type"], model.params["experiment"], model.conv_dim)
    model.params["kernels"] = static_data["kernels"]
    t0 = time.time()
    X, y, metadata, model_layers, model.params = load_data_deep_models(model.data_feeder, model.data_types, model.model_layers,
                                                                        model.params, model.cluster, True, False, True)
    print(f"\nload_data_deep_models in {time.time() - t0:.1f}s")
    for tag_name, value in X.items():
        print(f"   {tag_name:<14} {describe(value)}   dates {len(metadata[tag_name]['dates'])}")
    print(f"   {'target':<14} frame {y.shape[0]} x {y.shape[1]}  columns {list(y.columns)[:4]}{' ...' if y.shape[1] > 4 else ''}")
    model.params["n_out"] = y.shape[1]
    X_train, y_train, mask = get_slice_for_nets(copy.deepcopy(X), metadata, dates=cv_masks[0], y=y)
    print(f"   train slice: {len(mask)} dates (cv mask 0 of {len(cv_masks)}), batch size {model.batch_size}")
    train_dataset = feed_dataset(X_train, y_train, device, batch_size=int(model.batch_size), shuffle=True)
    x_sample, y_sample = next(iter(train_dataset))
    print("   one batch:", describe(x_sample), "->", describe(y_sample))

    # ---- the graph on the real batch
    torch.manual_seed(args.seed)
    net = build_graph(x_sample, model_layers, model.params, is_fuzzy=False, device=device).to(device)
    print("\ngraph")
    for branch, block in net.model_layers_built.items():
        if not hasattr(block, "layers_built"):
            print(f"   [{branch}] {type(block).__name__}  {count(block):,d} params")
            continue
        spec = model_layers.get(branch if branch != "hidden_output" else "output", [])
        print(f"   [{branch}]")
        for i, layer in enumerate(block.layers_built):
            shape = getattr(layer, "output_shape", None)
            name = spec[i][0] if i < len(spec) else type(layer).__name__
            print(f"      {i:2d} {name:<24} -> {list(int(s) for s in shape)[1:] if shape is not None else '?'}  {count(layer):>10,d} params")
    print(f"   total {count(net):,d} parameters")

    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    times = []
    for b, (xb, yb) in enumerate(train_dataset):
        if b >= args.batches:
            break
        t0 = time.time()
        optimizer.zero_grad()
        out = net(xb)
        if b == 0:
            print(f"\nforward: output {tuple(int(s) for s in out.shape)}  target {tuple(int(s) for s in yb.shape)}  nan={bool(torch.isnan(out).any())}")
        point = out[:, net.median_index] if out.dim() == 3 else out          # quantile head: the median
        loss = torch.mean((point - yb) ** 2)
        loss.backward()
        if b == 0:
            missing = [n for n, p in net.named_parameters() if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
            print(f"backward: loss {float(loss):.5f}; parameters without gradient: {len(missing)}" + (f" -> {missing[:6]}" if missing else ""))
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    if len(times) > 1:
        print(f"{len(times)} training batches: {np.mean(times[1:]) * 1000:.0f} ms per batch (first {times[0] * 1000:.0f} ms)")
    if device.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    print(f"\nok. Next: --epochs 3 for the real loop, or run_training.py --prepare-only and trial_runner.py for a study trial.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
