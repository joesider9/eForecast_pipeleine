# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

import copy

import numpy as np
import re

import torch
import torch.nn as nn

from eforecast.deep_models.pytorch_2x.layers import layers_func
from eforecast.deep_models.pytorch_2x.layers import _init_weights
from eforecast.deep_models.pytorch_2x.layers import make_act
from eforecast.deep_models.pytorch_2x.global_builders import RBFLayer
from eforecast.deep_models.pytorch_2x.global_builders import apply_activations
from eforecast.deep_models.pytorch_2x.global_builders import act_nan_layer


class FarmMetaBranch(nn.Module):
    """Two-layer MLP that maps physical farm metadata to a conditioning vector.

    Input:  (batch, meta_dim) — normalised physical features (lat, lon, rated, ...)
    Output: (batch, embed_dim)

    Preferred over FarmEmbedding when physical metadata is available because
    the mapping is physically motivated (nearby farms share weights) rather
    than arbitrary per-farm lookup vectors.
    """

    def __init__(self, meta_dim, embed_dim=16):
        super(FarmMetaBranch, self).__init__()
        self.embed_dim = embed_dim
        hidden = max(embed_dim * 2, 32)
        self.net = nn.Sequential(
            nn.Linear(meta_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, farm_meta):
        return self.net(farm_meta)


class FarmEmbedding(nn.Module):
    """Learnable farm embedding — fallback when no physical metadata is available.

    Maps each farm integer ID to a dense vector concatenated with the hidden
    representation before the output layer.
    """

    def __init__(self, n_farms, embed_dim=16):
        super(FarmEmbedding, self).__init__()
        self.n_farms = n_farms
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_farms, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, farm_id):
        return self.embedding(farm_id)

## THESE TWO FUNCS ARE FROM UPDATE AND SHOULD BE CHECKED !!!!!!!!!!!
def _branch_layer_params(params, branch_name):
    experiment_layer_params = params.get('experiment_layer_params') or {}
    experiment_tag = params.get('experiment_tag')
    if not isinstance(experiment_layer_params, dict):
        return None
    if experiment_tag in experiment_layer_params:
        experiment_layer_params = experiment_layer_params[experiment_tag]
    return experiment_layer_params.get(branch_name)


def is_set(value):
    """Whether a layer parameter holds a value: ``None`` and empty text do not.

    A parameter a layer of a graph is given stands in front of the method's own. One without a
    value is a field that was added and left empty, or emptied, in the graph editor: it is
    not set, and the method's value -- or the layer's default -- stands.
    """
    return value is not None and not (isinstance(value, str) and not value.strip())


def _merge_layer_params(params, branch_layer_params, layer_id, layer_name):
    """The parameters of one layer: the method's, with what this layer of the graph sets in front."""
    if not branch_layer_params or layer_id >= len(branch_layer_params):
        return params
    layer_config = branch_layer_params[layer_id]
    if not isinstance(layer_config, dict):
        return params
    configured_type = layer_config.get('type')
    if configured_type is not None and configured_type != layer_name:
        if 'conv' not in str(configured_type) or 'conv' not in str(layer_name):
            print(f'layer {layer_id}: configured type {configured_type!r} does not match the built layer '
                  f'{layer_name!r}; its parameters are ignored')
            return params
    params_temp = dict(params)
    params_temp.update({name: value for name, value in copy.deepcopy(layer_config.get('params') or {}).items()
                        if is_set(value)})
    if layer_config.get('name') is not None:
        params_temp['layer_config_name'] = layer_config['name']
    return params_temp


layers_functions = layers_func()


def get_size(size, layer_id, layers, output_shape):
    """The width of a dense layer from its configured ``size``.

    * a value >= 8 is the width itself
    * a value below 8 is a multiplier of the input features (e.g. 0.5 halves them, 2 doubles them)
    * ``'linear'`` interpolates between the input features and the next layer's size
    * sets / lists were already reduced to one value by the trial
    """
    if isinstance(size, bool):
        return size
    if isinstance(size, set):
        size = list(size)
    if isinstance(size, list):
        size = size[0]
    if size == 'linear':
        if layer_id + 1 >= len(layers):
            raise ValueError('Cannot be linear the last layer')
        size_next = layers[layer_id + 1][1]
        if isinstance(size_next, set):
            size_next = list(size_next)
        if isinstance(size_next, list):
            size_next = size_next[0]
        return int((size_next + np.prod(output_shape)) / 2)
    elif size < 8:
        return int(size * np.prod(output_shape))
    else:
        return size


def get_output_shape(model, image_dim):
    shapes = list(model(torch.rand(10, *image_dim[1:])).data.shape)
    return shapes


class build_layer(nn.Module):
    def __init__(self, shape, layers, name_scope, params, train=True, branch_layer_params=None):
        super(build_layer, self).__init__()
        output_shape = shape
        print(f'Graph of {name_scope} building')
        self.layers_built = nn.ModuleList()
        for layer_id, layer_tuple in enumerate(layers):
            layer_name, size = layer_tuple
            print(f'Input has shape {output_shape}')
            if len(output_shape) == 3 and '3d' in layer_name:
                layer_tuple = ('conv_2d', size)
                layer_name, size = layer_tuple
            if layer_name == 'dense':
                if isinstance(output_shape, dict):
                    output_shape = list(output_shape.values())[0]
                # a flat vector: all features; a token sequence: the features of one token
                size = get_size(size, layer_id, layers, output_shape[1:] if len(output_shape) == 2 else output_shape[-1:])
            if isinstance(size, set):
                size = list(size)
            if isinstance(size, list):
                if len(size) > 0:
                    size = size[0]

            if layer_name not in {'Flatten', 'Dropout', 'Reshape', 'concatenate', 'tokens'}:
                if 'lstm' in layer_name:
                    # the last step only when a Flatten follows; the whole sequence otherwise (tokens, dense)
                    params['is_lstm_output'] = layer_id + 1 < len(layers) and layers[layer_id + 1][0] == 'Flatten'
                params_layer = _merge_layer_params(params, branch_layer_params, layer_id, layer_name)
                if layer_name.startswith('time_distr_conv'):
                    # the last convolution of a time-distributed branch summarises the grid per lag and runs
                    # an LSTM over the lags -> (b, lags, filters); earlier ones keep the grid
                    following = [l[0] for l in layers[layer_id + 1:]]
                    last_conv = not any(('conv' in name or 'cnn' in name) for name in following)
                    params_layer = dict(params_layer, temporal_summary=last_conv)
                layer_ = layers_functions[layer_name](output_shape, params_layer, size, name_scope,
                                                      str(layer_id), train=train)
                self.layers_built.append(layer_)
            elif layer_name == 'Reshape':
                lstm_built = [l for l in self.layers_built if isinstance(l, layers_functions['lstm'])]
                if not lstm_built:
                    raise ValueError(f'{name_scope}: a Reshape layer needs an lstm layer before it')
                lstm_layers = lstm_built[-1]
                lstm_lags = lstm_layers.lstm.input_size
                layer_ = layers_functions[layer_name]([lstm_lags, int(output_shape[1] / lstm_lags)],
                                                      str(layer_id))
                self.layers_built.append(layer_)
            elif layer_name == 'concatenate':
                layer_ = layers_functions[layer_name](output_shape, str(layer_id))
                self.layers_built.append(layer_)
            elif layer_name == 'tokens':
                layer_ = layers_functions[layer_name](output_shape, str(layer_id))
                self.layers_built.append(layer_)
            elif layer_name == 'Dropout':
                layer_ = nn.Dropout(size)
                self.layers_built.append(layer_)
            elif layer_name == 'Flatten':
                layer_ = nn.Flatten()
                self.layers_built.append(layer_)
            else:
                names = [name for name in layers_functions.keys()]
                raise ValueError(f"Unknown layer name {layer_name}. Valid names {names}")
            output_shape = layer_.output_shape if hasattr(layer_, 'output_shape') \
                else get_output_shape(layer_, output_shape)
        self.output_shape = output_shape

    def forward(self, x):
        for layer in self.layers_built:
            x = layer(x)
        return x


class HorizonHead(nn.Module):
    """A small MLP per horizon step on top of the shared trunk, evaluated for all steps at once.

    Input (b, f) -> per step k: act(x W1[k] + b1[k]) W2[k] + b2[k] -> output (b, n_out).
    Gives every step its own features out of the shared representation, which n_out linear
    heads cannot (they are one linear layer of width n_out)."""

    def __init__(self, in_features, n_out, units=32, act_func='gelu'):
        super(HorizonHead, self).__init__()
        self.n_out, self.units = n_out, units
        self.w1 = nn.Parameter(torch.empty(n_out, in_features, units))
        self.b1 = nn.Parameter(torch.zeros(n_out, units))
        self.w2 = nn.Parameter(torch.empty(n_out, units))
        self.b2 = nn.Parameter(torch.zeros(n_out))
        self.act = make_act(act_func)
        for k in range(n_out):
            nn.init.xavier_uniform_(self.w1.data[k])
        nn.init.xavier_uniform_(self.w2.data)
        self.output_shape = [10, n_out]

    def forward(self, x):
        hidden = self.act(torch.einsum('bi,kiu->bku', x, self.w1) + self.b1)
        return (hidden * self.w2).sum(-1) + self.b2


class MoEHead(nn.Module):
    """Mixture-of-experts head gated by the trunk: a learned soft partition of the samples.

    ``experts`` small sub-networks (in -> ``units`` -> n_out, the HorizonHead shape, one per expert) and
    a gate ``softmax(W x / temperature)`` over them; the output is the gate-weighted sum of the experts.
    The fuzzy clusterer partitions the samples by hand-picked variables and trains one model per rule;
    this head learns the partition from the shared representation (which carries the calendar,
    observation and NWP features after the merge) and lets the experts share the trunk, so the
    transitional regimes between rules get a mixture instead of a hard assignment.

    ``topk`` > 0 keeps only the k largest gates per sample (renormalised, sparse routing); 0 is the
    dense mixture.  A load-balancing term (``balance``, 0.01) discourages one expert from taking every
    sample: ``experts * sum_k (mean gate_k)^2`` is 1 for a uniform use and ``experts`` for a collapsed
    gate; it is exposed as ``aux_loss`` after a training forward and added to the loss by the trainer.
    ``last_gate`` (b, experts) and ``usage`` (mean gate per expert) are kept for inspection."""

    def __init__(self, in_features, n_out, experts=4, units=32, act_func='gelu', topk=0, balance=0.01, temperature=1.0):
        super(MoEHead, self).__init__()
        from eforecast.deep_models.pytorch_2x.layers import act_funcs
        self.n_out, self.experts, self.units = int(n_out), int(experts), int(units)
        self.topk = int(topk or 0)
        self.balance = float(balance)
        self.temperature = float(temperature)
        self.act = act_funcs[act_func or 'gelu']
        self.gate = nn.Linear(in_features, self.experts)
        self.w1 = nn.Parameter(torch.empty(self.experts, in_features, self.units))
        self.b1 = nn.Parameter(torch.zeros(self.experts, self.units))
        self.w2 = nn.Parameter(torch.empty(self.experts, self.units, self.n_out))
        self.b2 = nn.Parameter(torch.zeros(self.experts, self.n_out))
        for k in range(self.experts):
            nn.init.xavier_uniform_(self.w1.data[k])
            nn.init.xavier_uniform_(self.w2.data[k])
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, std=0.01)
        self.aux_loss = None
        self.last_gate = None
        self.usage = None
        self.output_shape = [10, self.n_out]

    def forward(self, x):
        gate = torch.softmax(self.gate(x) / self.temperature, dim=-1)                 # (b, experts)
        if self.topk and self.topk < self.experts:
            top, idx = gate.topk(self.topk, dim=-1)
            sparse = torch.zeros_like(gate).scatter(-1, idx, top)
            gate = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-9)
        hidden = self.act(torch.einsum('bi,kiu->bku', x, self.w1) + self.b1)           # (b, experts, units)
        per_expert = torch.einsum('bku,kuo->bko', hidden, self.w2) + self.b2            # (b, experts, n_out)
        out = (gate.unsqueeze(-1) * per_expert).sum(1)
        self.last_gate = gate.detach()
        self.usage = gate.detach().mean(0)
        if self.training and self.balance > 0:
            importance = gate.mean(0)
            self.aux_loss = self.balance * (self.experts * (importance ** 2).sum() - 1.0)
        else:
            self.aux_loss = None
        return out


def adapt_legacy_heads(state_dict, model):
    """Weights saved before the single head: one ``output_k`` linear head per horizon step, each
    ``(1, in)``, stacked into the ``output.dense`` head ``(n_out, in)`` the graph builds today.
    Returns the state dict to load (unchanged when it has no legacy heads)."""
    own = model.state_dict()
    target_w, target_b = 'model_layers_built.output.dense.weight', 'model_layers_built.output.dense.bias'
    if target_w not in own or target_w in state_dict:
        return state_dict
    pattern = re.compile(r'^model_layers_built\.output_(\d+)\.dense\.weight$')
    steps = sorted((int(m.group(1)), k) for k in state_dict for m in [pattern.match(k)] if m)
    n_out = own[target_w].shape[0]
    if len(steps) != n_out:
        return state_dict
    adapted = {k: v for k, v in state_dict.items() if not k.startswith('model_layers_built.output_')}
    adapted[target_w] = torch.cat([state_dict[k].reshape(1, -1) for _, k in steps], dim=0)
    adapted[target_b] = torch.cat([state_dict[k.replace('.weight', '.bias')].reshape(-1) for _, k in steps], dim=0)
    return adapted


class QuantileHead(nn.Module):
    """Quantile head: one linear layer to ``n_out x n_quantiles``, non-crossing by construction.

    The lowest quantile is predicted directly and every next one adds a softplus increment, so the
    quantiles are ordered for every sample and step.  Output ``(batch, n_quantiles, n_out)``; the
    median (the quantile closest to 0.5) is what the deterministic metrics, the checkpoints and
    ``predict`` report, the other quantiles are the interval.  Trained with ``HorizonWeightedPinball``."""

    def __init__(self, in_features, n_out, quantiles):
        super(QuantileHead, self).__init__()
        self.quantiles = sorted(float(q) for q in quantiles)
        if len(self.quantiles) != len(set(self.quantiles)) or not all(0 < q < 1 for q in self.quantiles):
            raise ValueError(f'quantiles must be distinct and in (0, 1): {quantiles}')
        self.n_out, self.n_q = int(n_out), len(self.quantiles)
        self.median_index = int(np.argmin([abs(q - 0.5) for q in self.quantiles]))
        self.linear = nn.Linear(in_features, self.n_out * self.n_q)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.output_shape = [10, self.n_q, self.n_out]

    def forward(self, x):
        raw = self.linear(x).view(x.shape[0], self.n_q, self.n_out)
        if self.n_q == 1:
            return raw
        lowest = raw[:, :1]
        increments = torch.nn.functional.softplus(raw[:, 1:])
        return torch.cat([lowest, lowest + torch.cumsum(increments, dim=1)], dim=1)


class proba_output(nn.Module):
    def __init__(self, shape, quantiles):
        super(proba_output, self).__init__()
        self.shape = shape
        self.quantiles = quantiles
        self.layers = nn.ModuleDict({f"model_output_{i}_q{q}": nn.Linear(shape[1], 1)
                                     for i, q in enumerate(quantiles)})

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.dense.weight.size())}")

    def forward(self, x):
        return torch.cat([self.layers[f"model_output_{i}_q{q}"](x).unsqueeze(1) for i, q in enumerate(self.quantiles)], 1)


ATTENTION_MERGES = {'cross_attention', 'cross_attention_both', 'perceiver'}

#: The parameter that says whether a branch that came whole meets the final merge beside the
#: tags that came in groups. A network trained before 2026-11-15 does not carry it, and is
#: built as it was trained (``network.DeepNetwork.load``).
WHOLE_BRANCHES_JOIN = 'whole_branches_join'


def merge_spec_name(name_scope, model_layers):
    """The spec that merges ``name_scope``: the group merge for ``group/branch`` names, else the output."""
    if '/' in name_scope and 'hidden_layer' in model_layers:
        return 'hidden_layer'
    return 'output'


def branches_for_merge(model_layers):
    """Branch specs adapted to their merge layer.

    A ``cross_attention`` merge attends over positions, so every ``Flatten`` of a branch merged by it
    becomes ``tokens`` (the sequence, grid cells or lags stay separate and the dense layers that
    follow act per token).  A ``concatenate`` or gated merge flattens its branches itself, so those
    keep their ``Flatten``.  ``model_layers`` is not modified."""
    adapted = copy.deepcopy(model_layers)
    for name_scope, layers in model_layers.items():
        if name_scope in {'output', 'hidden_layer'}:
            continue
        spec = model_layers.get(merge_spec_name(name_scope, model_layers)) or []
        if spec and spec[0][0] in ATTENTION_MERGES:
            adapted[name_scope] = [('tokens', []) if layer[0] == 'Flatten' else layer for layer in layers]
            if adapted[name_scope] != list(layers):
                print(f'{name_scope}: Flatten -> tokens (merged by {spec[0][0]})')
    return adapted


def get_shapes(x):
    shapes = dict()
    if isinstance(x, dict):
        for name, inp in sorted(x.items()):
            if 'act' not in name and 'clustering' not in name and not name.startswith('_'):
                shapes[name] = get_shapes(inp)
    else:
        shapes = list(x.shape)
        shapes[0] = 10
    return shapes


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class build_graph(nn.Module):
    def __init__(self, x, model_layers, params, is_fuzzy=False, is_train=True,
                 probabilistic=False, quantiles=None, device='cpu'):
        super(build_graph, self).__init__()
        self.act_nans = 0
        self.is_train = is_train
        self.params = params
        self.device = device
        self.is_fuzzy = is_fuzzy
        self.output_shapes = dict()
        self.model_layers_built = nn.ModuleDict()
        self.non_input_branch_names = set()
        if is_fuzzy:
            self.thres_act = self.params['thres_act']
            # membership gate: 0 at thres_act, 1 at membership_upper, linear in between.
            # membership_upper = 1.0 (default) lets the rule widths receive gradients from every
            # active sample; 1.1 * thres_act reproduces the former near-crisp gate.
            upper = float(self.params.get('membership_upper', 1.0))
            self.thres_act_tf = torch.Tensor([self.thres_act]).squeeze().to(self.device)
            self.thres_act_tf_up = torch.Tensor([upper]).squeeze().to(self.device)
            self.thres_act_tf_ml = torch.Tensor([1 / (upper - self.thres_act)]).squeeze().to(self.device)
            self.model_layers_built['clustering'] = RBFLayer(params)
            self.model_layers_built['act_nan'] = act_nan_layer(params['thres_act'])
            self.non_input_branch_names.add(f'clustering')
            self.non_input_branch_names.add(f'act_nan')
            for rule in sorted(params['rules']):
                self.model_layers_built[f'apply_act_{rule}'] = apply_activations(params['thres_act'])
                self.non_input_branch_names.add(f'apply_act_{rule}')
        shapes = get_shapes(x)
        model_layers = branches_for_merge(model_layers)
        name_scope_list = []
        for name_scope in sorted(shapes.keys()):
            shape = shapes[name_scope]
            if name_scope in {'output'}:
                continue
            layer = model_layers[name_scope]

            params_temp = copy.deepcopy(params)

            prime_layers = build_layer(shape, layer, name_scope, params_temp, train=is_train,
                                       branch_layer_params=_branch_layer_params(params, name_scope))

            self.output_shapes[name_scope] = prime_layers.output_shape
            self.model_layers_built[name_scope] = prime_layers
            name_scope_list.append(name_scope)

        self.connections = []
        group_layer_names = set(['/'.join(name_split.split('/')[:-1]) for name_split in name_scope_list
                                 if '/'.join(name_split.split('/')[:-1]) != ''])
        branch_layer_names = set([name_split.split('/')[-1] for name_split in name_scope_list])
        # A tag whose data came in groups -- the weather by variable or by area -- is joined
        # group by group first. A branch that came whole, the calendar beside it, waits for the
        # final merge. Until 2026-11-15 it was left out there: the groups were joined, and no
        # other branch reached the head. The weights of a network trained that way fit that
        # network only, so it is still built that way (the parameter is then false).
        keep_whole = bool(params.get(WHOLE_BRANCHES_JOIN, True))
        while len(group_layer_names) > 0:
            new_name_scope_list = [name for name in name_scope_list if '/' not in name] if keep_whole else []
            group_connections = dict()
            for group_name in sorted(group_layer_names):
                model_list = []
                shape_list = dict()
                for name_branch in sorted(name_scope_list):
                    if group_name + '/' in name_branch:
                        model_list.append(name_branch)
                        shape_list[name_branch] = self.output_shapes[name_branch]
                group_connections[group_name] = model_list
                layers_name = 'output' if 'hidden_layer' not in model_layers.keys() else 'hidden_layer'
                layers_branch_built = build_layer(shape_list, model_layers[layers_name], group_name,
                                                  params, train=is_train,
                                                  branch_layer_params=_branch_layer_params(params, layers_name))
                self.model_layers_built[group_name] = layers_branch_built
                self.output_shapes[group_name] = layers_branch_built.output_shape
                self.non_input_branch_names.add(group_name)
                new_name_scope_list.append(group_name)
            name_scope_list = copy.deepcopy(new_name_scope_list)
            group_layer_names = set(['/'.join(name_split.split('/')[:-1]) for name_split in name_scope_list
                                     if '/'.join(name_split.split('/')[:-1]) != ''])
            branch_layer_names = set([name_split.split('/')[-1] for name_split in name_scope_list])
            self.connections.append(group_connections)

        if len(branch_layer_names) > 0:
            model_list = []
            shape_list = dict()
            for name_branch in sorted(name_scope_list):
                model_list.append(name_branch)
                shape_list[name_branch] = self.output_shapes[name_branch]
            self.connections.append({'hidden_output': model_list})
            name_scope = 'hidden_output'
            self.non_input_branch_names.add(name_scope)
            layer_output = build_layer(shape_list, model_layers['output'],
                                       name_scope, params, train=is_train,
                                       branch_layer_params=_branch_layer_params(params, 'output'))
            self.model_layers_built[name_scope] = layer_output
            self.output_shapes[name_scope] = layer_output.output_shape

        name_scope = 'output'
        shape = list(self.output_shapes['hidden_output'])
        # CLIP alignment of the branches of the final merge (needs at least two branches and a loss mode)
        if self.params.get('clip_loss_mode') and len(shape_list) > 1:
            self.clip = layers_functions['clip'](shape_list, self.params)
        else:
            self.clip = None
        self.non_input_branch_names.add(name_scope)
        self.quantiles = None
        self.median_index = None
        if probabilistic or params.get('head') == 'quantile':
            # head: 'quantile' = the quantiles of every step (pinball loss); the median stands in for the
            # point forecast wherever the loop and the evaluation expect one
            quantiles = quantiles or params.get('quantiles') or [0.1, 0.5, 0.9]
            self.model_layers_built['output'] = QuantileHead(shape[1], params['n_out'], quantiles)
            self.quantiles = self.model_layers_built['output'].quantiles
            self.median_index = self.model_layers_built['output'].median_index
        elif True:
            # head: 'linear' = one linear layer of width n_out (the same function as one linear head per
            # step); 'mlp' = a small sub-network per horizon step on top of the shared trunk
            if params.get('head') == 'moe':
                # head: 'moe' = a gated mixture of small expert heads, a learned regime partition
                self.model_layers_built['output'] = MoEHead(shape[1], params['n_out'], experts=int(params.get('experts', 4)),
                                                            units=int(params.get('head_units', 32)),
                                                            act_func=params.get('act_func') or 'gelu',
                                                            topk=int(params.get('moe_topk', 0) or 0),
                                                            balance=float(params.get('moe_balance', 0.01)),
                                                            temperature=float(params.get('moe_temperature', 1.0)))
            elif params.get('head', 'linear') == 'mlp' and params['n_out'] > 1:
                self.model_layers_built['output'] = HorizonHead(shape[1], params['n_out'],
                                                                units=int(params.get('head_units', 32)),
                                                                act_func=params.get('act_func') or 'gelu')
            else:
                self.model_layers_built['output'] = layers_functions['dense'](shape, {'act_func': None},
                                                                              params['n_out'], name_scope,
                                                                              'output', train=is_train)

    def forward(self, x, get_activations=False, get_embeddings=False):
        self.act_nans = 0
        activations = None
        outputs = dict()
        if self.is_fuzzy:
            for key in self.model_layers_built.keys():
                if key in {'clustering'}:
                    if 'clustering' not in self.model_layers_built.keys() or 'clustering' not in x.keys():
                        raise ValueError('Since the model is fuzzy, clustering layers should include to model layers or'
                                         'clustering inputs should inlude inside x inputs')
                    activations = self.model_layers_built['clustering'](x['clustering'].to(self.device))
                    if get_activations:
                        return activations
                    self.act_nans = self.model_layers_built['act_nan'](activations)  # raw activations vs thres_act
                    activations = torch.mul(self.thres_act_tf_ml,
                                            torch.sub(torch.clip(activations, self.thres_act_tf, self.thres_act_tf_up),
                                                      self.thres_act_tf))

        for key in self.model_layers_built.keys():
            if key not in self.non_input_branch_names:
                if isinstance(x[key], dict):
                    for key1 in x[key].keys():
                        x[key][key1] = x[key][key1].to(self.device)
                else:
                    x[key] = x[key].to(self.device)
                if self.is_fuzzy and activations is not None and 'rule' in key.split('/')[-1]:
                    rule = key.split('/')[-1]
                    n_rule = int(key.split('_')[-1])
                    out_branch = self.model_layers_built[key](x[key])
                    outputs[key] = self.model_layers_built[f'apply_act_{rule}'](out_branch,
                                                                                       activations[:, n_rule].unsqueeze(1))
                else:
                    outputs[key] = self.model_layers_built[key](x[key])

        embeddings = None
        if len(self.connections) > 0:
            for group_connection in self.connections:
                for name_group, model_list in group_connection.items():
                    if 'apply_act_rule' not in name_group and name_group not in {'clustering', 'output'}:
                        input_brunch = [outputs[key] for key in model_list]
                        if len(input_brunch) == 1:
                            input_brunch = input_brunch[0]
                        if self.clip is not None and self.is_train and name_group == 'hidden_output' \
                                and isinstance(input_brunch, list):
                            embeddings = self.clip(input_brunch)
                        outputs[name_group] = self.model_layers_built[name_group](input_brunch)

        hidden_out = outputs['hidden_output']

        out = self.model_layers_built['output'](hidden_out)
        self.aux_loss = getattr(self.model_layers_built['output'], 'aux_loss', None)   # the MoE balance term (train mode)
        if get_embeddings:
            return out, embeddings
        else:
            return out
