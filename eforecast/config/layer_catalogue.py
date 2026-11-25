# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""What every layer of the network builder is, in words the graph editor can show.

The names are the keys of ``eforecast.deep_models.pytorch_2x.layers.layers_func`` plus the
configuration aliases the builder resolves (``conv`` -> ``conv_2d`` or ``time_distr_conv_2d``).
A test keeps this table and the registry in step without importing torch here.

kinds of tensors between layers
    flat      (batch, features)                       a vector per sample
    sequence  (batch, steps, features)                LSTM / transformer outputs, tokens
    grid      (batch, channels, lat, lon)             a feature map
    nwp       (batch, lags, variables, lat, lon)      the NWP tensor of a site
    lstm      dict of past/future observations, calendar and covariates (the ``lstm`` data tag)
"""
from __future__ import annotations

from typing import Any

BRANCHES: dict[str, str] = {
    "nwp": "NWP fields around the site: (lags, variables, lat, lon). Convolutions, the grid transformer or a time-distributed stack.",
    "lstm": "Past observations and calendar plus future covariates as sequences: LSTM, LSTM with attention, transformers.",
    "row_all": "Every row feature: observation lags, calendar, NWP statistics. Dense layers.",
    "row_obs": "Observation lags only. Dense layers.",
    "row_calendar": "Calendar features (hour, month, ...). Dense layers; the usual query of a cross-attention merge.",
    "row_nwp": "NWP statistics as a vector. Dense layers.",
    "row_obs_calendar": "Observation lags and calendar. Dense layers.",
    "row_obs_nwp": "Observation lags and NWP statistics. Dense layers.",
    "images": "Satellite images (rare). Pretrained vision networks or the U-Net.",
}

MERGES: tuple[str, ...] = ("cross_attention", "cross_attention_both", "perceiver", "gated_fusion", "gated_fusion_both", "concatenate")

# name -> catalogue entry
LAYERS: dict[str, dict[str, Any]] = {
    "conv": {"label": "Convolution",
             "kind": "conv",
             "inputs": ["nwp", "grid"],
             "output": "nwp",
             "size": "kernel size (made odd); filters come from the method parameter `filters` x (layer index + 1)",
             "size_type": "int",
             "params": [{"name": "filters", "type": "int", "default": None, "help": "channels of the convolution; the method's `filters` by default"},
                        {"name": "filters_grow", "type": "bool", "default": True, "help": "filters x (layer index + 1); false keeps `filters`"},
                        {"name": "kernel_time", "type": "int", "default": 3, "help": "kernel over the lags"},
                        {"name": "pool", "type": "int", "default": None, "help": "average pooling factor; default 2 while the grid is at least 8 wide"},
                        {"name": "residual", "type": "bool", "default": True, "help": "add the input when the channel count is unchanged and there is no pooling"},
                        {"name": "batch_norm", "type": "bool", "default": False, "help": "GroupNorm after the convolution"},
                        {"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}],
             "description": "One convolution block shared over the NWP lags: conv (same padding) -> GroupNorm -> activation -> optional residual -> average pooling. Under short_term the builder makes it time-distributed: the last convolution of the branch pools the grid and runs an LSTM over the lags, giving one vector per lead time."},
    "conv_3d": {"label": "Convolution 3-D",
                "kind": "conv",
                "inputs": ["nwp"],
                "output": "nwp",
                "size": "kernel size over lat/lon (made odd)",
                "size_type": "int",
                "params": [{"name": "kernel_time", "type": "int", "default": 3, "help": "kernel over the lags"},
                            {"name": "pool", "type": "int", "default": None, "help": "average pooling factor; default 2 while the grid is at least 8 wide"},
                            {"name": "filters", "type": "int", "default": None, "help": "channels of the convolution; the method's `filters` by default"},
                            {"name": "filters_grow", "type": "bool", "default": True, "help": "filters x (layer index + 1); false keeps `filters`"},
                            {"name": "residual", "type": "bool", "default": True, "help": "add the input when the channel count is unchanged and there is no pooling"},
                            {"name": "batch_norm", "type": "bool", "default": False, "help": "GroupNorm after the convolution"},
                            {"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}],
                "description": "The same block with a 3-D kernel over (lags, lat, lon): mixes lead times and space at once."},
    "time_distr_conv": {"label": "Time-distributed convolution",
                        "kind": "conv",
                        "inputs": ["nwp"],
                        "output": "sequence",
                        "size": "kernel size",
                        "size_type": "int",
                        "params": [{"name": "filters", "type": "int", "default": None, "help": "channels of the convolution; the method's `filters` by default"},
                                    {"name": "pool", "type": "int", "default": None, "help": "average pooling factor; default 2 while the grid is at least 8 wide"},
                                    {"name": "residual", "type": "bool", "default": True, "help": "add the input when the channel count is unchanged and there is no pooling"},
                                    {"name": "batch_norm", "type": "bool", "default": False, "help": "GroupNorm after the convolution"},
                                    {"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}], "description": "The shared convolution block over the lags; the last one summarises the grid per lag and runs an LSTM over the lags -> (lags, filters)."},
    "time_distr_conv_3d": {"label": "Time-distributed convolution 3-D",
                           "kind": "conv", "inputs": ["nwp"],
                           "output": "sequence",
                           "size": "kernel size",
                           "size_type": "int",
                           "params": [{"name": "kernel_time", "type": "int", "default": 3, "help": "kernel over the lags"},
                                        {"name": "pool", "type": "int", "default": None, "help": "average pooling factor; default 2 while the grid is at least 8 wide"},
                                        {"name": "filters", "type": "int", "default": None, "help": "channels of the convolution; the method's `filters` by default"},
                                        {"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}],
                           "description": "Alias of the 3-D block with the temporal summary at the end of the branch."},
    "cnnT": {"label": "Convolution (legacy twin)",
             "kind": "conv",
             "inputs": ["nwp", "grid"],
             "output": "nwp",
             "size": "kernel size",
             "size_type": "int",
             "params": [{"name": "filters", "type": "int", "default": None, "help": "channels of the convolution; the method's `filters` by default"},
                        {"name": "pool", "type": "int", "default": None, "help": "average pooling factor; default 2 while the grid is at least 8 wide"},
                        {"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}],
             "description": "Kept for old templates: now the same block as `conv`."},
    "cnn_att": {"label": "Convolutional attention",
                "kind": "attention",
                "inputs": ["nwp", "grid"],
                "output": "nwp",
                "size": "the attention type: cbam | cbam1 | att (spatial transformer) | geodatt",
                "size_type": "str",
                "params": [{"name": "attention", "type": "str", "default": "cbam", "help": "cbam | cbam1 | att (spatial transformer) | geodatt; the size may name it instead"},
                        {"name": "filters", "type": "int", "default": None, "help": "channels of the convolution; the method's `filters` by default"},
                        {"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}],
                "description": "Channel and spatial attention on the NWP fields (CBAM), a spatial-transformer mask (att) or the geo dual attention (geodatt). Keeps the shape of its input."},
    "grid_transformer": {"label": "Grid transformer",
                         "kind": "attention",
                         "inputs": ["nwp", "grid"],
                         "output": "sequence",
                         "size": "d_model when >= 8 (else 64)",
                         "size_type": "int",
                         "params": [{"name": "patch", "type": "int", "default": 1, "help": "cells per token side"},
                                    {"name": "depth", "type": "int", "default": 2, "help": "encoder layers"},
                                    {"name": "heads", "type": "int", "default": 4, "help": "attention heads"},
                                    {"name": "attention_dropout", "type": "float", "default": 0.1, "help": ""}],
                         "description": "One token per grid cell (or patch) with the lags and variables as features, a learned positional embedding and a small transformer encoder: attention over the map, trained from scratch. Output: one token per cell."},
    "afno": {"label": "AFNO (Fourier operator on the grid)",
             "kind": "attention",
             "inputs": ["nwp", "grid"],
             "output": "grid",
             "size": "channels when >= 8 (else 64)",
             "size_type": "int",
             "params": [{"name": "depth", "type": "int", "default": 2, "help": "blocks"},
                        {"name": "num_blocks", "type": "int", "default": 4, "help": "channel blocks of the block-diagonal Fourier MLP"},
                        {"name": "patch", "type": "int", "default": 1, "help": "cells per token side"},
                        {"name": "sparsity_threshold", "type": "float", "default": 0.01, "help": "soft-shrink lambda"},
                        {"name": "hard_thresholding_fraction", "type": "float", "default": 1.0, "help": "fraction of the low modes kept"},
                        {"name": "mlp_ratio", "type": "float", "default": 2, "help": ""},
                        {"name": "dropout", "type": "float", "default": None, "help": "channel MLP dropout; attention_dropout by default"},
                        {"name": "output", "type": "str", "default": "grid", "help": "grid (feature map) | tokens"}],
             "description": "Adaptive Fourier neural operator (the mixer of FourCastNet): every cell is a token with the lags and variables as features; each block mixes all cells at once in Fourier space (2-D FFT, block-diagonal complex MLP per frequency, soft-shrink, inverse FFT) and follows with a channel MLP. Global spatial mixing on a 17x17 field for few parameters. Output a feature map (channels, lat, lon): follow it with tokens (attention merge), a Flatten, or an explicit conv_2d; a plain conv after it would be made time-distributed under short_term and read the channels as lags."},
    "lstm": {"label": "LSTM", "kind": "recurrent", "inputs": ["lstm", "sequence"], "output": "sequence", "size": "hidden = size x input width", "size_type": "float",
             "params": [{"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}], "description": "Recurrent encoder of the past observations; with `use_nwp` the future covariates are encoded too. Output: one vector per step (the last one when a Flatten follows)."},
    "lstm_att": {"label": "LSTM encoder-decoder with attention", "kind": "recurrent", "inputs": ["lstm"], "output": "sequence", "size": "hidden = size x input width", "size_type": "float",
                 "params": [{"name": "heads", "type": "int", "default": 4, "help": "attention heads"},
                        {"name": "attention_dropout", "type": "float", "default": 0.1, "help": ""},
                        {"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}],
                 "description": "Encodes the past with one LSTM, decodes the future covariates with another initialised from the encoder state, and lets every future step attend over the encoded past steps. Output: one vector per future step."},
    "timexer": {"label": "TimeXer encoder", "kind": "attention", "inputs": ["lstm", "sequence"], "output": "sequence", "size": "d_model when >= 8 (else 64)", "size_type": "int",
                "params": [{"name": "patch", "type": "int", "default": 4, "help": "past steps per endogenous token"},
                        {"name": "depth", "type": "int", "default": 2, "help": "encoder blocks"},
                        {"name": "heads", "type": "int", "default": 4, "help": "attention heads"},
                        {"name": "attention_dropout", "type": "float", "default": 0.1, "help": ""},
                        {"name": "output", "type": "str", "default": "tokens", "help": "tokens | global | flat"},
                        {"name": "use_nwp", "type": "bool", "default": True, "help": "encode the future covariates too; the method's use_nwp by default"}],
                "description": "TimeXer (NeurIPS 2024): the past observations are patched into endogenous tokens plus a learned global token; every exogenous series (past calendar columns, and with use_nwp each future covariate and future calendar column) is one variate token. Each block: self-attention among the endogenous tokens, cross-attention of the global token over the exogenous tokens, feed-forward. Output: the endogenous tokens (n_patches + 1, d_model) for a Flatten, a dense per token or an attention merge."},
    "tide": {"label": "TiDE dense encoder", "kind": "dense", "inputs": ["lstm", "sequence"], "output": "sequence", "size": "features per future step", "size_type": "int",
             "params": [{"name": "hidden", "type": "int", "default": 128, "help": "units of the residual blocks"},
                        {"name": "depth", "type": "int", "default": 2, "help": "encoder and decoder blocks"},
                        {"name": "cov_dim", "type": "int", "default": 8, "help": "projected covariate features per step"},
                        {"name": "decoder_dim", "type": "int", "default": 16, "help": "decoded features per future step"},
                        {"name": "dropout", "type": "float", "default": 0.1, "help": ""},
                        {"name": "skip", "type": "bool", "default": True, "help": "linear history-to-horizon residual as an extra feature"},
                        {"name": "output", "type": "str", "default": "sequence", "help": "sequence | flat"},
                        {"name": "use_nwp", "type": "bool", "default": True, "help": "encode the future covariates too; the method's use_nwp by default"}],
             "description": "TiDE (2023), MLP only: the past observations are the history, the past calendar and (with use_nwp) the future covariates and calendar are projected per step; a dense encoder over everything flattened, a dense decoder to one vector per future step, and a temporal decoder that joins each step's vector with its projected covariates. Output: one vector of size (+1 skip) per future step. Cheap: a minute per trial, a good cluster model."},
    "tsmixer": {"label": "TSMixer", "kind": "dense", "inputs": ["lstm", "sequence"], "output": "sequence", "size": "features per future step", "size_type": "int",
                "params": [{"name": "hidden", "type": "int", "default": 64, "help": "features per block of the past and the future"},
                        {"name": "depth", "type": "int", "default": 2, "help": "mixer blocks"},
                        {"name": "ff_ratio", "type": "float", "default": 2, "help": "hidden width of the feature mixing, x channels"},
                        {"name": "dropout", "type": "float", "default": 0.1, "help": ""},
                        {"name": "output", "type": "str", "default": "sequence", "help": "sequence | flat"},
                        {"name": "use_nwp", "type": "bool", "default": True, "help": "encode the future covariates too; the method's use_nwp by default"}],
                "description": "TSMixer (2023), MLPs only: the past block is aligned to the horizon by a linear layer over its steps and projected; with use_nwp the future covariates are projected per step and concatenated; mixer blocks then alternate a linear layer over the steps (time mixing) and an MLP over the features (feature mixing), each with a residual. Output: one vector of size per future step. Cheap, a cluster model next to TiDE."},
    "bi_lstm": {"label": "Bidirectional LSTM pair", "kind": "recurrent", "inputs": ["lstm", "nwp"], "output": "sequence", "size": "hidden factor", "size_type": "float", "params": [{"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}],
                "description": "Two LSTMs (past and future) whose last steps feed a bidirectional LSTM. On an NWP tensor the grid is flattened per lag."},
    "transformer": {"label": "Time-series transformer", "kind": "attention", "inputs": ["lstm"], "output": "sequence", "size": "d_model factor", "size_type": "float",
                    "params": [{"name": "model_name", "type": "str", "default": None, "help": "model of the time series library"},
                        {"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}],
                    "description": "A model of the time series library (Informer, Autoformer, ...) over the past sequence."},
    "bi_transformer": {"label": "Transformer pair", "kind": "attention", "inputs": ["lstm"], "output": "flat", "size": "d_model factor", "size_type": "float",
                       "params": [{"name": "model_name", "type": "str", "default": None, "help": "model of the time series library"},
                        {"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}], "description": "Two time-series transformers (past and future) joined by a linear layer."},
    "bi_spacetimeformer": {"label": "Spacetimeformer pair", "kind": "attention", "inputs": ["lstm", "nwp"], "output": "sequence", "size": "unused", "size_type": None, "params": [{"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}],
                           "description": "The spacetimeformer over past and future; experimental."},
    "dense": {"label": "Dense", "kind": "dense", "inputs": ["flat", "sequence"], "output": "same rank", "size": "units (>= 8), or a multiplier of the input features (< 8), or 'linear'", "size_type": "float",
              "params": [{"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}], "description": "A linear layer with the method's activation. On a token sequence it acts per token."},
    "Flatten": {"label": "Flatten", "kind": "shape", "inputs": ["any"], "output": "flat", "size": "-", "size_type": None, "params": [],
                "description": "Collapses everything but the batch into one vector. Before a cross-attention merge the builder turns it into `tokens` so positions survive."},
    "tokens": {"label": "Tokens", "kind": "shape", "inputs": ["any"], "output": "sequence", "size": "-", "size_type": None, "params": [],
               "description": "Keeps positions as tokens for an attention merge: sequences stay, a grid becomes one token per cell, the NWP tensor one token per cell with lags and variables as features. Put it after the layers that produce positions (convolutions, grid transformer, LSTM) and before the dense layers that should act per token; under a cross-attention merge the builder turns a Flatten into tokens by itself, and under concatenate it is not needed."},
    "Dropout": {"label": "Dropout", "kind": "shape", "inputs": ["any"], "output": "same", "size": "drop probability", "size_type": "float", "params": [], "description": "Dropout with the given probability."},
    "Reshape": {"label": "Reshape", "kind": "shape", "inputs": ["flat"], "output": "sequence", "size": "-", "size_type": None, "params": [], "description": "Back to (lags, features) after an LSTM; rarely needed."},
    "cross_attention": {"label": "Cross attention (merge)", "kind": "merge", "inputs": ["branches"], "output": "flat", "size": "d_model when >= 8 (else 64)", "size_type": "int",
                        "params": [{"name": "heads", "type": "int", "default": 4, "help": "attention heads"},
                        {"name": "attention_dropout", "type": "float", "default": 0.1, "help": ""},
                        {"name": "attention_query", "type": "list", "default": None, "help": "branch names used as queries (default: the flat branches)"},
                        {"name": "d_model", "type": "int", "default": None, "help": "token width; the size by default"}],
                        "description": "The flat branches (calendar, observations) query the token branches (NWP cells, past steps): softmax over positions, residual, feed-forward. With a single branch a learned query pools it; with only flat branches it weights the modalities."},
    "cross_attention_both": {"label": "Cross attention, both ways (merge)", "kind": "merge", "inputs": ["branches"], "output": "flat", "size": "d_model", "size_type": "int", "params": [{"name": "heads", "type": "int", "default": 4, "help": "attention heads"},
                        {"name": "attention_dropout", "type": "float", "default": 0.1, "help": ""},
                        {"name": "attention_query", "type": "list", "default": None, "help": "branch names used as queries"},
                        {"name": "d_model", "type": "int", "default": None, "help": "token width; the size by default"}],
                             "description": "Cross attention plus a pooled summary of the token branches, concatenated."},
    "perceiver": {"label": "Perceiver with horizon queries (merge)", "kind": "merge", "inputs": ["branches"], "output": "flat", "size": "d_model when >= 8 (else 64)", "size_type": "int",
                  "params": [{"name": "latents", "type": "int", "default": 16, "help": "learned latent tokens"},
                        {"name": "depth", "type": "int", "default": 2, "help": "rounds of cross + self attention"},
                        {"name": "heads", "type": "int", "default": 4, "help": "attention heads"},
                        {"name": "attention_dropout", "type": "float", "default": 0.1, "help": ""},
                        {"name": "horizon_queries", "type": "int", "default": None, "help": "decoder queries (default: one per horizon step)"},
                        {"name": "output", "type": "str", "default": "flat", "help": "flat (queries x d_model) | steps (one scalar per query)"}],
                  "description": "All branches become tokens (grid cells, past and future steps, one token per flat branch). A small array of learned latents attends over every token, then among itself, for a few rounds; finally one learned query per horizon step attends over the latents, so each lead time reads its own mix of NWP cells, past steps and calendar. Output: the step tokens flattened for dense layers and the head, or a direct per-step readout."},
    "gated_fusion": {"label": "Gated fusion (merge)", "kind": "merge", "inputs": ["branches"], "output": "flat", "size": "-", "size_type": None, "params": [],
                     "description": "The former cross_attention: a multiplicative gate on a projection of one branch conditioned on the other. No positions, no attention map."},
    "gated_fusion_both": {"label": "Gated fusion, both ways (merge)", "kind": "merge", "inputs": ["branches"], "output": "flat", "size": "-", "size_type": None, "params": [], "description": "Two gates, one per direction."},
    "concatenate": {"label": "Concatenate (merge)", "kind": "merge", "inputs": ["branches"], "output": "flat", "size": "-", "size_type": None, "params": [],
                    "description": "Flattens every branch and concatenates the vectors."},
    "clip": {"label": "CLIP alignment", "kind": "merge", "inputs": ["branches"], "output": "-", "size": "-", "size_type": None, "params": [{"name": "clip_dim", "type": "int", "default": 128, "help": "width of the shared embedding"}], "description": "Internal: built by the graph when `clip_loss_mode` is set."},
    "vit_net": {"label": "Vision transformer (timm / HF)", "kind": "image", "inputs": ["images", "nwp"], "output": "sequence", "size": "-", "size_type": None,
                "params": [{"name": "model_name", "type": "str", "default": None, "help": "timm or Hugging Face model"},
                        {"name": "pretrained", "type": "str", "default": "tune", "help": "tune | finetune (frozen) | none"},
                        {"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}],
                "description": "A pretrained vision model; the input is resized to what the model expects."},
    "timm_net": {"label": "timm network", "kind": "image", "inputs": ["images", "nwp"], "output": "sequence", "size": "-", "size_type": None, "params": [{"name": "model_name", "type": "str", "default": None, "help": "timm or Hugging Face model"},
                        {"name": "pretrained", "type": "str", "default": "tune", "help": "tune | finetune (frozen) | none"}], "description": "Alias of the vision transformer layer."},
    "time_distr_vit_net": {"label": "Time-distributed vision transformer", "kind": "image", "inputs": ["images"], "output": "flat", "size": "-", "size_type": None, "params": [{"name": "model_name", "type": "str", "default": None, "help": "timm or Hugging Face model"},
                        {"name": "pretrained", "type": "str", "default": "tune", "help": "tune | finetune (frozen) | none"}], "description": "One vision model per lag and an LSTM over the lags."},
    "unet": {"label": "U-Net", "kind": "image", "inputs": ["images", "grid"], "output": "flat", "size": "-", "size_type": None,
             "params": [{"name": "unet_type", "type": "str", "default": "Unet", "help": "architecture of segmentation_models_pytorch"},
                        {"name": "encoder_name", "type": "str", "default": "resnet18", "help": ""},
                        {"name": "pretrained", "type": "str", "default": "tune", "help": "tune | finetune (frozen) | none"},
                        {"name": "act_func", "type": "str", "default": None, "help": "activation of this layer; the method's act_func by default"}],
             "description": "A segmentation U-Net whose output is flattened."},
}

ALIASES: dict[str, str] = {"conv": "conv_2d", "time_distr_conv": "time_distr_conv_2d", "time_distr_conv_3d": "time_distr_conv_3d",
                           "conv_3d": "conv_3d"}


def layer_params(name: str) -> list[dict[str, Any]]:
    """Every parameter the editor may offer for ``name``, ``size`` first when the layer takes one.

    ``size`` is kept first because it is the one parameter the builder passes positionally
    (``layers_functions[type](shape, params, size, ...)``); the rest reach the layer inside ``params``.
    """
    entry = LAYERS.get(name)
    if entry is None:
        return []
    params: list[dict[str, Any]] = []
    if entry.get("size_type") is not None:
        params.append({"name": "size", "type": entry["size_type"], "default": None, "help": entry["size"]})
    params.extend(entry.get("params") or [])
    return params


def param_names(name: str) -> list[str]:
    """The predefined parameter names of a layer type, for validation and the editor's dropdown."""
    return [param["name"] for param in layer_params(name)]


def catalogue() -> dict[str, Any]:
    layers = [{"name": name, **entry, "params": layer_params(name)} for name, entry in LAYERS.items()]
    return {"layers": layers, "branches": [{"name": n, "description": d} for n, d in BRANCHES.items()],
            "merges": list(MERGES), "aliases": ALIASES}
