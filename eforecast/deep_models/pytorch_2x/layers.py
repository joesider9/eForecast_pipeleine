# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

import numpy as np
import torch
import torch.nn as nn
from eforecast.deep_models.pytorch_2x.time_series_library.models.get_model import init_ts_model
from eforecast.deep_models.pytorch_2x.conv_att.cbam import CBAM
from eforecast.deep_models.pytorch_2x.conv_att.cbam_ver2 import CBAM_ver2
from eforecast.deep_models.pytorch_2x.conv_att.GeoDANet import GeoDualAttentionNetwork

import torch.nn.functional as F
from einops import rearrange
from einops import einsum

# transformers, timm, torchvision and the spacetimeformer model are imported by the three layers
# that use them (vit_net, unet, bi_spacetimeformer), not here. Every trial is a process that
# imports this module, and those four took 5.6 of its 7.3 seconds for graphs that are mostly
# dense and convolutional.

ACTIVATIONS = {'elu': nn.ELU, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh,
               'lrelu': nn.LeakyReLU, 'prelu': nn.PReLU}


def make_act(name):
    """A fresh activation module for one layer.

    PReLU carries a learnable slope: a single shared instance (the former module-level ModuleDict)
    tied every layer of every model in the process to the same parameter."""
    if name is None:
        return None
    if name not in ACTIVATIONS:
        raise ValueError(f'unknown activation {name!r}; one of {sorted(ACTIVATIONS)}')
    return ACTIVATIONS[name]()


class _ActivationFactory:
    """``act_funcs[name]`` keeps working at every call site, returning a new module each time."""

    def __getitem__(self, name):
        return make_act(name)

    def __contains__(self, name):
        return name in ACTIVATIONS

    def keys(self):
        return ACTIVATIONS.keys()


act_funcs = _ActivationFactory()


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    elif isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            nn.init.xavier_uniform_(module.in_proj_weight)
        if module.out_proj.weight is not None:
            nn.init.xavier_uniform_(module.out_proj.weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)


class cnn_attention(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(cnn_attention, self).__init__()
        self.act_func = act_funcs[params['act_func']]
        self.name_scope = name_scope
        self.layer_id = layer_id
        self.name = f'{name_scope}_conv_att_{layer_id}'
        self.params = params
        if isinstance(size, str):
            self.attention = size
            size = 2
        else:
            self.attention = params['attention']
        self.dim = len(shape)
        self.shape = shape
        if self.attention in {'cbam', 'cbam1', 'att'}:
            shape, layer_id, kernels, pool_size = conv_util(shape, size, layer_id)
            self.size = int(size)
            if self.attention == 'cbam1':
                self.conv1 = CBAM_ver2(shape[1], reduction_ratio=2)
            elif self.attention == 'cbam':
                self.conv1 = CBAM(shape[1], reduction_factor=2, kernel_size=3)
            elif self.attention == 'att':
                self.conv1 = cnn_att(shape, params['filters'], act_func=params['act_func'])
                self.apply(_init_weights)
            x = torch.zeros(10, *shape[1:])
            x_shape1 = list(x.size())
            if len(x_shape1) == 3:
                x = torch.unsqueeze(x, dim=1)
            if len(x_shape1) == 5:
                x = rearrange(x, 'b c1 c2 h w -> b (c1 c2) h w')
            # transform the input

            # Perform the usual forward pass
            x = self.conv1(x)
            if len(x_shape1) == 3:
                x = torch.squeeze(x, dim=1)
            if self.dim == 5:
                x = rearrange(x, 'b (c1 c2) h w -> b c1 c2 h w', c1=self.shape[1], c2=self.shape[2])
            self.output_shape = x.shape
        elif self.attention in {'datt', 'geodatt', 'timegeodatt'}:
            if self.attention == 'geodatt':
                self.conv1 = GeoDualAttentionNetwork(shape)

            x = torch.zeros(10, *shape[1:])
            x_shape1 = list(x.size())
            if len(x_shape1) == 3:
                x = torch.unsqueeze(x, dim=1)
                x = torch.unsqueeze(x, dim=1)

            # Perform the usual forward pass
            x = self.conv1(x)
            if len(x_shape1) == 3:
                x = torch.squeeze(x, dim=1)
                x = torch.squeeze(x, dim=1)
            self.output_shape = x.shape


    def forward(self, x):
        x_shape1 = list(x.size())
        if self.attention in {'cbam', 'cbam1', 'att'}:
            if len(x_shape1) == 3:
                x = torch.unsqueeze(x, dim=1)
            if len(x_shape1) == 5:
                x = rearrange(x, 'b c1 c2 h w -> b (c1 c2) h w')
        else:
            if len(x_shape1) == 3:
                x = torch.unsqueeze(x, dim=1)
                x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        if self.dim == 5 and self.attention in {'cbam', 'cbam1', 'att'}:
            x = rearrange(x, 'b (c1 c2) h w -> b c1 c2 h w', c1=self.shape[1], c2=self.shape[2])
        return x

def conv_util(shape, size, layer_id):
    if not isinstance(shape, list):
        shape = list(shape)
    if len(shape) == 3:
        shape.insert(1, 1)
    if isinstance(layer_id, str):
        layer_id = layer_id.split('_')[0]
    if len(shape) == 5:
        shape = [shape[0], shape[1] * shape[2], shape[3], shape[4]]
    if isinstance(layer_id, str):
        layer_id = int(layer_id.split('_lag')[0])
    kernels = []
    pool_size = []
    for s in shape[-2:]:
        if (s / 2) > size:
            kernels.append(size)
            if s > 50:
                pool_size.append(3)
            elif s > 10:
                pool_size.append(2)
            else:
                pool_size.append(1)
        else:
            kernels.append(int(s / 2)) if int(s / 2) > 1 else kernels.append(1)
            if s > 12:
                pool_size.append(2) if int(s / 2) > 1 else pool_size.append(1)
            else:
                pool_size.append(1)
    return shape, layer_id, kernels, pool_size

class cnn_att(nn.Module):
    def __init__(self, shape, filters, act_func='relu'):
        super(cnn_att, self).__init__()

        self.act_func = act_funcs[act_func]
        self.conv1 = nn.Conv2d(shape[1], int(filters), kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(int(filters), int(2 * filters), kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d()
        # a fixed 2x2 summary whatever the grid size (the former layout assumed a 17x17 grid)
        self.summary = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(int(2 * filters * 2 * 2), 256)
        self.fc2 = nn.Linear(256, int(np.prod(shape[1:])))

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(shape[1], int(filters), kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(int(filters), int(2 * filters), kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(int(2 * filters * 2 * 2), int(4 * filters)),
            nn.ReLU(True),
            nn.Linear(int(4 * filters), 3 * 2)
        )

        self.apply(_init_weights)
        # Initialize the weights/bias with identity transformation (must come after general init)
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = rearrange(xs, 'b c h w -> b (c h w)')
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        """A sigmoid attention mask, computed on the spatially transformed input, applied to the input."""
        shape = list(x.size())
        # transform the input
        x1 = self.stn(x)

        # Perform the usual forward pass
        x1 = self.act_func(F.max_pool2d(self.conv1(x1), 2))
        x1 = self.act_func(self.conv2_drop(self.conv2(x1)))
        x1 = rearrange(self.summary(x1), 'b c h w -> b (c h w)')
        x1 = self.act_func(self.fc1(x1))
        x1 = self.fc2(x1)
        x1 = F.sigmoid(x1)
        x1 = rearrange(x1, 'b (c h w) -> b c h w', c=shape[1], h=shape[2], w=shape[3])
        return x * x1     # the former version returned x: the whole attention path was dead


def _norm(channels, batch_norm, dims=2):
    """GroupNorm by default (batch-size independent); BatchNorm when the method asks for it."""
    if batch_norm:
        return nn.BatchNorm2d(channels) if dims == 2 else nn.BatchNorm3d(channels)
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


def _odd(k):
    k = int(max(1, k))
    return k if k % 2 == 1 else k + 1


def _default_pool(h, w):
    return 2 if min(h, w) >= 8 else 1


class ConvBlock(nn.Module):
    """One convolution block, shared over the lags of an NWP tensor.

    Input (b, c, h, w) or (b, t, c, h, w).  The lags t are folded into the batch, so one set of
    kernels serves every lag (time-distributed with weight sharing); the output keeps the lag
    axis: (b, t, filters, h', w').  Block: conv (kernel ``size`` made odd, same padding) -> norm
    (GroupNorm, or BatchNorm with ``batch_norm``) -> activation -> optional residual (when the
    channel count is unchanged and there is no pooling) -> average pooling (``pool``: an int per
    layer; default 2 while the grid is at least 8 wide, else 1).  Filters: ``filters`` x
    (layer index + 1), as before, or ``filters`` alone with ``filters_grow: false``.

    With ``temporal_summary`` (set by the builder for the last convolution of a time-distributed
    branch) the block ends with global average pooling over the grid and an LSTM over the lags,
    giving (b, t, filters): one vector per lead time, ready for a Flatten, a dense per token or an
    attention merge over the lags.
    """

    def __init__(self, shape, params, size, name_scope, layer_id, train=True, dims=2):
        super(ConvBlock, self).__init__()
        shape = list(shape)
        if len(shape) == 3:                      # (b, h, w): a single field
            shape = [shape[0], 1, 1, shape[1], shape[2]]
        elif len(shape) == 4:                    # (b, c, h, w): no lag axis
            shape = [shape[0], 1, shape[1], shape[2], shape[3]]
        self.in_rank = len(shape)
        B, T, C, H, W = shape
        self.shape = shape
        self.dims = dims
        idx = int(str(layer_id).split('_')[0]) if str(layer_id).split('_')[0].isdigit() else 0
        filters = int(params['filters'])
        out_channels = filters * (idx + 1) if params.get('filters_grow', True) else filters
        kernel = _odd(size if isinstance(size, (int, float)) and size >= 1 else 3)
        kernel = min(kernel, _odd(min(H, W)))
        pool = int(params.get('pool', _default_pool(H, W)))
        self.pool_size = max(1, pool)
        self.name = f'{name_scope}_conv_{dims}d_{layer_id}'
        act = act_funcs[params.get('act_func') or 'gelu']
        if dims == 2:
            self.conv = nn.Conv2d(C, out_channels, kernel_size=kernel, padding='same')
            self.pool = nn.AvgPool2d(self.pool_size) if self.pool_size > 1 else nn.Identity()
        else:
            kt = min(_odd(params.get('kernel_time', 3)), _odd(T))
            self.conv = nn.Conv3d(C, out_channels, kernel_size=(kt, kernel, kernel), padding='same')
            self.pool = nn.AvgPool3d((1, self.pool_size, self.pool_size)) if self.pool_size > 1 else nn.Identity()
        self.norm = _norm(out_channels, params.get('batch_norm', False), dims)
        self.act = act
        self.residual = bool(params.get('residual', True)) and C == out_channels
        self.temporal_summary = bool(params.get('temporal_summary', False))
        if self.temporal_summary:
            self.lstm = nn.LSTM(out_channels, out_channels, batch_first=True)
        self.apply(_init_weights)
        with torch.no_grad():
            self.output_shape = list(self.forward(torch.zeros(10, *shape[1:])).shape)

    def _block(self, x):
        y = self.act(self.norm(self.conv(x)))
        if self.residual:
            y = y + x
        return self.pool(y)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1).unsqueeze(1)
        elif x.dim() == 4:
            x = x.unsqueeze(1)
        b, t = x.shape[0], x.shape[1]
        if self.dims == 2:
            y = self._block(rearrange(x, 'b t c h w -> (b t) c h w'))
            y = rearrange(y, '(b t) f h w -> b t f h w', b=b, t=t)
        else:
            y = self._block(rearrange(x, 'b t c h w -> b c t h w'))
            y = rearrange(y, 'b f t h w -> b t f h w')
        if self.temporal_summary:
            pooled = y.mean(dim=(-2, -1))                      # (b, t, f): one vector per lag
            out, _ = self.lstm(pooled)
            return self.act(out)
        return y


class conv_2d(ConvBlock):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(conv_2d, self).__init__(shape, params, size, name_scope, layer_id, train=train, dims=2)


class conv_3d(ConvBlock):
    """The same block with a 3-D kernel over (lags, lat, lon); pooling on the grid only."""

    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(conv_3d, self).__init__(shape, params, size, name_scope, layer_id, train=train, dims=3)


# time-distributed = the shared block; the builder switches the temporal summary on for the last
# convolution of the branch.  cnnT (a second convolution on the transposed grid) added cost without
# benefit and maps to the block.
time_distr_conv_2d = conv_2d
time_distr_conv_3d = conv_3d
cnnT = conv_2d


def init_lstm_data(shape, layer_id, use_embedding, zero_embed=True):
    # The steps a layer is to answer for: the future block when there is one, else the past one.
    # A case whose sequences are observations only -- no calendar, no covariates, no future block
    # -- has ``past_obs`` and nothing else; it used to end here with KeyError: 'past_calendar'.
    for block in ('future_data', 'future_calendar', 'past_data', 'past_calendar', 'past_obs'):
        if block in shape.keys():
            B, Tpred, Ctime = shape[block]
            break
    else:
        raise ValueError(f'the sequence data holds none of future_data, future_calendar, past_data, past_calendar '
                         f'and past_obs (it holds {sorted(shape.keys())}): there is nothing to read')
    if use_embedding:
        obs_embed = RnnEmbedding(shape['past_obs'], Tpred, layer_id, not zero_embed) \
            if 'past_obs' in shape.keys() else None
    else:
        obs_embed = None
    if 'past_calendar' in shape.keys() and 'future_calendar' in shape.keys():
        x_mark_enc = torch.zeros(shape['past_calendar'])
        x_mark_dec = torch.zeros(shape['future_calendar'])
    elif 'past_calendar' in shape.keys():
        x_mark_enc = torch.zeros(shape['past_calendar'])
        x_mark_dec = None
    elif 'future_calendar' in shape.keys():
        x_mark_enc = None
        x_mark_dec = torch.zeros(shape['future_calendar'])
    else:
        x_mark_enc = None
        x_mark_dec = None

    if 'past_data' in shape.keys() and 'future_data' in shape.keys():
        x_enc = torch.zeros(shape['past_data'])
        x_dec = torch.zeros(shape['future_data'])
    elif 'past_data' in shape.keys():
        x_enc = torch.zeros(shape['past_data'])
        x_dec = None
    elif 'future_data' in shape.keys():
        x_enc = None
        x_dec = torch.zeros(shape['future_data'])
    else:
        x_enc = None
        x_dec = None

    if 'past_obs' in shape.keys():
        past_obs = torch.zeros(shape['past_obs'])
    else:
        past_obs = None

    past_obs_emd = obs_embed(past_obs) if obs_embed is not None and past_obs is not None else None
    past_obs = cat_without_none([past_obs, past_obs_emd], -1)
    return past_obs, past_obs_emd, obs_embed, x_enc, x_dec, x_mark_enc, x_mark_dec, B, Tpred

def init_lstm_forward_data(x, obs_embed):
    if 'past_calendar' in x.keys() and 'future_calendar' in x.keys():
        x_mark_enc = x['past_calendar']
        x_mark_dec = x['future_calendar']
    elif 'past_calendar' in x.keys():
        x_mark_enc = x['past_calendar']
        x_mark_dec = None
    elif 'future_calendar' in x.keys():
        x_mark_enc = None
        x_mark_dec = x['future_calendar']
    else:
        x_mark_enc = None
        x_mark_dec = None

    if 'past_data' in x.keys() and 'future_data' in x.keys():
        x_enc = x['past_data']
        x_dec = x['future_data']
    elif 'past_data' in x.keys():
        x_enc = x['past_data']
        x_dec = None
    elif 'future_data' in x.keys():
        x_enc = None
        x_dec = x['future_data']
    else:
        x_enc = None
        x_dec = None

    past_obs = x['past_obs'] if 'past_obs' in x.keys() else None

    past_obs_emd = obs_embed(past_obs) if obs_embed is not None and past_obs is not None else None
    past_obs = cat_without_none([past_obs, past_obs_emd], -1)

    return past_obs, past_obs_emd, x_enc, x_dec, x_mark_enc, x_mark_dec

#: The parameter that says a recurrent layer may take a case's past data for its past when the case
#: has no observations. A network trained before 2026-11-24 does not carry it and is built as it was
#: trained (``network.DeepNetwork.load``), so its weights fit; what is trained from then on carries it.
PAST_DATA_IS_PAST = 'past_data_is_past'


def past_of(past_obs, x_enc, params=None):
    """What a recurrent layer reads as the past: the observations, and where a case has none, its past data.

    A series that a case gives both before and after the row (the wind of a park whose power is
    reconstructed from wind: lags -28 ... -1 in the past block, 0 ... 28 in the future block)
    is no observation to the pipeline but a covariate, ``past_data``. ``lstm`` and ``lstm_att``
    took the past from the observations and the past calendar only: without observations
    ``lstm`` left the past data out and said nothing, and ``lstm_att`` could not be built
    when there was no calendar either.

    Nothing changes for a case that has observations, whatever else it has. A case without
    them that was trained before this (``PAST_DATA_IS_PAST`` absent from the saved network, read
    as false) is built as it was.
    """
    if past_obs is not None or not (params or {}).get(PAST_DATA_IS_PAST, True):
        return past_obs
    return x_enc


def cat_without_none(list_none, axis):
    list_without_none = [mat for mat in list_none if mat is not None]
    if len(list_without_none) == 0:
        return None
    elif len(list_without_none) == 1:
        return list_without_none[0]
    else:
        return torch.cat(list_without_none, axis)

class lstm(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(lstm, self).__init__()
        self.shape = shape
        self.size = size
        self.params = params
        self.is_lstm_output = params['is_lstm_output']
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_lstm_{layer_id}'
        if isinstance(shape, dict):
            past_obs, _, obs_embed, x_enc, x_dec, x_mark_enc, x_mark_dec, B, Tpred = init_lstm_data(shape, layer_id,
                                                                                                            False)
            self.obs_embed = obs_embed
            if not self.params['use_nwp']:
                x_dec = None
                x_mark_dec = None
            # future_calendar = cat_without_none([x_mark_enc, x_mark_dec], 1)
            future_data = cat_without_none([x_dec, x_mark_dec], -1)
            x_enc2 = future_data
            # x_enc2 = cat_without_none([future_data, future_calendar], -1)
            x_enc1 = cat_without_none([past_of(past_obs, x_enc, self.params), x_mark_enc], -1)
            if x_enc1 is not None and x_enc2 is not None:
                if x_enc1.shape[1] != x_enc2.shape[1]:
                    self.enc_in = max([x_enc1.shape[-1], x_enc2.shape[-1]])
                    self.lstm = nn.ModuleList([nn.LSTM(x_enc1.shape[-1], int(size * self.enc_in), batch_first=True),
                                 nn.LSTM(x_enc2.shape[-1], int(size * self.enc_in), batch_first=True),
                                 nn.LSTM(int(size * self.enc_in), int(size * self.enc_in), num_layers=int(size), batch_first=True)])
                    hidden_state = (torch.zeros(1, B, int(size * self.enc_in)), torch.zeros(1, B, int(size * self.enc_in)))
                    lstm_out1, hidden_state = self.lstm[0](x_enc1, hidden_state)
                    lstm_out1 = self.act_func(lstm_out1)
                    lstm_out2, hidden_state = self.lstm[1](x_enc2, hidden_state)
                    lstm_out2 = self.act_func(lstm_out2)
                    hidden_state = (torch.zeros(int(size), B, int(size * self.enc_in)), torch.zeros(int(size), B, int(size * self.enc_in)))
                    lstm_out, hidden_state = self.lstm[2](torch.cat([lstm_out1, lstm_out2], 1), hidden_state)
                    self.output_shape = self.act_func(lstm_out[:, -params['n_out']:, :]).shape
                else:
                    # the message used to say the two "have different shapes": the opposite
                    raise ValueError(
                        f'{self.name}: the past and the future block are both {x_enc1.shape[1]} steps long, and '
                        f'this layer is written for blocks of different lengths. Use lstm_att, timexer, tide or '
                        f'tsmixer in this graph, or switch use_nwp off so that only the past is read')
            else:
                x_enc = cat_without_none([x_enc1, x_enc2], 1)
                if x_enc is None:
                    raise ValueError("x_enc is None")
                self.enc_in = x_enc.shape[-1]
                self.lstm = nn.LSTM(self.enc_in, int(size * self.enc_in), batch_first=True)
                hidden_state = (torch.zeros(1, B, int(size * self.enc_in)), torch.zeros(1, B, int(size * self.enc_in)))
                lstm_out = self.lstm(x_enc, hidden_state)[0]
                self.output_shape = self.act_func(lstm_out).shape
        else:
            B = shape[0]
            Tpred = shape[1]
            x_dec = torch.zeros(shape)
            self.enc_in = shape[-1]
            self.lstm = nn.LSTM(self.enc_in, int(size * self.enc_in), batch_first=True)
            hidden_state = (torch.zeros(1, B, int(size * self.enc_in)), torch.zeros(1, B, int(size * self.enc_in)))
            lstm_out = self.lstm(x_dec, hidden_state)[0]
            if self.is_lstm_output:
                self.output_shape = self.act_func(lstm_out)[:, -Tpred:, :].shape
            else:
                self.output_shape = self.act_func(lstm_out).shape
        self.apply(_init_weights)


    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.lstm.weight.size())}")


    def forward(self, x):
        if isinstance(x, dict):
            past_obs, _, x_enc, x_dec, x_mark_enc, x_mark_dec = init_lstm_forward_data(x, self.obs_embed)
            if not self.params['use_nwp']:
                x_dec = None
                x_mark_dec = None
            # future_calendar = cat_without_none([x_mark_enc, x_mark_dec], 1)
            future_data = cat_without_none([x_dec, x_mark_dec], -1)
            x_enc2 = future_data
            # x_enc2 = cat_without_none([future_data, future_calendar], -1)
            x_enc1 = cat_without_none([past_of(past_obs, x_enc, self.params), x_mark_enc], -1)
            if x_enc1 is not None and x_enc2 is not None:
                if x_enc1.shape[1] != x_enc2.shape[1]:
                    device = x_enc1.device
                    B = x_enc1.shape[0]
                    hidden_state = (torch.zeros(1, B, int(self.size * self.enc_in)).to(device),
                                    torch.zeros(1, B, int(self.size * self.enc_in)).to(device))
                    lstm_out1, hidden_state = self.lstm[0](x_enc1, hidden_state)
                    lstm_out1 = self.act_func(lstm_out1)
                    lstm_out2, hidden_state = self.lstm[1](x_enc2, hidden_state)
                    lstm_out2 = self.act_func(lstm_out2)
                    hidden_state = (torch.zeros(int(self.size), B, int(self.size * self.enc_in)).to(device),
                                    torch.zeros(int(self.size), B, int(self.size * self.enc_in)).to(device))
                    lstm_out, hidden_state = self.lstm[2](torch.cat([lstm_out1, lstm_out2], 1), hidden_state)
                    return self.act_func(lstm_out[:, -self.params['n_out']:, :])
            else:
                x_enc = cat_without_none([x_enc1, x_enc2], 1)
                B = x_enc.shape[0]
                if x_enc is None:
                    raise ValueError("x_enc is None")
                device = x_enc.device
                hidden_state = (torch.zeros(1, B, int(self.size * self.enc_in)).to(device),
                                torch.zeros(1, B, int(self.size * self.enc_in)).to(device))
                lstm_out = self.lstm(x_enc, hidden_state)[0]
                return self.act_func(lstm_out)
        else:
            B = x.shape[0]
            inp = x
            device = x.device
            hidden_state = (torch.zeros(1, B, int(self.size * self.enc_in)).to(device),
                            torch.zeros(1, B, int(self.size * self.enc_in)).to(device))
            lstm_out = self.lstm(inp, hidden_state)[0]
            if self.is_lstm_output:
                return self.act_func(lstm_out)[:, -1, :]
            else:
                return self.act_func(lstm_out)


class lstm_att(nn.Module):
    """Encoder-decoder LSTM with cross attention over the past steps.

    Dictionary input (the ``lstm`` data tag): the past observations and calendar are encoded by one
    LSTM, the future covariates and calendar by another (initialised from the encoder state); the
    future steps then attend over the encoded past steps with multi-head attention (softmax over
    the past positions), residual + feed-forward.  Output: one vector per future step,
    ``(b, T_future, hidden)``, a token sequence for a Flatten, a dense per token or an attention
    merge.  ``hidden = size x input width`` rounded to a multiple of ``heads`` (4).

    The former version flattened both LSTM outputs and pushed them through the gate with a head
    dimension equal to the square of the input width (projections of tens of thousands of units).
    Tensor input: a plain LSTM, as ``lstm``.
    """

    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(lstm_att, self).__init__()
        self.shape = shape
        self.size = size
        self.params = params
        self.is_lstm_output = params['is_lstm_output']
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_lstm_att_{layer_id}'
        heads = int(params.get('heads', 4))
        dropout = float(params.get('attention_dropout', 0.1))
        if isinstance(shape, dict):
            past_obs, _, obs_embed, x_enc, x_dec, x_mark_enc, x_mark_dec, B, Tpred = init_lstm_data(shape, layer_id,
                                                                                                 False)
            self.obs_embed = obs_embed
            if not self.params['use_nwp']:
                x_dec, x_mark_dec = past_of(past_obs, x_enc, self.params), x_mark_enc
            x_past = cat_without_none([past_of(past_obs, x_enc, self.params), x_mark_enc], -1)
            x_future = cat_without_none([x_dec, x_mark_dec], -1)
            if x_past is None or x_future is None:
                raise ValueError('lstm_att needs past and future inputs (past_obs/past_calendar and future data/calendar)')
            width = max(x_past.shape[-1], x_future.shape[-1])
            hidden = int(np.ceil(max(size * width, heads) / heads) * heads)
            self.hidden = hidden
            self.encoder = nn.LSTM(x_past.shape[-1], hidden, batch_first=True)
            self.decoder = nn.LSTM(x_future.shape[-1], hidden, batch_first=True)
            self.norm_q = nn.LayerNorm(hidden)
            self.norm_kv = nn.LayerNorm(hidden)
            self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))
            self.last_attention = None
            self.apply(_init_weights)
            with torch.no_grad():
                self.output_shape = list(self._forward_dict(x_past, x_future).shape)
        else:
            B, Tpred = shape[0], shape[1]
            self.enc_in = shape[-1]
            self.lstm = nn.LSTM(self.enc_in, int(size * self.enc_in), batch_first=True)
            self.apply(_init_weights)
            with torch.no_grad():
                out = self.act_func(self.lstm(torch.zeros(shape))[0])
                self.output_shape = list((out[:, -1, :] if self.is_lstm_output else out).shape)

    def _forward_dict(self, x_past, x_future):
        encoded, state = self.encoder(x_past)
        decoded, _ = self.decoder(x_future, state)
        attended, weights = self.attn(self.norm_q(decoded), self.norm_kv(encoded), self.norm_kv(encoded),
                                      need_weights=True)
        self.last_attention = weights.detach()
        out = decoded + attended
        return out + self.ffn(out)

    def forward(self, x):
        if isinstance(x, dict):
            past_obs, _, x_enc, x_dec, x_mark_enc, x_mark_dec = init_lstm_forward_data(x, self.obs_embed)
            if not self.params['use_nwp']:
                x_dec, x_mark_dec = past_of(past_obs, x_enc, self.params), x_mark_enc
            x_past = cat_without_none([past_of(past_obs, x_enc, self.params), x_mark_enc], -1)
            x_future = cat_without_none([x_dec, x_mark_dec], -1)
            return self._forward_dict(x_past, x_future)
        out = self.act_func(self.lstm(x)[0])
        return out[:, -1, :] if self.is_lstm_output else out


class _TimeXerBlock(nn.Module):
    """One TimeXer encoder layer: self-attention among the endogenous tokens (patches + global token),
    cross-attention of the global token over the exogenous variate tokens, feed-forward; pre-norm."""

    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.last_attention = None

    def forward(self, en, ex):
        h = self.norm_self(en)
        en = en + self.drop(self.self_attn(h, h, h, need_weights=False)[0])
        if ex is not None:
            glb = en[:, -1:, :]
            kv = self.norm_kv(ex)
            attended, weights = self.cross_attn(self.norm_q(glb), kv, kv, need_weights=True)
            self.last_attention = weights.detach()
            en = torch.cat([en[:, :-1, :], glb + self.drop(attended)], dim=1)
        return en + self.drop(self.ffn(self.norm_ff(en)))


class timexer(nn.Module):
    """TimeXer-style encoder for the ``lstm`` data (Wang et al., NeurIPS 2024): endogenous patches,
    exogenous variate tokens, a global token that links the two.

    Dictionary input.  The *endogenous* series is the past block (past observations, plus ``past_data``
    when present): non-overlapping patches of ``patch`` steps (default 4; the series is zero-padded in
    front to a multiple of it) become tokens through one linear layer over ``patch x features``, with a
    learned positional embedding, followed by a learned global token.  Every *exogenous* series is one
    token: each past calendar column embedded over the past length, and with ``use_nwp`` each future
    covariate and future calendar column embedded over the future length (the known NWP inputs of the
    horizon).  ``depth`` (2) blocks: self-attention among the endogenous tokens, cross-attention of the
    global token over the exogenous tokens, feed-forward.  ``d_model = size`` when >= 8 (else 64),
    rounded to a multiple of ``heads`` (4).

    Output (``output`` parameter): ``tokens`` (default) the ``n_patches + 1`` tokens of ``d_model`` for a
    Flatten, a dense per token or an attention merge; ``global`` the global token ``(b, d_model)``;
    ``flat`` all tokens concatenated.  ``last_attention`` keeps the global-to-exogenous weights of the
    last block, one column per exogenous series.  A plain ``(b, T, F)`` tensor is treated as endogenous
    only (no cross-attention).
    """

    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(timexer, self).__init__()
        self.shape = shape
        self.params = params
        self.name = f'{name_scope}_timexer_{layer_id}'
        heads = int(params.get('heads', 4))
        dropout = float(params.get('attention_dropout', 0.1))
        d_model = int(size) if isinstance(size, (int, float)) and size >= 8 else 64
        self.d_model = int(np.ceil(d_model / heads) * heads)
        self.depth = int(params.get('depth', 2))
        self.output_mode = str(params.get('output', 'tokens'))
        self.use_nwp = bool(params.get('use_nwp', True))
        if isinstance(shape, dict):
            past_obs, _, _, x_enc, x_dec, x_mark_enc, x_mark_dec, _, _ = init_lstm_data(shape, layer_id, False)
            en, ex_past, ex_future = self._split(past_obs, x_enc, x_dec, x_mark_enc, x_mark_dec)
        else:
            en, ex_past, ex_future = torch.zeros(shape), None, None
        if en is None:
            raise ValueError('timexer needs past observations (past_obs / past_data) as the endogenous series')
        L, F = en.shape[1], en.shape[2]
        self.patch = max(1, min(int(params.get('patch', 4)), L))
        self.n_patches = int(np.ceil(L / self.patch))
        self.pad = self.n_patches * self.patch - L
        self.en_embed = nn.Linear(self.patch * F, self.d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
        self.glb = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.glb, std=0.02)
        self.ex_embed_past = nn.Linear(ex_past.shape[1], self.d_model) if ex_past is not None else None
        self.ex_embed_future = nn.Linear(ex_future.shape[1], self.d_model) if ex_future is not None else None
        self.n_exogenous = (ex_past.shape[2] if ex_past is not None else 0) + (ex_future.shape[2] if ex_future is not None else 0)
        self.blocks = nn.ModuleList([_TimeXerBlock(self.d_model, heads, dropout) for _ in range(self.depth)])
        self.norm = nn.LayerNorm(self.d_model)
        self.apply(_init_weights)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.glb, std=0.02)
        with torch.no_grad():
            self.output_shape = list(self._forward_parts(en, ex_past, ex_future).shape)

    def _split(self, past_obs, x_enc, x_dec, x_mark_enc, x_mark_dec):
        """(endogenous (b, L, F), exogenous past (b, L_past, M1) or None, exogenous future (b, L_future, M2) or None)."""
        en = cat_without_none([past_obs, x_enc], -1)
        ex_past = x_mark_enc
        ex_future = cat_without_none([x_dec, x_mark_dec], -1) if self.use_nwp else None
        return en, ex_past, ex_future

    def _forward_parts(self, en, ex_past, ex_future):
        if self.pad:
            en = torch.nn.functional.pad(en, (0, 0, self.pad, 0))
        b = en.shape[0]
        tokens = self.en_embed(en.reshape(b, self.n_patches, self.patch * en.shape[-1])) + self.pos
        tokens = torch.cat([tokens, self.glb.expand(b, -1, -1)], dim=1)
        ex = []
        if ex_past is not None and self.ex_embed_past is not None:
            ex.append(self.ex_embed_past(ex_past.transpose(1, 2)))            # (b, M1, d): one token per past exogenous series
        if ex_future is not None and self.ex_embed_future is not None:
            ex.append(self.ex_embed_future(ex_future.transpose(1, 2)))        # (b, M2, d): one token per future covariate
        ex = torch.cat(ex, dim=1) if ex else None
        for block in self.blocks:
            tokens = block(tokens, ex)
        tokens = self.norm(tokens)
        if self.output_mode == 'global':
            return tokens[:, -1, :]
        if self.output_mode == 'flat':
            return tokens.reshape(b, -1)
        return tokens

    @property
    def last_attention(self):
        return self.blocks[-1].last_attention if self.blocks else None

    def forward(self, x):
        if isinstance(x, dict):
            past_obs, _, x_enc, x_dec, x_mark_enc, x_mark_dec = init_lstm_forward_data(x, None)
            en, ex_past, ex_future = self._split(past_obs, x_enc, x_dec, x_mark_enc, x_mark_dec)
            return self._forward_parts(en, ex_past, ex_future)
        return self._forward_parts(x, None, None)


class _ResidualBlock(nn.Module):
    """TiDE residual block: Linear -> ReLU -> Linear -> dropout, a linear skip, LayerNorm."""

    def __init__(self, n_in, hidden, n_out, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out), nn.Dropout(dropout))
        self.skip = nn.Linear(n_in, n_out)
        self.norm = nn.LayerNorm(n_out)

    def forward(self, x):
        return self.norm(self.net(x) + self.skip(x))


class tide(nn.Module):
    """TiDE, the time-series dense encoder (Das et al., 2023), for the ``lstm`` data: MLP only, future
    covariates native, a minute per trial.

    Dictionary input.  The past observations (plus ``past_data``) are the target history; the past
    calendar and, with ``use_nwp``, the future covariates and future calendar are the covariates, each
    step projected to ``cov_dim`` (8) features by a residual block shared over time (one per side).
    Encoder: the flattened history and the flattened projected covariates of both sides go through
    ``depth`` (2) residual blocks of ``hidden`` (128) units.  Decoder: ``depth`` residual blocks and a
    linear layer to ``horizon x decoder_dim`` (16), reshaped to one vector per future step.  Temporal
    decoder: per future step, the decoded vector concatenated with that step's projected covariates
    through one residual block to ``size`` features.  A linear skip from the flattened history to the
    horizon is appended as one extra feature per step (``skip``, default on), TiDE's global residual.

    Output ``(b, horizon, size + 1)`` (``output='sequence'``, default): a token per future step for a
    Flatten, a dense per step or an attention merge; ``output='flat'`` concatenates them.  The horizon
    is the future length of the data (``n_out`` for a plain tensor input).
    """

    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(tide, self).__init__()
        self.shape = shape
        self.params = params
        self.name = f'{name_scope}_tide_{layer_id}'
        self.use_nwp = bool(params.get('use_nwp', True))
        hidden = int(params.get('hidden', 128))
        depth = int(params.get('depth', 2))
        dropout = float(params.get('dropout', params.get('attention_dropout', 0.1)))
        self.cov_dim = int(params.get('cov_dim', 8))
        self.decoder_dim = int(params.get('decoder_dim', 16))
        self.out_dim = max(1, int(size)) if isinstance(size, (int, float)) else 32
        self.use_skip = bool(params.get('skip', True))
        self.output_mode = str(params.get('output', 'sequence'))
        if isinstance(shape, dict):
            past_obs, _, _, x_enc, x_dec, x_mark_enc, x_mark_dec, _, Tpred = init_lstm_data(shape, layer_id, False)
            history, cov_past, cov_future = self._split(past_obs, x_enc, x_dec, x_mark_enc, x_mark_dec)
            horizon = cov_future.shape[1] if cov_future is not None else \
                (x_mark_dec.shape[1] if x_mark_dec is not None else (x_dec.shape[1] if x_dec is not None else int(Tpred)))
        else:
            history, cov_past, cov_future = torch.zeros(shape), None, None
            horizon = int(params.get('n_out', shape[1]))
        if history is None:
            raise ValueError('tide needs past observations (past_obs / past_data) as the target history')
        self.horizon = int(horizon)
        L, F = history.shape[1], history.shape[2]
        self.proj_past = _ResidualBlock(cov_past.shape[2], hidden, self.cov_dim, dropout) if cov_past is not None else None
        self.proj_future = _ResidualBlock(cov_future.shape[2], hidden, self.cov_dim, dropout) if cov_future is not None else None
        enc_in = L * F + (L * self.cov_dim if cov_past is not None else 0) + (self.horizon * self.cov_dim if cov_future is not None else 0)
        encoder = [_ResidualBlock(enc_in, hidden, hidden, dropout)] + [_ResidualBlock(hidden, hidden, hidden, dropout) for _ in range(depth - 1)]
        self.encoder = nn.Sequential(*encoder)
        decoder = [_ResidualBlock(hidden, hidden, hidden, dropout) for _ in range(depth - 1)] + \
                  [_ResidualBlock(hidden, hidden, self.horizon * self.decoder_dim, dropout)]
        self.decoder = nn.Sequential(*decoder)
        self.temporal = _ResidualBlock(self.decoder_dim + (self.cov_dim if cov_future is not None else 0), hidden, self.out_dim, dropout)
        self.skip = nn.Linear(L * F, self.horizon) if self.use_skip else None
        self.apply(_init_weights)
        with torch.no_grad():
            self.output_shape = list(self._forward_parts(history, cov_past, cov_future).shape)

    def _split(self, past_obs, x_enc, x_dec, x_mark_enc, x_mark_dec):
        history = cat_without_none([past_obs, x_enc], -1)
        cov_past = x_mark_enc
        cov_future = cat_without_none([x_dec, x_mark_dec], -1) if self.use_nwp else None
        return history, cov_past, cov_future

    def _forward_parts(self, history, cov_past, cov_future):
        b = history.shape[0]
        parts = [history.reshape(b, -1)]
        if cov_past is not None and self.proj_past is not None:
            parts.append(self.proj_past(cov_past).reshape(b, -1))
        proj_future = self.proj_future(cov_future) if (cov_future is not None and self.proj_future is not None) else None
        if proj_future is not None:
            parts.append(proj_future.reshape(b, -1))
        e = self.encoder(torch.cat(parts, dim=-1))
        decoded = self.decoder(e).reshape(b, self.horizon, self.decoder_dim)
        step_in = torch.cat([decoded, proj_future], dim=-1) if proj_future is not None else decoded
        out = self.temporal(step_in)                                             # (b, horizon, out_dim)
        if self.skip is not None:
            out = torch.cat([out, self.skip(history.reshape(b, -1)).unsqueeze(-1)], dim=-1)
        return out.reshape(b, -1) if self.output_mode == 'flat' else out

    def forward(self, x):
        if isinstance(x, dict):
            past_obs, _, x_enc, x_dec, x_mark_enc, x_mark_dec = init_lstm_forward_data(x, None)
            history, cov_past, cov_future = self._split(past_obs, x_enc, x_dec, x_mark_enc, x_mark_dec)
            return self._forward_parts(history, cov_past, cov_future)
        return self._forward_parts(x, None, None)


class _AFNOMixer(nn.Module):
    """Adaptive Fourier neural operator token mixing on a 2-D grid (Guibas et al., ICLR 2022): a 2-D real
    FFT over the grid, a block-diagonal two-layer complex MLP per frequency (``num_blocks`` blocks of
    channels), soft-shrinkage for sparsity, hard thresholding of the high modes, inverse FFT."""

    def __init__(self, channels, num_blocks=4, sparsity_threshold=0.01, hard_thresholding_fraction=1.0, hidden_factor=1):
        super().__init__()
        assert channels % num_blocks == 0, 'channels must be divisible by num_blocks'
        self.num_blocks = num_blocks
        self.block_size = channels // num_blocks
        self.sparsity_threshold = float(sparsity_threshold)
        self.hard_thresholding_fraction = float(hard_thresholding_fraction)
        hidden = self.block_size * int(hidden_factor)
        scale = 0.02
        self.w1 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size, hidden))
        self.b1 = nn.Parameter(scale * torch.randn(2, num_blocks, hidden))
        self.w2 = nn.Parameter(scale * torch.randn(2, num_blocks, hidden, self.block_size))
        self.b2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size))

    def forward(self, x):                                   # (b, h, w, c) real
        b, h, w, c = x.shape
        dtype = x.dtype
        x = torch.fft.rfft2(x.float(), dim=(1, 2), norm='ortho')
        x = x.reshape(b, h, w // 2 + 1, self.num_blocks, self.block_size)
        kept_h = max(1, int((h // 2 + 1) * self.hard_thresholding_fraction))
        kept_w = max(1, int((w // 2 + 1) * self.hard_thresholding_fraction))
        rows = torch.cat([torch.arange(0, kept_h), torch.arange(h - kept_h + 1, h)]) if kept_h < h // 2 + 1 else torch.arange(h)
        rows = rows.to(x.device)
        xs = x[:, rows, :kept_w]
        o1_real = torch.relu(torch.einsum('...bi,bio->...bo', xs.real, self.w1[0]) - torch.einsum('...bi,bio->...bo', xs.imag, self.w1[1]) + self.b1[0])
        o1_imag = torch.relu(torch.einsum('...bi,bio->...bo', xs.imag, self.w1[0]) + torch.einsum('...bi,bio->...bo', xs.real, self.w1[1]) + self.b1[1])
        o2_real = torch.einsum('...bi,bio->...bo', o1_real, self.w2[0]) - torch.einsum('...bi,bio->...bo', o1_imag, self.w2[1]) + self.b2[0]
        o2_imag = torch.einsum('...bi,bio->...bo', o1_imag, self.w2[0]) + torch.einsum('...bi,bio->...bo', o1_real, self.w2[1]) + self.b2[1]
        mixed = torch.nn.functional.softshrink(torch.stack([o2_real, o2_imag], dim=-1), lambd=self.sparsity_threshold)
        out = torch.zeros(b, h, w // 2 + 1, self.num_blocks, self.block_size, 2, device=x.device, dtype=mixed.dtype)
        out[:, rows, :kept_w] = mixed
        out = torch.view_as_complex(out.contiguous()).reshape(b, h, w // 2 + 1, c)
        return torch.fft.irfft2(out, s=(h, w), dim=(1, 2), norm='ortho').to(dtype)


class _AFNOBlock(nn.Module):
    def __init__(self, channels, num_blocks, sparsity_threshold, hard_thresholding_fraction, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.mixer = _AFNOMixer(channels, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, channels), nn.Dropout(dropout))

    def forward(self, x):                                   # (b, h, w, c)
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class afno(nn.Module):
    """Adaptive Fourier neural operator on the NWP grid (Guibas et al., 2022; the mixer of FourCastNet).

    Input ``(b, t, c, h, w)`` or ``(b, c, h, w)``.  Every grid cell (or ``patch`` x ``patch`` cells)
    becomes a token whose features are the lags and variables of the cell, projected to ``size``
    channels (>= 8, else 64; rounded to a multiple of ``num_blocks``) with a learned 2-D positional
    embedding.  ``depth`` (2) blocks: LayerNorm, AFNO mixing (a 2-D FFT over the grid, a block-diagonal
    complex MLP with ``num_blocks`` (4) blocks per frequency, soft-shrink with ``sparsity_threshold``
    (0.01), the low ``hard_thresholding_fraction`` (1.0 = all) of the modes kept, inverse FFT) with a
    residual, then a channel MLP (``mlp_ratio`` 2) with a residual.  Global spatial mixing on a 17x17
    physical field for a few thousand parameters per block, an alternative to the convolution stack
    and to the grid transformer.

    Output (``output``): ``grid`` (default) the feature map ``(b, size, h, w)`` for a convolution, a
    Flatten or ``tokens`` (one token per cell); ``tokens`` the sequence ``(b, h*w, size)`` directly.
    """

    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(afno, self).__init__()
        shape = list(shape)
        if len(shape) == 4:
            shape = [shape[0], 1, shape[1], shape[2], shape[3]]
        B, T, C, H, W = shape
        self.name = f'{name_scope}_afno_{layer_id}'
        self.num_blocks = int(params.get('num_blocks', 4))
        d_model = int(size) if isinstance(size, (int, float)) and size >= 8 else 64
        self.d_model = int(np.ceil(d_model / self.num_blocks) * self.num_blocks)
        self.patch = max(1, min(int(params.get('patch', 1)), H, W))
        self.hp, self.wp = H // self.patch, W // self.patch
        self.crop = (self.hp * self.patch, self.wp * self.patch)
        self.output_mode = str(params.get('output', 'grid'))
        dropout = float(params.get('dropout', params.get('attention_dropout', 0.1)))
        self.proj = nn.Linear(T * C * self.patch * self.patch, self.d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.hp, self.wp, self.d_model))
        self.blocks = nn.ModuleList([_AFNOBlock(self.d_model, self.num_blocks, params.get('sparsity_threshold', 0.01),
                                                params.get('hard_thresholding_fraction', 1.0), params.get('mlp_ratio', 2), dropout)
                                     for _ in range(int(params.get('depth', 2)))])
        self.norm = nn.LayerNorm(self.d_model)
        self.apply(_init_weights)
        nn.init.trunc_normal_(self.pos, std=0.02)
        with torch.no_grad():
            self.output_shape = list(self.forward(torch.zeros(10, *shape[1:])).shape)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = x[..., :self.crop[0], :self.crop[1]]
        tokens = rearrange(x, 'b t c (hp p1) (wp p2) -> b hp wp (t c p1 p2)', p1=self.patch, p2=self.patch)
        tokens = self.proj(tokens) + self.pos
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)                                          # (b, hp, wp, d)
        if self.output_mode == 'tokens':
            return tokens.reshape(tokens.shape[0], -1, self.d_model)
        return tokens.permute(0, 3, 1, 2).contiguous()                      # (b, d, hp, wp)


class _MixerBlock(nn.Module):
    """TSMixer block on (b, steps, channels): time mixing (a linear layer over the steps, shared across the
    channels) and feature mixing (a two-layer MLP over the channels, shared across the steps), each with
    LayerNorm, dropout and a residual."""

    def __init__(self, steps, channels, ff, dropout):
        super().__init__()
        self.norm_t = nn.LayerNorm(channels)
        self.time = nn.Sequential(nn.Linear(steps, steps), nn.ReLU(), nn.Dropout(dropout))
        self.norm_f = nn.LayerNorm(channels)
        self.feat = nn.Sequential(nn.Linear(channels, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, channels), nn.Dropout(dropout))

    def forward(self, x):
        y = self.time(self.norm_t(x).transpose(1, 2)).transpose(1, 2)
        x = x + y
        return x + self.feat(self.norm_f(x))


class tsmixer(nn.Module):
    """TSMixer with auxiliary information (Chen et al., 2023) for the ``lstm`` data: MLPs only, mixing
    alternately along time and along the features.

    Dictionary input.  The past block (past observations, ``past_data``, past calendar) is aligned to the
    horizon by a linear layer over its steps and projected to ``hidden`` (64) features; with ``use_nwp``
    the future block (future covariates and calendar) is projected to ``hidden`` features per future
    step and concatenated, so every future step carries its aligned history and its own covariates.
    ``depth`` (2) mixer blocks follow (time mixing over the future steps, feature mixing with a hidden
    width of ``ff_ratio`` (2) x channels), then a linear layer to ``size`` features per step.

    Output ``(b, horizon, size)`` (``output='sequence'``, default) or ``flat``.  A plain ``(b, T, F)``
    tensor is mixed over its own steps: ``(b, T, size)``.  The horizon is the future length of the data
    (``n_out`` for a plain tensor input).
    """

    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(tsmixer, self).__init__()
        self.shape = shape
        self.params = params
        self.name = f'{name_scope}_tsmixer_{layer_id}'
        self.use_nwp = bool(params.get('use_nwp', True))
        hidden = int(params.get('hidden', 64))
        depth = int(params.get('depth', 2))
        ff_ratio = float(params.get('ff_ratio', 2))
        dropout = float(params.get('dropout', params.get('attention_dropout', 0.1)))
        self.out_dim = max(1, int(size)) if isinstance(size, (int, float)) else 32
        self.output_mode = str(params.get('output', 'sequence'))
        if isinstance(shape, dict):
            past_obs, _, _, x_enc, x_dec, x_mark_enc, x_mark_dec, _, Tpred = init_lstm_data(shape, layer_id, False)
            past, future = self._split(past_obs, x_enc, x_dec, x_mark_enc, x_mark_dec)
            horizon = future.shape[1] if future is not None else \
                (x_mark_dec.shape[1] if x_mark_dec is not None else (x_dec.shape[1] if x_dec is not None else int(Tpred)))
            self.plain = False
        else:
            past, future = torch.zeros(shape), None
            horizon = int(shape[1])
            self.plain = True
        if past is None:
            raise ValueError('tsmixer needs past observations (past_obs / past_data)')
        self.horizon = int(horizon)
        L, F = past.shape[1], past.shape[2]
        self.align = nn.Linear(L, self.horizon) if not self.plain else None
        self.proj_past = nn.Linear(F, hidden)
        self.proj_future = nn.Linear(future.shape[2], hidden) if future is not None else None
        channels = hidden * (2 if future is not None else 1)
        self.blocks = nn.ModuleList([_MixerBlock(self.horizon, channels, int(channels * ff_ratio), dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, self.out_dim)
        self.apply(_init_weights)
        with torch.no_grad():
            self.output_shape = list(self._forward_parts(past, future).shape)

    def _split(self, past_obs, x_enc, x_dec, x_mark_enc, x_mark_dec):
        past = cat_without_none([past_obs, x_enc, x_mark_enc], -1)
        future = cat_without_none([x_dec, x_mark_dec], -1) if self.use_nwp else None
        return past, future

    def _forward_parts(self, past, future):
        b = past.shape[0]
        if self.align is not None:
            past = self.align(past.transpose(1, 2)).transpose(1, 2)      # (b, horizon, F): the history aligned to the future steps
        x = self.proj_past(past)
        if future is not None and self.proj_future is not None:
            x = torch.cat([x, self.proj_future(future)], dim=-1)
        for block in self.blocks:
            x = block(x)
        out = self.head(self.norm(x))                                     # (b, horizon, out_dim)
        return out.reshape(b, -1) if self.output_mode == 'flat' else out

    def forward(self, x):
        if isinstance(x, dict):
            past_obs, _, x_enc, x_dec, x_mark_enc, x_mark_dec = init_lstm_forward_data(x, None)
            past, future = self._split(past_obs, x_enc, x_dec, x_mark_enc, x_mark_dec)
            return self._forward_parts(past, future)
        return self._forward_parts(x, None)


class grid_transformer(nn.Module):
    """A small transformer over the NWP grid: one token per grid cell (or per patch of cells).

    Input ``(b, t, c, h, w)`` or ``(b, c, h, w)``.  Cells (patches of ``patch`` x ``patch`` cells,
        default 1) become tokens whose features are the variables and lags of the patch; a linear
    projection to ``d_model`` (``size`` when >= 8, else 64), a learned 2-D positional embedding,
    ``depth`` (2) pre-norm encoder layers with ``heads`` (4) heads and ``attention_dropout`` (0.1).
    Output: the token sequence ``(b, n_cells, d_model)``, for a Flatten, a dense per token or an
    attention merge.  Trained from scratch: pretrained image models bring little to 17x17
    physical fields; attention over the grid is the spatial mixing a wind or solar field needs.
    """

    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(grid_transformer, self).__init__()
        shape = list(shape)
        if len(shape) == 4:
            shape = [shape[0], 1, shape[1], shape[2], shape[3]]
        B, T, C, H, W = shape
        self.name = f'{name_scope}_grid_transformer_{layer_id}'
        heads = int(params.get('heads', 4))
        d_model = int(size) if isinstance(size, (int, float)) and size >= 8 else 64
        d_model = int(np.ceil(d_model / heads) * heads)
        self.patch = int(params.get('patch', 1))
        self.patch = max(1, min(self.patch, H, W))
        self.hp, self.wp = H // self.patch, W // self.patch
        self.crop = (self.hp * self.patch, self.wp * self.patch)
        n_tokens = self.hp * self.wp
        self.proj = nn.Linear(T * C * self.patch * self.patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        dropout = float(params.get('attention_dropout', 0.1))
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(params.get('depth', 2)))
        self.norm = nn.LayerNorm(d_model)
        self.apply(_init_weights)
        with torch.no_grad():
            self.output_shape = list(self.forward(torch.zeros(10, *shape[1:])).shape)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = x[..., :self.crop[0], :self.crop[1]]
        tokens = rearrange(x, 'b t c (hp p1) (wp p2) -> b (hp wp) (t c p1 p2)', p1=self.patch, p2=self.patch)
        tokens = self.proj(tokens) + self.pos
        return self.norm(self.encoder(tokens))


class RnnEmbedding(nn.Module):
    def __init__(self, shape, embed_size, layer_id, use_embedding):
        super().__init__()
        self.embed_size = embed_size
        self.use_embedding = use_embedding
        self.name_embed = f'RnnEmbedding{layer_id}'
        if self.use_embedding:
            self.rnn = nn.LSTM(shape[-1], embed_size, bias= False, batch_first=True)
            output, (hn, cn) = self.rnn(torch.zeros(shape))
            output = rearrange(output, 'b t c -> b c t')
            self.output_shape = output[:, :, -shape[-1]:].shape
        else:
            self.output_shape = torch.zeros([shape[0], embed_size, shape[-1]]).shape
        self.apply(_init_weights)


    def forward(self, x):
        shape = x.shape
        if self.use_embedding:
            output, (hn, cn) = self.rnn(x)
            output = rearrange(output, 'b t c -> b c t')
            return output[:, :, -shape[-1]:]
        else:
            return torch.zeros([shape[0], self.embed_size, shape[-1]])

class transformer(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(transformer, self).__init__()
        self.shape = shape
        self.size = size
        self.params = params
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_transformer_{layer_id}'
        name_trms = params['model_name']
        if isinstance(shape, dict):
            past_obs, _, _, x_enc, x_dec, x_mark_enc, x_mark_dec, B, Tpred = init_lstm_data(shape, layer_id, False)
            if x_enc is None:
                x_enc = past_obs
            else:
                try:
                    x_enc = cat_without_none([past_obs, x_enc], -1)
                except:
                    pass
            x_enc = cat_without_none([x_enc, x_mark_enc], -1)
            x_dec = cat_without_none([x_dec, x_mark_dec], -1)
            self.enc_in1 = x_enc.shape[-1]
            self.T1 = x_enc.shape[1]
        else:
            raise NotImplementedError(f"bi_transformer layer not implemented for input tensor. Only dict is supported")
        self.transformer = init_ts_model(name_trms, self.T1, params['n_out'], self.enc_in1, self.enc_in1,
                                          int(size * self.enc_in1), params)
        transformer_out = self.transformer(x_enc, x_mark_enc)
        self.output_shape = self.act_func(transformer_out).shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.transformer.weight.size())}")

    def forward(self, x):
        if isinstance(x, dict):
            past_obs, _, x_enc, x_dec, x_mark_enc, x_mark_dec = init_lstm_forward_data(x, None)
            if x_enc is None:
                x_enc = past_obs
            else:
                try:
                    x_enc = cat_without_none([past_obs, x_enc], -1)
                except:
                    pass
            x_enc = cat_without_none([x_enc, x_mark_enc], -1)
            x_dec = cat_without_none([x_dec, x_mark_dec], -1)
            B = x_enc.shape[0]
            device = x_enc.device
        else:
            raise NotImplementedError(f"bi_transformer layer not implemented for input tensor. Only dict is supported")

        transformer_out = self.act_func(self.transformer(x_enc, x_mark_enc))

        return transformer_out

class bi_lstm(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(bi_lstm, self).__init__()
        self.shape = shape
        self.size = size
        self.params = params
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_lstm_{layer_id}'
        if isinstance(shape, dict):
            past_obs, _, _, x_enc, x_dec, x_mark_enc, x_mark_dec, B, Tpred = init_lstm_data(shape, layer_id, False)
            if x_enc is None:
                x_enc = past_obs
            else:
                try:
                    x_enc = cat_without_none([past_obs, x_enc], -1)
                except:
                    pass
            if not self.params['use_nwp']:
                x_dec = past_obs
                x_mark_dec = x_mark_enc
            # x_enc = cat_without_none([x_enc, x_mark_enc], -1)
            # x_dec = cat_without_none([x_dec, x_mark_dec], -1)
            self.enc_in1 = x_enc.shape[-1]
            self.enc_in2 = x_dec.shape[-1]
            self.T1 = x_enc.shape[1]
            self.T2 = x_dec.shape[1]
        else:
            self.shape = list(shape[:2]) + [shape[2] * shape[3] * shape[4]]
            B = self.shape[0]
            self.enc_in1 = self.shape[-1]
            self.enc_in2 = self.shape[-1]
            size = 1 / size
            self.size = size
            x_enc = torch.zeros(self.shape)
            x_dec = torch.zeros(self.shape)
        self.lstm1 = nn.LSTM(self.enc_in1, int(size * self.enc_in1), batch_first=True)
        self.lstm2 = nn.LSTM(self.enc_in2, int(size * self.enc_in2), batch_first=True)
        hidden_state1 = (torch.zeros(1, B, int(size * self.enc_in1)), torch.zeros(1, B, int(size * self.enc_in1)))
        hidden_state2 = (torch.zeros(1, B, int(size * self.enc_in2)), torch.zeros(1, B, int(size * self.enc_in2)))
        lstm_out1 = self.lstm1(x_enc, hidden_state1)[0]
        lstm_out2 = self.lstm2(x_dec, hidden_state2)[0]
        self.T3 = 4
        if self.T3 > lstm_out1.shape[1] + lstm_out2.shape[1]:
            raise ValueError(f"keep_dim {self.T3} is greater than the sum of the output dimensions "
                             f"of the two LSTMs {lstm_out1.shape[1] + lstm_out2.shape[1]}")
        lstm_out = torch.cat([lstm_out1[:, -self.T3:, :], lstm_out2[:, -self.T3:, :]], -1)
        self.d_model3 = int(size * (self.enc_in1 + self.enc_in2))
        if self.T3 > 1:
            self.lstm_in3 = lstm_out.shape[-1]
            self.lstm3 = nn.LSTM(self.lstm_in3, self.d_model3, batch_first=True, bidirectional=True)
            hidden_state3 = (torch.zeros(2, B, self.d_model3), torch.zeros(2, B, self.d_model3))
            lstm_out3 = self.lstm3(lstm_out, hidden_state3)[0]
        else:
            self.lstm_in3 = lstm_out.shape[-1]
            self.lstm3 = nn.Linear(self.lstm_in3, self.d_model3, bias=True)
            lstm_out3 = self.lstm3(lstm_out)
        self.output_shape = self.act_func(lstm_out3).shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.lstm.weight.size())}")

    def forward(self, x):
        if isinstance(x, dict):
            past_obs, _, x_enc, x_dec, x_mark_enc, x_mark_dec = init_lstm_forward_data(x, None)
            if x_enc is None:
                x_enc = past_obs
            else:
                try:
                    x_enc = cat_without_none([past_obs, x_enc], -1)
                except:
                    pass
            if not self.params['use_nwp']:
                x_dec = past_obs
                x_mark_dec = x_mark_enc
            # x_enc = cat_without_none([x_enc, x_mark_enc], -1)
            # x_dec = cat_without_none([x_dec, x_mark_dec], -1)
            B = x_enc.shape[0]
            device = x_enc.device
        else:
            device = x.device
            B = x.shape[0]
            x = rearrange(x, 'b c t l1 l2 -> b c (t l1 l2)')
            x_enc = x
            x_dec = x.flip(1)
        hidden_state1 = (torch.zeros(1, B, int(self.size * self.enc_in1)).to(device),
                         torch.zeros(1, B, int(self.size * self.enc_in1)).to(device))
        hidden_state2 = (torch.zeros(1, B, int(self.size * self.enc_in2)).to(device),
                         torch.zeros(1, B, int(self.size * self.enc_in2)).to(device))
        lstm_out1 = self.lstm1(x_enc, hidden_state1)[0]
        lstm_out2 = self.lstm2(x_dec, hidden_state2)[0]
        lstm_out = torch.cat([lstm_out1[:, -self.T3:, :], lstm_out2[:, -self.T3:, :].flip(1)], -1)
        if self.T3 > 1:
            hidden_state3 = (torch.zeros(2, B, self.d_model3).to(device),
                             torch.zeros(2, B, self.d_model3).to(device))
            lstm_out3 = self.lstm3(lstm_out, hidden_state3)[0]
        else:
            lstm_out3 = self.lstm3(lstm_out)
        return self.act_func(lstm_out3)

class bi_spacetimeformer(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(bi_spacetimeformer, self).__init__()
        from eforecast.deep_models.pytorch_2x.STFormer.spacetimeformer.spacetimeformer_model.nn.model_ver2 import \
            Spacetimeformer_Forecaster_ver2
        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_transformer_{layer_id}'
        if isinstance(shape, dict):
            past_obs, _, _, x_enc, x_dec, x_mark_enc, x_mark_dec, B, Tpred = init_lstm_data(shape, layer_id, False)
            # x_enc = cat_without_none([x_enc, x_mark_enc], -1)
            if x_enc is None:
                x_enc = past_obs
            else:
                try:
                    x_enc = cat_without_none([past_obs, x_enc], -1)
                except:
                    pass
            # x_dec = cat_without_none([x_dec, x_mark_dec], -1)
            self.enc_in1 = x_enc.shape[-1]
            self.enc_in2 = x_dec.shape[-1]
            self.T1 = x_enc.shape[1]
            self.T2 = x_dec.shape[1]
            self.spacetimeformer = Spacetimeformer_Forecaster_ver2(self.enc_in1, self.T1, self.enc_in2, self.T2,
                                                                   x_mark_enc.shape[-1], x_mark_enc.shape[-1],
                                                                   params['n_out'])
            out1 = self.spacetimeformer(x_mark_enc, x_enc, x_mark_dec, x_dec)

            self.output_shape = out1.shape
        else:

            self.shape = [self.shape[0], 3, 5 * 25]
            self.enc_in1 = self.shape[2]
            self.enc_in2 = self.shape[2]
            self.T1 = self.shape[1]
            self.T2 = self.shape[1]
            self.spacetimeformer = Spacetimeformer_Forecaster_ver2(d_yc=self.enc_in1, d_yt=self.T1, d_yc_dec=self.enc_in2, d_yt_dec=self.T2, d_x=0, d_x_dec=0, out_dim=params['n_out'], d_model=4)
            x_enc = torch.zeros(self.shape)
            x_dec = torch.zeros(self.shape)
            out1 = self.spacetimeformer(None, x_enc, None, x_dec)

            self.output_shape = out1.shape
        # self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights transformer1 {self.name} has shape {list(self.transformer1.weight.size())}")
        print(f"layer weights transformer2 {self.name} has shape {list(self.transformer2.weight.size())}")

    def forward(self, x):
        if isinstance(x, dict):
            past_obs, _, x_enc, x_dec, x_mark_enc, x_mark_dec = init_lstm_forward_data(x, None)
            # x_enc = cat_without_none([x_enc, x_mark_enc], -1)
            if x_enc is None:
                x_enc = past_obs
            else:
                try:
                    x_enc = cat_without_none([past_obs, x_enc], -1)
                except:
                    pass
            # x_dec = cat_without_none([x_dec, x_mark_dec], -1)
            B = x_enc.shape[0]
            device = x_enc.device
            out1 = self.spacetimeformer(x_mark_enc, x_enc, x_mark_dec, x_dec)
        else:

            x = x[:, :, :, 6:11, 6:11]
            x = rearrange(x, 'b c t l1 l2 -> b c (t l1 l2)')
            x_enc = x
            x_dec = x.flip(1)
            out1 = self.spacetimeformer(None, x_enc, None, x_dec)

        return out1


class bi_transformer(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(bi_transformer, self).__init__()
        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_transformer_{layer_id}'
        name_trms = params['model_name']
        if isinstance(shape, dict):
            past_obs, _, _, x_enc, x_dec, x_mark_enc, x_mark_dec, B, Tpred = init_lstm_data(shape, layer_id, False)
            # x_enc = cat_without_none([x_enc, x_mark_enc], -1)
            if x_enc is None:
                x_enc = past_obs
            else:
                try:
                    x_enc = cat_without_none([past_obs, x_enc], -1)
                except:
                    pass
            # x_dec = cat_without_none([x_dec, x_mark_dec], -1)
            self.enc_in1 = x_enc.shape[-1]
            self.enc_in2 = x_dec.shape[-1]
            self.T1 = x_enc.shape[1]
            self.T2 = x_dec.shape[1]
        else:
            raise NotImplementedError(f"bi_transformer layer not implemented for input tensor. Only dict is supported")
        self.transformer1 = init_ts_model(name_trms, self.T1, params['n_out'], self.enc_in1, self.enc_in1, int(size * self.enc_in1), params)

        self.transformer2 = init_ts_model(name_trms, self.T2, params['n_out'], self.enc_in2, self.enc_in2, int(size * self.enc_in1), params)

        transformer_out1 = self.act_func(self.transformer1(x_enc, x_mark_enc))
        transformer_out2 = self.act_func(self.transformer2(x_dec, x_mark_dec))
        transformer_out = torch.cat([transformer_out1, transformer_out2], -1)
        self.transformer_T3 = transformer_out.shape[1]
        self.transformer_in3 = transformer_out.shape[-1]
        transformer_out = rearrange(transformer_out, 'B T C -> B (T C)')
        self.transformer3 = nn.Linear((self.transformer_T3 * self.transformer_in3), self.transformer_in3, bias=True)
        transformer_out3 = self.transformer3(transformer_out)
        self.output_shape = self.act_func(transformer_out3).shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights transformer1 {self.name} has shape {list(self.transformer1.weight.size())}")
        print(f"layer weights transformer2 {self.name} has shape {list(self.transformer2.weight.size())}")

    def forward(self, x):
        if isinstance(x, dict):
            past_obs, _, x_enc, x_dec, x_mark_enc, x_mark_dec = init_lstm_forward_data(x, None)
            # x_enc = cat_without_none([x_enc, x_mark_enc], -1)
            if x_enc is None:
                x_enc = past_obs
            else:
                try:
                    x_enc = cat_without_none([past_obs, x_enc], -1)
                except:
                    pass
            # x_dec = cat_without_none([x_dec, x_mark_dec], -1)
            B = x_enc.shape[0]
            device = x_enc.device
        else:
            raise NotImplementedError(f"bi_transformer layer not implemented for input tensor. Only dict is supported")

        transformer_out1 = self.act_func(self.transformer1(x_enc, x_mark_enc))
        transformer_out2 = self.act_func(self.transformer2(x_dec, x_mark_dec))
        transformer_out = torch.cat([transformer_out1, transformer_out2], -1)
        transformer_out = rearrange(transformer_out, 'B T C -> B (T C)')
        transformer_out3 = self.transformer3(transformer_out)
        return self.act_func(transformer_out3)


class GatedFusion(nn.Module):
    """The former ``cross_attention`` merge, kept under its real name.

    Every branch arrives as one flat vector per sample, so there is no set of positions to attend
    over.  The layer projects the target vector to q and the source vector to k and v, multiplies q
    and k coordinate-wise, takes the softmax across the coordinates of each head (not across
    positions) and multiplies v coordinate-wise.  That is a multiplicative gate on a projection of
    the source, conditioned on the target: the same family as GLU / squeeze-and-excitation / FiLM.
    It trains fine as a fusion layer, but it has no attention map and the softmax across features
    shrinks the value by the head size.  ``cross_attention`` is now the token-level ``CrossAttention``
    below; use ``gated_fusion`` in an experiment to get this layer.
    """

    def __init__(
        self, shape, params, size, name_scope, layer_id, train=True,
        heads=4,
        dim_head=128,
        dropout=0.0,
    ):
        super(GatedFusion, self).__init__()
        if not isinstance(shape, dict):
            raise ValueError('shape must be dict of names of branches that should be cross-attention')
        if len(shape) > 2:
            tags = [k for k in shape.keys()]
            shape1 = {n:v for n, v in shape.items() if n != tags[0]}
            self.concat_layer = Concat_(shape1, layer_id)
            shape = {tags[0]: shape[tags[0]], '_'.join(tags[1:]): self.concat_layer .output_shape}
        names_branch = [k for k in shape.keys()]
        if len(shape) != 2:
            self.output_shape = shape[names_branch[0]]
            return
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        names_branch = [k for k in shape.keys()]
        self.to_q = nn.Linear(shape[names_branch[1]][1], inner_dim, bias=False)

        self.to_kv = nn.Linear(shape[names_branch[0]][1], inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, shape[names_branch[0]][1]), nn.Dropout(dropout))
        self.apply(_init_weights)
        self.output_shape = self.forward([torch.zeros(10, shape[names_branch[0]][1]),
                                          torch.zeros(10, shape[names_branch[1]][1])]).shape

    def forward(self, x_list):
        if not isinstance(x_list, list):
            return x_list
        if len(x_list) == 1:
            return x_list[0]
        if len(x_list) > 2:
            x1 = self.concat_layer(x_list[1:])
            x = [x_list[0], x1]
        else:
            x = x_list
        src, tgt = x
        q = self.to_q(tgt)

        qkv = (q, *self.to_kv(src).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, "b (h d) -> (b h) d", h=self.heads), qkv
        )

        dots = einsum(q, k, "b i, b i -> b i") * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum(attn, v, "b j, b j -> b j")
        out = rearrange(out, "(b h) d -> b (h d)", h=self.heads)
        return self.to_out(out)


class GatedFusionBoth(nn.Module):
    """Two ``GatedFusion`` gates, one in each direction, outputs concatenated (former ``cross_attention_both``)."""

    def __init__(
        self, shape, params, size, name_scope, layer_id, train=True,
        heads=4,
        dim_head=128,
        dropout=0.0,
    ):
        super(GatedFusionBoth, self).__init__()
        if not isinstance(shape, dict):
            raise ValueError('shape must be dict of names of branches that should be cross-attention')
        if len(shape) > 2:
            tags = [k for k in shape.keys()]
            shape1 = {n:v for n, v in shape.items() if n != tags[0]}
            self.concat_layer = Concat_(shape1, layer_id)
            shape = {tags[0]: shape[tags[0]], '_'.join(tags[1:]): self.concat_layer .output_shape}
        names_branch = [k for k in shape.keys()]
        if len(shape) != 2:
            self.output_shape = shape[names_branch[0]]
            return
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q_f = nn.Linear(shape[names_branch[1]][1], inner_dim, bias=False)
        self.to_q_b = nn.Linear(shape[names_branch[0]][1], inner_dim, bias=False)

        self.to_kv_f = nn.Linear(shape[names_branch[0]][1], inner_dim * 2, bias=False)
        self.to_kv_b = nn.Linear(shape[names_branch[1]][1], inner_dim * 2, bias=False)

        self.to_out_f = nn.Sequential(nn.Linear(inner_dim, shape[names_branch[0]][1]), nn.Dropout(dropout))
        self.to_out_b = nn.Sequential(nn.Linear(inner_dim, shape[names_branch[1]][1]), nn.Dropout(dropout))
        self.apply(_init_weights)
        self.output_shape = self.forward([torch.zeros(10, shape[names_branch[0]][1]),
                                          torch.zeros(10, shape[names_branch[1]][1])]).shape

    def forward(self, x_list):
        if not isinstance(x_list, list):
            return x_list
        if len(x_list) == 1:
            return x_list[0]
        if len(x_list) > 2:
            x1 = self.concat_layer(x_list[1:])
            x = [x_list[0], x1]
        else:
            x = x_list
        src_f, tgt_f = x
        q_f = self.to_q_f(tgt_f)

        qkv_f = (q_f, *self.to_kv_f(src_f).chunk(2, dim=-1))

        q_f, k_f, v_f = map(
            lambda t: rearrange(t, "b (h d) -> (b h) d", h=self.heads), qkv_f
        )

        dots_f = einsum(q_f, k_f, "b i, b i -> b i") * self.scale

        attn_f = self.attend(dots_f)
        attn_f = self.dropout(attn_f)

        out_f = einsum(attn_f, v_f, "b j, b j -> b j")
        out_f = rearrange(out_f, "(b h) d -> b (h d)", h=self.heads)
        out_f = self.to_out_f(out_f)

        src_b, tgt_b = tgt_f, src_f
        q_b = self.to_q_b(tgt_b)

        qkv_b = (q_b, *self.to_kv_b(src_b).chunk(2, dim=-1))

        q_b, k_b, v_b = map(
            lambda t: rearrange(t, "b (h d) -> (b h) d", h=self.heads), qkv_b
        )

        dots_b = einsum(q_b, k_b, "b i, b i -> b i") * self.scale

        attn_b = self.attend(dots_b)
        attn_b = self.dropout(attn_b)

        out_b = einsum(attn_b, v_b, "b j, b j -> b j")
        out_b = rearrange(out_b, "(b h) d -> b (h d)", h=self.heads)
        out_b = self.to_out_b(out_b)
        return torch.cat([out_f, out_b], -1)


class unet(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(unet, self).__init__()
        import torchvision.transforms as transforms

        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_{params["unet_type"]}_{layer_id}'
        import segmentation_models_pytorch as smp      # was never imported: the layer could not be built
        arch = getattr(smp, params["unet_type"])

        self.resize = transforms.Resize((128, 128), antialias=False)
        if params['pretrained'] in {'tune', 'finetune'}:
            is_train = True
        else:
            is_train = False
        if is_train:
            from eforecast.model_library import smp_model
            self.unet_model = smp_model(arch, params['encoder_name'], shape[1],
                                       revision=params.get('model_revision'), encoder_depth=3,
                                       decoder_channels=(64, 32, 16), classes=1)
        else:
            self.unet_model = arch(encoder_name=params['encoder_name'], in_channels=shape[1],
                                   encoder_depth=3, decoder_channels=(64, 32, 16),
                                   encoder_weights=None, classes=1)
        if params['pretrained'] == 'finetune':
            self.unet_model.requires_grad_(False)      # freeze the pretrained network (the attribute form did nothing)
        self.flat = nn.Flatten()


    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")

    def forward(self, x):
        if self.resize is not None:
            x = self.resize(x)
        return self.act_func(self.flat(self.unet_model(x)))

class vit_net(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(vit_net, self).__init__()
        import timm
        import torchvision.transforms as transforms
        from transformers import AutoImageProcessor, AutoModel, AutoConfig

        self.shape = shape
        if len(self.shape) == 5:
            B, T, C, H, W = self.shape
            C = int(T * C)
        elif len(self.shape) == 4:
            B, C, H, W = self.shape
        else:
            raise ValueError(f'Wrong input shape given dimension {len(self.shape)} expected 4 or 5')
        x_sample = torch.zeros([B, C, H, W])
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_{params["model_name"]}_{layer_id}'
        self.model_name = params["model_name"]
        try:
            s = int(params["model_name"].split('_')[-1])
        except:
            s =  H
        if s > 600:
            raise ValueError('Too Large input size')
        self.img_size = s
        from eforecast import model_library
        # Is this a timm architecture, or a Hugging Face model? The per-project configurations
        # carried a list of timm names in the family's parameters (``timm_models``); the presets
        # of this pipeline never did, so the lookup stopped every graph with this layer on
        # KeyError 'timm_models'. A list that is given is still honoured; otherwise timm is asked.
        listed = params.get('timm_models')
        if (self.model_name in listed if listed is not None else timm.is_model(self.model_name)) \
                or 'timm' in self.model_name:
            try:
                self.vit_model = model_library.timm_model(self.model_name, revision=params.get('model_revision'),
                                                    num_classes=0, global_pool='', in_chans=C)
            except (model_library.ModelPrerequisite, OSError, MemoryError, torch.OutOfMemoryError):
                raise
            except Exception:
                try:
                    self.vit_model = model_library.timm_model(self.model_name, revision=params.get('model_revision'),
                                                        num_classes=self.img_size * C, in_chans=C)
                except (model_library.ModelPrerequisite, OSError, MemoryError, torch.OutOfMemoryError):
                    raise
                except Exception:
                    self.vit_model = timm.create_model(self.model_name, pretrained=False,
                                                       num_classes=self.img_size * C, in_chans=C)
            data_config = timm.data.resolve_model_data_config(self.vit_model)
            self.resize = transforms.Resize(data_config['input_size'][1:], antialias=False)

            img = self.resize(x_sample)
            outputs = self.vit_model(img)
            if len(outputs.shape) == 2:
                outputs = rearrange(outputs, 'b (c h) -> b c h', c=self.img_size)
            output_shape = outputs.shape
        else:
            ref = model_library.resolve(self.model_name, revision=params.get('model_revision'),
                                        profile='vision', preprocessing={'channels': C, 'height': H, 'width': W})
            with model_library.loading(ref):
                try:
                    self.processor = AutoImageProcessor.from_pretrained(ref.path, local_files_only=True)
                    self.vit_model = AutoModel.from_pretrained(ref.path, local_files_only=True)
                    inputs = self.processor(images=x_sample, return_tensors="pt", input_data_format='channels_first',
                                            do_rescale=False)
                    outputs = self.vit_model(**inputs)
                except (OSError, MemoryError, torch.OutOfMemoryError):
                    raise
                except Exception:
                    # Keep the existing channel/architecture adaptation, using the
                    # SAME verified configuration; a cache miss cannot reach it.
                    self.config = AutoConfig.from_pretrained(ref.path, local_files_only=True)
                    self.config.num_channels = C
                    if hasattr(self.config, 'image_size'):
                        self.config.image_size = H if isinstance(self.config.image_size, int) else [H, W]
                    self.vit_model = AutoModel.from_config(self.config)
                    outputs = self.vit_model(x_sample)
            output_shape = outputs.last_hidden_state.shape
            outputs = outputs.last_hidden_state
        self.flat = nn.Flatten()
        self.pool = nn.AvgPool1d(2)
        self.output_shape = self.pool(self.flat(outputs)).shape
        self.nan_reported = False
        if params['pretrained'] == 'finetune':
            self.vit_model.requires_grad_(False)       # freeze the pretrained backbone (the attribute form did nothing)


    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer {self.name} has output_shape {self.output_shape}")

    def forward(self, x):
        if isinstance(x, dict):
            x = x['images']
        if len(x.size()) == 5:
            x = rearrange(x, 'b t c h w -> b (t c) h w')
        if hasattr(self, 'resize'):
            output = self.vit_model(self.resize(x))
            if len(output.shape) == 2:
                output = rearrange(output, 'b (c h) -> b c h', c=self.img_size)
        elif hasattr(self, 'config'):
            output = self.vit_model(x).last_hidden_state
        else:
            inputs = self.processor(images=x, return_tensors="pt", input_data_format='channels_first',
                                    do_rescale=False).to(x.device)
            output = self.vit_model(**inputs).last_hidden_state
        output = self.act_func(self.pool(self.flat(output)))
        if torch.isnan(output).any():
            # the former version re-created the pretrained model here, in the middle of training
            if not self.nan_reported:
                print(f'{self.name}: NaN in the output; check the image normalisation (values replaced by 0)')
                self.nan_reported = True
            output = torch.nan_to_num(output, nan=0.0)
        return output


class time_distr_vit_net(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(time_distr_vit_net, self).__init__()
        self.shape = shape
        B, T, C, H, W = self.shape
        self.size = size
        self.act_func = act_funcs['sigmoid']
        self.name = f'{name_scope}_time_distr_vit_net_{layer_id}'
        self.vit_list = nn.ModuleList()
        for i in range(T):
            self.vit_list.append(vit_net([B, C, H, W], params, size, self.name, i))
        self.lstm = nn.LSTM(int(self.vit_list[0].output_shape[-1]), int(self.vit_list[0].output_shape[-1]),
                            batch_first=True)
        self.flat = nn.Flatten()
        x = torch.zeros([10, T, C, H, W])
        time_steps = x.size()[1]
        output = torch.tensor([])
        for i in range(time_steps):
            output_t = self.vit_list[i](x[:, i, ...])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        hidden_state = (torch.zeros(1, B, output.shape[-1]), torch.zeros(1, B, output.shape[-1]))
        lstm_out = self.lstm(output, hidden_state)[0]
        self.output_shape = self.act_func(self.flat(lstm_out)).shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(
            f"layer weights {self.name} has {self.shape[1]} conv of shape{list(self.conv_list[0].weight.size())}")
        print(f"layer {self.name} has output_shape {self.output_shape}")

    def forward(self, x):
        if isinstance(x, dict):
            x = x['images']
        time_steps = x.size()[1]
        B = x.size()[0]
        output = torch.tensor([]).to(x.device)
        for i in range(time_steps):
            output_t = self.vit_list[i](x[:, i, ...])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        hidden_state = (torch.zeros(1, B, output.shape[-1]).to(x.device),
                        torch.zeros(1, B, output.shape[-1]).to(x.device))
        lstm_out = self.lstm(output, hidden_state)[0]
        return self.act_func(self.flat(lstm_out))

class dense(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(dense, self).__init__()
        self.shape = shape
        self.size = size
        if params['act_func'] is not None:
            self.act_func = act_funcs[params['act_func']]
        else:
            self.act_func = None
        self.name = f'{name_scope}_dense_{layer_id}'
        # acts on the last axis: a flat vector (b, f) or a token sequence (b, n, f) alike
        self.dense = nn.Linear(shape[-1], int(size))
        self.output_shape = self.dense(torch.zeros([10, *self.shape[1:]])).shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.dense.weight.size())}")

    def forward(self, x):
        if self.act_func is not None:
            return self.act_func(self.dense(x))
        else:
            return self.dense(x)


class Reshape(nn.Module):
    def __init__(self, shape, layer_id):
        super(Reshape, self).__init__()
        self.shape = shape
        self.name = f'Reshape_{layer_id}'
        self.output_shape = [10] + self.shape

    def forward(self, x):
        if (self.shape[0] * self.shape[1]) != x.shape[1]:
            x = x[:, :int(self.shape[0] * self.shape[1])]
        return rearrange(x, 'b (c h) -> b c h', c=self.shape[0], h=self.shape[1])


class Concat_(nn.Module):
    """Concatenation merge: every branch is flattened to (batch, features) first, whatever its rank."""

    def __init__(self, shape, layer_id):
        super(Concat_, self).__init__()
        self.shape = shape
        shapes = [list(s) for s in shape.values()] if isinstance(shape, dict) else [list(shape)]
        self.output_shape = [shapes[0][0], int(sum(int(np.prod(s[1:])) for s in shapes))]
        self.name = f'Concat_{layer_id}'

    def forward(self, x):
        if isinstance(x, list):
            return torch.cat([xi.flatten(start_dim=1) for xi in x], -1)
        return x.flatten(start_dim=1)


class Clip(nn.Module):
    """CLIP-style alignment of the branches that enter the final merge.

    Each branch output is flattened and projected by its own linear head to ``clip_dim`` (params
    ``clip_dim``, default 128), L2-normalised, and the pairwise cosine-similarity matrices of the
    batch are returned, scaled by a learned temperature.  The trainer turns them into a loss whose
    positives are the same sample seen by two branches; samples closer in time than
    ``clip_neighbour_window`` are excluded from the negatives (see ``trainer.compute_loss``).
    """

    def __init__(self, shape, params=None, layer_id=0):
        super(Clip, self).__init__()
        params = params or {}
        self.name = f'Clip_{layer_id}'
        self.shape = shape
        if not isinstance(shape, dict) or len(shape) < 2:
            self.names, self.proj = [], None
            self.output_shape = shape
            return
        dim = int(params.get('clip_dim', 128))
        self.names = list(shape.keys())
        self.proj = nn.ModuleDict({self._key(n): nn.Linear(int(np.prod(list(s)[1:])), dim) for n, s in shape.items()})
        # Learnable log-temperature initialised to ln(1/0.07) (CLIP default)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.07)))
        self.output_shape = shape
        self.apply(_init_weights)

    @staticmethod
    def _key(name):
        return name.replace('/', '__').replace('.', '_')

    def forward(self, x):
        if self.proj is None or not isinstance(x, list):
            return []
        embeds = [F.normalize(self.proj[self._key(n)](xi.flatten(start_dim=1)), dim=-1) for n, xi in zip(self.names, x)]
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return [logit_scale * embeds[i] @ embeds[j].T for i in range(len(embeds)) for j in range(i + 1, len(embeds))]


def token_view(shape):
    """``(n_tokens, n_features)`` of a branch output seen as a token sequence.

    (b, f)          -> 1 token of f features            a flat vector, e.g. a dense branch
    (b, n, f)       -> n tokens of f features           a sequence, e.g. LSTM / transformer steps
    (b, c, h, w)    -> h*w tokens of c features         a feature map: one token per grid cell
    (b, t, c, h, w) -> h*w tokens of t*c features       NWP: one token per grid cell, lags and variables as features
    """
    shape = list(shape)
    if len(shape) == 2:
        return 1, int(shape[1])
    if len(shape) == 3:
        return int(shape[1]), int(shape[2])
    if len(shape) == 4:
        return int(shape[2] * shape[3]), int(shape[1])
    if len(shape) == 5:
        return int(shape[3] * shape[4]), int(shape[1] * shape[2])
    raise ValueError(f'cannot see a tensor of shape {shape} as tokens')


def to_tokens(x):
    """The token view of a tensor: (batch, n_tokens, n_features), see ``token_view``."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    if x.dim() == 4:
        return rearrange(x, 'b c h w -> b (h w) c')
    if x.dim() == 5:
        return rearrange(x, 'b t c h w -> b (h w) (t c)')
    raise ValueError(f'cannot see a tensor of rank {x.dim()} as tokens')


class Tokens(nn.Module):
    """Keeps the positions of a branch alive for an attention merge (the counterpart of Flatten)."""

    def __init__(self, shape, layer_id):
        super(Tokens, self).__init__()
        n_tokens, n_features = token_view(shape)
        self.name = f'Tokens_{layer_id}'
        self.output_shape = [10, n_tokens, n_features]

    def forward(self, x):
        return to_tokens(x)


def named_branches(wanted):
    """The branch names an ``attention_query`` holds: one name or a list of names; ``None`` if none.

    A value that is no name -- a number, a yes or no, a list without text in it -- names no
    branch and is read as a parameter that is not set. Stored cases carry such values: until
    November 2026 the graph editor started this parameter at the number 64 when it was added,
    and offered no way to enter a list.
    """
    if isinstance(wanted, str):
        wanted = [wanted]
    if not isinstance(wanted, (list, tuple, set, frozenset)):
        return None
    return [name for name in wanted if isinstance(name, str) and name.strip()] or None


class CrossAttention(nn.Module):
    """Token-level cross attention between the branches of a merge.

    Every branch is seen as a token sequence (``token_view``), projected to a common width
    ``d_model`` and given a learned branch embedding and, for sequences, a learned positional
    embedding.  The flat branches (one token each: row data, calendar, a dense branch) are the
    queries; the sequence branches (NWP grid cells, LSTM or transformer steps) are the keys and
    values.  Multi-head attention with softmax over the key positions answers "which lead times,
    grid cells or past steps matter for this sample"; the residual keeps the query itself, and a
    small feed-forward block follows.  When no branch is flat, a learned query token pools the
    sequences; when every branch is flat, the learned query attends over the set of branches, a
    per-sample weighting of the modalities.  Output: the attended query tokens flattened,
    (batch, n_queries * d_model), so dense layers can follow.

    ``both=True`` (``cross_attention_both``) adds the opposite direction: a learned token pools the
    sequence branches and the result is concatenated, (batch, n_queries * d_model + d_model).

    Parameters read from ``params``: ``heads`` (4), ``attention_dropout`` (0.1), ``d_model`` (else
    ``size`` when >= 8, else 64), ``attention_query`` (the name, or the names, of the branches to
    use as queries; a value that names none is read as not set, see ``named_branches``).
    Attention maps of the last forward are kept in ``self.last_attention`` (batch, n_q, n_kv).
    """

    def __init__(self, shape, params, size, name_scope, layer_id, train=True, both=False):
        super(CrossAttention, self).__init__()
        params = params or {}
        if not isinstance(shape, dict):
            shape = {'input': list(shape)}
        self.names = list(shape.keys())
        self.both = both
        self.name = f'{name_scope}_cross_attention_{layer_id}'
        heads = int(params.get('heads', 4))
        d_model = params.get('d_model')
        if d_model is None:
            d_model = int(size) if isinstance(size, (int, float)) and size >= 8 else 64
        d_model = int(np.ceil(d_model / heads) * heads)
        self.d_model = d_model
        dropout = float(params.get('attention_dropout', 0.1))
        self.views = {name: token_view(s) for name, s in shape.items()}
        wanted = params.get('attention_query')
        asked = named_branches(wanted)
        if asked:
            self.query_names = [n for n in self.names if n in asked or n.split('/')[-1] in asked]
        else:
            if wanted:
                print(f'{self.name}: attention_query is {wanted!r}, which names no branch; '
                      'it is read as not set and the flat branches are the queries')
            self.query_names = [n for n in self.names if self.views[n][0] == 1]
        if len(self.query_names) == len(self.names) or not self.query_names:
            self.kv_names = list(self.names)          # all flat, or all sequences: the learned query attends everything
        else:
            self.kv_names = [n for n in self.names if n not in self.query_names]
        self.use_learned_query = not self.query_names or len(self.query_names) == len(self.names)
        self.proj = nn.ModuleDict({self._key(n): nn.Linear(f, d_model) for n, (_, f) in self.views.items()})
        self.branch_embed = nn.ParameterDict({self._key(n): nn.Parameter(torch.zeros(1, 1, d_model)) for n in self.names})
        self.pos_embed = nn.ParameterDict({self._key(n): nn.Parameter(torch.zeros(1, t, d_model))
                                           for n, (t, _) in self.views.items() if t > 1})
        for p in list(self.branch_embed.values()) + list(self.pos_embed.values()):
            nn.init.trunc_normal_(p, std=0.02)
        if self.use_learned_query:
            self.learned_query = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.learned_query, std=0.02)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        if both:
            self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            self.attn_back = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.last_attention = None
        self.apply(_init_weights)
        with torch.no_grad():
            sample = [torch.zeros(10, *list(shape[n])[1:]) for n in self.names]
            self.output_shape = list(self.forward(sample if len(sample) > 1 else sample[0]).shape)

    @staticmethod
    def _key(name):
        return name.replace('/', '__').replace('.', '_')

    def _embed(self, x, name):
        tokens = self.proj[self._key(name)](to_tokens(x)) + self.branch_embed[self._key(name)]
        if self._key(name) in self.pos_embed:
            tokens = tokens + self.pos_embed[self._key(name)][:, :tokens.shape[1]]
        return tokens

    def forward(self, x_list):
        if not isinstance(x_list, list):
            x_list = [x_list]
        branches = dict(zip(self.names, x_list))
        kv = torch.cat([self._embed(branches[n], n) for n in self.kv_names], 1)
        if self.use_learned_query:
            queries = self.learned_query.expand(kv.shape[0], -1, -1)
        else:
            queries = torch.cat([self._embed(branches[n], n) for n in self.query_names], 1)
        attended, weights = self.attn(self.norm_q(queries), self.norm_kv(kv), self.norm_kv(kv), need_weights=True)
        self.last_attention = weights.detach()
        out = queries + attended
        out = out + self.ffn(out)
        out = out.flatten(start_dim=1)
        if self.both:
            pooled, _ = self.attn_back(self.pool_query.expand(kv.shape[0], -1, -1), self.norm_kv(kv),
                                       self.norm_kv(kv), need_weights=False)
            out = torch.cat([out, pooled.flatten(start_dim=1)], -1)
        return out


class PerceiverMerge(nn.Module):
    """Perceiver-style merge with horizon queries.

    Every branch is seen as a token sequence (``token_view``: grid cells, past or future steps, one
    token for a flat branch), projected to ``d_model`` with a learned branch embedding and, for
    sequences, a learned positional embedding.  A small array of ``latents`` (16) learned tokens then
    runs ``depth`` (2) rounds of: cross-attention over all modality tokens, self-attention among the
    latents, feed-forward (pre-norm).  Finally ``horizon_queries`` learned queries (default ``n_out``,
    one per horizon step) attend over the latents: each lead time reads its own mix of NWP cells,
    past steps and calendar, without separate heads.

    Output (``output``): ``flat`` (default) the decoded step tokens flattened, ``(b, n_queries * d_model)``,
    so dense layers and the head can follow; ``steps`` a direct per-step scalar readout ``(b, n_queries)``
    (a shared linear layer plus one bias per step).  ``last_attention`` holds the horizon-to-latent
    weights of the decoder ``(b, n_queries, latents)`` and ``last_cross_attention`` the latent-to-token
    weights of the last round ``(b, latents, n_tokens)``.  Parameters from ``params``: ``latents``,
    ``depth``, ``heads`` (4), ``attention_dropout`` (0.1), ``horizon_queries``, ``output``; ``size`` is
    ``d_model`` when >= 8 (else 64).
    """

    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(PerceiverMerge, self).__init__()
        params = params or {}
        if not isinstance(shape, dict):
            shape = {'input': list(shape)}
        self.names = list(shape.keys())
        self.name = f'{name_scope}_perceiver_{layer_id}'
        heads = int(params.get('heads', 4))
        d_model = int(size) if isinstance(size, (int, float)) and size >= 8 else 64
        self.d_model = int(np.ceil(d_model / heads) * heads)
        dropout = float(params.get('attention_dropout', 0.1))
        self.n_latents = int(params.get('latents', 16))
        self.depth = int(params.get('depth', 2))
        self.n_queries = int(params.get('horizon_queries', params.get('n_out', 1)) or 1)
        self.output_mode = str(params.get('output', 'flat'))
        self.views = {name: token_view(s) for name, s in shape.items()}
        self.kv_names = list(self.names)                     # every branch feeds the latents
        self.query_names = ['(latents)']                     # reported by the dry build
        self.proj = nn.ModuleDict({self._key(n): nn.Linear(f, self.d_model) for n, (_, f) in self.views.items()})
        self.branch_embed = nn.ParameterDict({self._key(n): nn.Parameter(torch.zeros(1, 1, self.d_model)) for n in self.names})
        self.pos_embed = nn.ParameterDict({self._key(n): nn.Parameter(torch.zeros(1, t, self.d_model))
                                           for n, (t, _) in self.views.items() if t > 1})
        self.latents = nn.Parameter(torch.zeros(1, self.n_latents, self.d_model))
        self.queries = nn.Parameter(torch.zeros(1, self.n_queries, self.d_model))
        self.norm_tokens = nn.LayerNorm(self.d_model)
        self.rounds = nn.ModuleList()
        for _ in range(self.depth):
            self.rounds.append(nn.ModuleDict({
                'norm_q': nn.LayerNorm(self.d_model),
                'cross': nn.MultiheadAttention(self.d_model, heads, dropout=dropout, batch_first=True),
                'norm_s': nn.LayerNorm(self.d_model),
                'self': nn.MultiheadAttention(self.d_model, heads, dropout=dropout, batch_first=True),
                'ffn': nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 2 * self.d_model), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(2 * self.d_model, self.d_model)),
            }))
        self.norm_dec_q = nn.LayerNorm(self.d_model)
        self.norm_dec_kv = nn.LayerNorm(self.d_model)
        self.decoder = nn.MultiheadAttention(self.d_model, heads, dropout=dropout, batch_first=True)
        self.dec_ffn = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 2 * self.d_model), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(2 * self.d_model, self.d_model))
        self.norm_out = nn.LayerNorm(self.d_model)
        if self.output_mode == 'steps':
            self.readout = nn.Linear(self.d_model, 1)
            self.step_bias = nn.Parameter(torch.zeros(self.n_queries))
        self.last_attention = None
        self.last_cross_attention = None
        self.apply(_init_weights)
        for p in [self.latents, self.queries, *self.branch_embed.values(), *self.pos_embed.values()]:
            nn.init.trunc_normal_(p, std=0.02)
        with torch.no_grad():
            sample = [torch.zeros(10, *list(shape[n])[1:]) for n in self.names]
            self.output_shape = list(self.forward(sample if len(sample) > 1 else sample[0]).shape)

    @staticmethod
    def _key(name):
        return name.replace('/', '__').replace('.', '_')

    def _embed(self, x, name):
        tokens = self.proj[self._key(name)](to_tokens(x)) + self.branch_embed[self._key(name)]
        if self._key(name) in self.pos_embed:
            tokens = tokens + self.pos_embed[self._key(name)][:, :tokens.shape[1]]
        return tokens

    def forward(self, x_list):
        if not isinstance(x_list, list):
            x_list = [x_list]
        branches = dict(zip(self.names, x_list))
        tokens = self.norm_tokens(torch.cat([self._embed(branches[n], n) for n in self.names], 1))
        b = tokens.shape[0]
        z = self.latents.expand(b, -1, -1)
        for r in self.rounds:
            attended, weights = r['cross'](r['norm_q'](z), tokens, tokens, need_weights=True)
            z = z + attended
            h = r['norm_s'](z)
            z = z + r['self'](h, h, h, need_weights=False)[0]
            z = z + r['ffn'](z)
        self.last_cross_attention = weights.detach()
        q = self.queries.expand(b, -1, -1)
        kv = self.norm_dec_kv(z)
        decoded, dec_weights = self.decoder(self.norm_dec_q(q), kv, kv, need_weights=True)
        self.last_attention = dec_weights.detach()
        out = q + decoded
        out = self.norm_out(out + self.dec_ffn(out))                       # (b, n_queries, d_model)
        if self.output_mode == 'steps':
            return self.readout(out).squeeze(-1) + self.step_bias
        return out.flatten(start_dim=1)


class CrossAttentionBoth(CrossAttention):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(CrossAttentionBoth, self).__init__(shape, params, size, name_scope, layer_id, train=train, both=True)


def layers_func():
    layers = {'conv_2d': conv_2d,
              'time_distr_conv_2d': time_distr_conv_2d,
              'conv_3d': conv_3d,
              'cnnT': cnnT,
              'time_distr_conv_3d': time_distr_conv_3d,
              'lstm': lstm,
              'lstm_att': lstm_att,
              'timexer': timexer,
              'tide': tide,
              'tsmixer': tsmixer,
              'grid_transformer': grid_transformer,
              'afno': afno,
              'cnn_att': cnn_attention,
              'bi_lstm': bi_lstm,
              'bi_transformer': bi_transformer,
              'bi_spacetimeformer': bi_spacetimeformer,
              'transformer': transformer,
              'Reshape': Reshape,
              'unet': unet,
              'timm_net': vit_net,
              'cross_attention': CrossAttention,          # token-level attention (new)
              'cross_attention_both': CrossAttentionBoth,
              'perceiver': PerceiverMerge,                # latent array + horizon queries
              'gated_fusion': GatedFusion,                # the former cross_attention: a conditioned gate
              'gated_fusion_both': GatedFusionBoth,
              'tokens': Tokens,
              'clip': Clip,
              'concatenate': Concat_,
              'vit_net': vit_net,
              'time_distr_vit_net': time_distr_vit_net,
              'dense': dense
              }
    return layers
