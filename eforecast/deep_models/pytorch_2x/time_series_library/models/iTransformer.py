import torch
import torch.nn as nn
import torch.nn.functional as F
from eforecast.deep_models.pytorch_2x.time_series_library.layers.Transformer_EncDec import Encoder, EncoderLayer
from eforecast.deep_models.pytorch_2x.time_series_library.layers.SelfAttention_Family import FullAttention, AttentionLayer
from eforecast.deep_models.pytorch_2x.time_series_library.layers.Embed import DataEmbedding_inverted
from einops import rearrange


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs, use_norm=False):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.projection1 = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.projection2 = nn.Linear(configs.enc_in, configs.d_model, bias=True)


    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # The embedding makes a token of every variable and of every time mark beside them. The
        # marks are context for the attention only: what is read out, and what the statistics
        # of the normalization belong to, are the N variables (as in the paper's own readout).
        enc_out = enc_out[:, :N, :]
        enc_out = rearrange(enc_out, 'b t v -> b v t')
        if self.use_norm:
            enc_out = enc_out * (stdev[:, 0, :].unsqueeze(1))
            enc_out = enc_out + (means[:, 0, :].unsqueeze(1))
        enc_out = self.projection2(enc_out)
        enc_out = rearrange(enc_out, 'b v t -> b t v')
        dec_out = self.projection1(enc_out)
        dec_out = rearrange(dec_out, 'b t v -> b v t')
        # De-Normalization from Non-stationary Transformer

        return dec_out



    def forward(self, x_enc, x_mark_enc):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out  # [B, L, D]

