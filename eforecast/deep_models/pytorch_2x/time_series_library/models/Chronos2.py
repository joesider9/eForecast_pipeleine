from eforecast.model_library import pretrained
import torch
from eforecast.execution_policy import torch_device
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from chronos import BaseChronosPipeline


class Model(nn.Module):
    def __init__(self, configs, use_norm=False):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.model = pretrained(BaseChronosPipeline,
            "amazon/chronos-2", revision=getattr(configs, "model_revision", None), profile="chronos", library="chronos-forecasting", device_map=str(torch_device(getattr(configs, 'device', 'auto'))))
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = use_norm
        self.d_model = configs.d_model
        self.projection = nn.Linear(configs.enc_in, self.d_model)

    def forecast(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc.sub(means)
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        x_enc = self.projection(x_enc)
        x_enc = x_enc.permute(0, 2, 1)
        quantiles, dec_out = self.model.predict_quantiles(x_enc, prediction_length=self.pred_len, quantile_levels=[0.1, 0.5, 0.9])
        dec_out = torch.stack(dec_out, dim=0).to(x_enc.device)
        dec_out= dec_out.permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * \
                      (stdev[:, 0, :].unsqueeze(1))
            dec_out = dec_out + \
                      (means[:, 0, :].unsqueeze(1))
        return dec_out

    def forward(self, x_enc, x_mark_enc):
        dec_out = self.forecast(x_enc)
        return dec_out

