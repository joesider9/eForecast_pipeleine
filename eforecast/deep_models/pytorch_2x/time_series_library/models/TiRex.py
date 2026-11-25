import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from eforecast.model_library import tirex_model
from eforecast.execution_policy import torch_device


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.model = tirex_model("NX-AI/TiRex", revision=getattr(configs, "model_revision", None), device=str(torch_device(getattr(configs, "device", "auto"))))
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for i in range(x_enc.shape[-1]):
            quantiles, output = self.model.forecast(x_enc[...,i], prediction_length=self.pred_len)
            outputs.append(output)
        dec_out = torch.stack(outputs, dim=-1).to(x_enc.device)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
