import os
import sys
import torch.nn as nn
# ── import Spacetimeformer (needs STFormer_good_layers on path) ───────────────
_stf_dir = os.path.dirname(__file__)
if _stf_dir not in sys.path:
    sys.path.insert(0, _stf_dir)
from eforecast.deep_models.pytorch_2x.STFormer.spacetimeformer.spacetimeformer_model.nn.model import Spacetimeformer

class spacetimeformer_model(nn.Module):
    def __init__(self, T, C, T_dec, C_dec, d_model, n_heads, e_layers, d_layers):
        super(spacetimeformer_model, self).__init__()
        self.stf = Spacetimeformer(
            d_yc=T,
            d_yt=C,
            d_x=C,
            d_yc_dec=T_dec,
            d_yt_dec=C_dec,
            d_x_dec=C_dec,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=256,
            d_queries_keys=128,
            d_values=128,
            start_token_len=0,
            pos_emb_type='t2v',
            max_seq_len=None,
            global_self_attn='performer',
            local_self_attn='none',  # required for temporal embedding
            global_cross_attn='performer',
            local_cross_attn='none',  # required for temporal embedding
            dropout_emb=0.05,
            dropout_ff=0.05,
            dropout_attn_out=0.0,
            dropout_qkv=0.0,
            dropout_attn_matrix=0.0,
            embed_method='temporal',  # seq len = T, not N*T
            norm='batch',
            use_final_norm=True,
            recon_mask_skip_all=1.0,
            verbose=False,
        )
    def forward(self, enc_x, enc_y, dec_x, dec_y):

        forecast_out, recon_out, classifier_enc_out, _ = self.stf(enc_x, enc_y, dec_x, dec_y)
        # forecast_out: [B, 1, N]
        return forecast_out.squeeze(1)                    # [B, N]
