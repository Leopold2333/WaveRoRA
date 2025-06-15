import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer, AttentionLayer, GatedAttentionLayer
from layers.Attention import FullAttention, RouterAttention
from layers.Embed import PatchEmbedding, TruncateModule
from utils.tools import Flatten_Head

from einops import rearrange

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.pred_len = configs.pred_len

        # patching settings
        if configs.seq_len % configs.stride==0:
            self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 1)
            process_layer = nn.Identity()
        else:
            # padding at tail
            if configs.padding_patch=="end":
                padding_length = configs.stride - (configs.seq_len % configs.stride)
                self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 2)
                process_layer = nn.ReplicationPad1d((0, padding_length))
            # if not padding, then execute cutting
            else:
                truncated_length = configs.seq_len - (configs.seq_len % configs.stride)
                self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 1)
                process_layer = TruncateModule(truncated_length)
        self.local_token_layer = PatchEmbedding(configs.seq_len, configs.d_model, configs.patch_len, configs.stride, configs.dropout, process_layer,
                                                pos_embed_type=None if configs.embed_type in [0, 2] else configs.pos_embed_type, learnable=configs.pos_learnable, 
                                                ch_ind=configs.ch_ind,
                                                deform_patch=False)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, 
                                      output_attention=configs.output_attention), 
                        configs.d_model, 
                        configs.n_heads
                    ) if configs.attn_type=="SA" else
                    GatedAttentionLayer(
                        RouterAttention(router_num=configs.router_num,
                                        d_model=configs.d_model, 
                                        rotary=configs.rotary,
                                        attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        residual=configs.residual,
                        gate=configs.gate
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=None
        )

        head_nf = configs.d_model * self.patch_num
        self.head = Flatten_Head(False, configs.enc_in, head_nf, configs.pred_len)

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, M]
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        enc_in, M = self.local_token_layer(x_enc.permute(0, 2, 1))  # [B*M, N, D]
        
        enc_out, attns = self.encoder(enc_in)  # [B*M, N, D]

        # head linear
        dec_out = rearrange(enc_out, '(B M) N D -> B M N D', M=M)
        dec_out = self.head(dec_out).permute(0, 2, 1)               # [B, H, M]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns
