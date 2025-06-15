import torch
import torch.nn as nn
import math
from layers.Transformer_EncDec import Encoder, EncoderLayer, AttentionLayer, GatedAttentionLayer
from layers.WaveRoRA_layer import WEncoderLayer
from pytorch_wavelets import DWT1D, IDWT1D
from layers.Attention import RouterAttention, LinearAttention, FullAttention


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.pred_len = configs.pred_len
        self.embed_type = configs.embed_type
        self.domain = configs.domain
        router_num = configs.router_num if hasattr(configs, 'router_num') and configs.router_num != 0 \
            else int(math.sqrt(configs.enc_in) + math.log2(configs.enc_in)) // 2
        if router_num % 2 != 0:
            router_num += 1
        
        if configs.domain == 'W':
            self.dwt = DWT1D(J=configs.wavelet_layers, wave=configs.wavelet_type, mode=configs.wavelet_mode)
            self.idwt = IDWT1D(wave=configs.wavelet_type, mode=configs.wavelet_mode)

            temp_seq = torch.rand(1, 1, configs.seq_len)
            temp_seq_yl, temp_seq_yh = self.dwt(temp_seq)
            seq_len_J = [y.shape[-1] for y in temp_seq_yh] + [temp_seq_yl.shape[-1]]
            temp_pred = torch.rand(1, 1, configs.pred_len)
            temp_pred_yl, temp_pred_yh = self.dwt(temp_pred)
            pred_len_J = [y.shape[-1] for y in temp_pred_yh] + [temp_pred_yl.shape[-1]]

            self.in_proj_h = nn.ModuleList([
                nn.Linear(seq_len_J[i], configs.wavelet_dim)
                for i in range(configs.wavelet_layers)
            ])
            self.in_proj_l = nn.Linear(seq_len_J[-1], configs.wavelet_dim)

            self.encoder = Encoder(
                [WEncoderLayer(
                    GatedAttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, 
                                      rotary=configs.rotary, d_model=configs.d_model, 
                                      n_heads=configs.n_heads), 
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        residual=configs.residual,
                        gate=configs.gate
                    ) if configs.attn_type == 'SA' else
                    GatedAttentionLayer(
                        LinearAttention(attention_dropout=configs.dropout, rotary=configs.rotary,
                                        d_model=configs.d_model, n_heads=configs.n_heads),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        residual=configs.residual,
                        gate=configs.gate
                    ) if configs.attn_type == 'LA' else 
                    GatedAttentionLayer(
                        RouterAttention(router_num=router_num,
                                        d_model=configs.d_model, 
                                        rotary=configs.rotary,
                                        attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        residual=configs.residual,
                        gate=configs.gate
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    expand=configs.wavelet_dim,
                    ks=configs.ks,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    batch_size=configs.batch_size,
                    J=configs.wavelet_layers
                ) for _ in range(configs.e_layers)],
                # norm_layer=torch.nn.LayerNorm(configs.d_model*(configs.wavelet_layers+1))
            )

            self.out_proj_h = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(configs.wavelet_dim, configs.wavelet_dim * 2),
                    nn.ReLU if configs.activation == 'relu' else nn.GELU(),
                    nn.Linear(configs.wavelet_dim * 2, pred_len_J[i])
                ) for i in range(configs.wavelet_layers)
            ])

            self.out_proj_l = nn.Sequential(
                nn.Linear(configs.wavelet_dim, configs.wavelet_dim * 2),
                nn.ReLU if configs.activation == 'relu' else nn.GELU(),
                nn.Linear(configs.wavelet_dim * 2, pred_len_J[-1])
            )
        elif configs.domain == 'T':
            self.in_proj = nn.Sequential(
                nn.Linear(configs.seq_len, configs.d_model),
                nn.Dropout(configs.dropout)
            )
            self.encoder = Encoder(
                [EncoderLayer(
                    GatedAttentionLayer(
                        RouterAttention(router_num=router_num,
                                        d_model=configs.d_model, 
                                        rotary=configs.rotary,
                                        attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        residual=configs.residual,
                        gate=configs.gate
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)],
                # norm_layer=torch.nn.LayerNorm(configs.d_model*(configs.wavelet_layers+1))
            )
            self.out_proj = nn.Linear(configs.d_model, configs.pred_len)
        elif configs.domain == 'F':
            self.in_proj = nn.Sequential(
                nn.Linear(configs.seq_len+2, configs.d_model),
                nn.Dropout(configs.dropout)
            )
            self.encoder = Encoder(
                [EncoderLayer(
                    GatedAttentionLayer(
                        RouterAttention(router_num=router_num,
                                        d_model=configs.d_model, 
                                        rotary=configs.rotary,
                                        attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        residual=configs.residual,
                        gate=configs.gate
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)],
                # norm_layer=torch.nn.LayerNorm(configs.d_model*(configs.wavelet_layers+1))
            )
            self.out_proj = nn.Linear(configs.d_model, configs.pred_len+2)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, M]
        _, _, M = x_enc.shape
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        if self.embed_type == 2:
            # M:=M+4
            x_enc = torch.cat([x_enc, x_mark_enc], dim=-1)
        if self.domain == 'W':
            # yl: [B, M, L/2^J], yh: [[B, M, L/2^j]]
            yl, yh = self.dwt(x_enc.permute(0, 2, 1))
            # [B, M, L/2^j] -> [B, M, 1, D]
            for i in range(len(yh)):
                yh[i] = self.in_proj_h[i](yh[i]).unsqueeze(-2)
            # yh[3] = torch.zeros_like(yh[3]).to(x_enc.device)
            yl = self.in_proj_l(yl).unsqueeze(-2)
            # [[B, M, 1, D]] -> [B, M, J, D]
            enc_in = torch.cat(yh + [yl], dim=-2)

            enc_out, attn = self.encoder(enc_in)

            enc_out = list(torch.unbind(enc_out, dim=-2))
            
            for i in range(len(yh)):
                yh[i] = self.out_proj_h[i](enc_out[i])
            yl = self.out_proj_l(enc_out[-1])
            output = self.idwt((yl, yh)).permute(0, 2, 1)[:, :, :M]
        elif self.domain == 'T':
            # [B, L, M] -> [B, M, D]
            enc_in = self.in_proj(x_enc.permute(0, 2, 1))
            enc_out, attn = self.encoder(enc_in)
            # [B, L, D] -> [B, H, M]
            output = self.out_proj(enc_out).permute(0, 2, 1)[..., :M]
        elif self.domain == 'F':
            x_enc = torch.fft.rfft(x_enc, dim=1)
            x_enc = torch.cat([x_enc.real, x_enc.imag], dim=1)
            enc_in = self.in_proj(x_enc.permute(0, 2, 1))
            enc_out, attn = self.encoder(enc_in)
            output = self.out_proj(enc_out).permute(0, 2, 1)[..., :M]
            output_real, output_imag = torch.split(output, split_size_or_sections=self.pred_len//2+1, dim=1)
            output = torch.complex(output_real, output_imag)  # (real, imag)
            output = torch.fft.irfft(output, dim=1)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return output, attn