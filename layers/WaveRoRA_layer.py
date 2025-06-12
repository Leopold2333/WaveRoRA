import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GatedAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads=8, d_keys=None, d_values=None,
                 residual=True, gate=True):
        super(GatedAttentionLayer, self).__init__()
        self.residual = residual
        self.gate = gate
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        if self.residual:
            self.skip_projection = nn.Linear(d_values * n_heads, d_model)
        if self.gate:
            self.z_projection = nn.Linear(d_model, d_model)
            self.act = nn.SiLU()

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        # multi-head
        q = self.query_projection(queries).view(B, L, self.n_heads, -1)
        k = self.key_projection(keys).view(B, S, self.n_heads, -1)
        v = values.view(B, S, self.n_heads, -1)

        out, attn = self.inner_attention(q, k, v, attn_mask)
        out = out.reshape(B, L, -1)
        if self.residual:
            out = out + self.skip_projection(values)
        if self.gate:
            out = out * self.act(self.z_projection(values))

        return out, attn


class WEncoderLayer(nn.Module):
    def __init__(self, 
                 attention1,
                 d_model, d_ff=None, expand=64,
                 ks=1,
                 dropout=0.1, activation="relu", 
                 batch_size=32, J=5):
        super(WEncoderLayer, self).__init__()
        self.B = batch_size
        self.J = J

        d_ff = d_ff or 4 * d_model
        self.attention1 = attention1

        self.expand_projection = nn.Linear(expand * (J + 1), d_model)
        self.out_projection = nn.Linear(d_model, expand * (J + 1))

        self.wb = nn.ModuleList([
            WaveNormBlock(d_model=expand, d_ff=d_ff, kernel_size=ks, dropout=dropout, activation=activation)
            for _ in range(J+1)
        ])
        

    def forward(self, x, attn_mask=None):
        # x: [B, M, J, D]
        new_x = rearrange(x, 'b m j d -> b m (j d)')
        # new_x: [B, M, D']
        new_x = self.expand_projection(new_x)
        new_x, attn = self.attention1(
            new_x, new_x, new_x,
            attn_mask=attn_mask
        )
        new_x = self.out_projection(new_x)

        new_x = rearrange(new_x, 'b m (j d) -> b m j d', j=self.J+1)
        new_x = list(torch.unbind(new_x, dim=-2))
        x = list(torch.unbind(x, dim=-2))
        for i in range(self.J+1):
            new_x[i] = self.wb[i](x[i], new_x[i]).unsqueeze(-2)
        y = torch.cat(new_x, dim=-2)

        return y, attn


class WaveNormBlock(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dropout, activation) -> None:
        super(WaveNormBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, new_x):
        # b m d
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)