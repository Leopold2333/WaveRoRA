import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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