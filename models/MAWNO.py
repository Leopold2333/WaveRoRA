import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import torch.fft
from pytorch_wavelets import DWT1D, IDWT1D
from einops import rearrange
from mamba_ssm import Mamba

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        dropout = 0.0
        dim_head = dim // heads
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, spatial_size=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def rearrange_qkv(self, q, k, v, h):
        b, n, _ = q.size()
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        return q, k, v

class WaveConv1d(nn.Module):
    def __init__(self, configs, index):
        super(WaveConv1d, self).__init__()
        self.config = configs
        # self.dummy = torch.tensor(np.expand_dims(np.repeat(self.t[:, np.newaxis], self.prob_dim, axis=1), axis=0), dtype=torch.float32).to(self.device).permute(0, 2, 1)
        self.in_channels = self.config.d_model
        self.out_channels = self.config.d_model
        self.level = 4
        self.modes1 = configs.seq_len // (2 ** self.level)
        self.dwt1d = DWT1D(wave='haar', J=self.level, mode="zero")
        self.idwt1d = IDWT1D(wave='haar', mode="zero")

        temp_seq = torch.rand(1, 1, configs.seq_len)
        _, temp_seq_yh = self.dwt1d(temp_seq)
        temp_seq_len = [y.shape[-1] for y in temp_seq_yh]

        # self.sa_c = SelfAttention(dim=self.config.d_model, heads=1)
        # self.sa_c = nn.Linear(self.config.d_model,self.config.d_model)
        # self.sa_c = nn.Conv1d(in_channels=self.config.d_model, out_channels=self.config.d_model, kernel_size=1, padding=0)
        self.sa_c = nn.ModuleList([
            nn.Linear(self.config.d_model, self.config.d_model)
            for _ in range(self.level)
        ])
        self.drop = nn.ModuleList([
            nn.Dropout(0.1) for _ in range(4)
        ])
        self.norm = nn.ModuleList([
            nn.LayerNorm(configs.d_model) for _ in range(4)
        ])

        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, temp_seq_len[-1]))
        # self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1))
        self.weights2 = [
            nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, temp_seq_len[i])).to(configs.device)
            for i in range(self.level)
        ]
        self.mamba = nn.ModuleList([
            Mamba(d_model=temp_seq_len[i], d_conv=2, expand=1)
            for i in range(4)
        ])
        self.mamba2 = nn.ModuleList([
            Mamba(d_model=self.config.d_model, d_conv=2, expand=1)
            for i in range(4)
        ])


    def mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # B, D, L
        x_ft, x_coeff = self.dwt1d(x)

        out_ft = self.mul1d(x_ft, self.weights1)

        for i in range(len(x_coeff)):
            res = x_coeff[i]
            x_coeff[i] = x_coeff[i].permute(0, 2, 1)
            x_coeff[i] = self.sa_c[i](x_coeff[i])       
            x_coeff[i] = F.gelu(x_coeff[i])
            x_coeff[i] = x_coeff[i].permute(0, 2, 1)

            x_coeff[i] = self.mamba[i](self.mul1d(x_coeff[i], self.weights2[i]))
            # x_coeff[i] = self.mamba2[i](x_coeff[i].permute(0, 2, 1)).permute(0, 2, 1)
            x_coeff[i] = self.norm[i]((res + self.drop[i](x_coeff[i])).permute(0, 2, 1)).permute(0, 2, 1)


        x = self.idwt1d((out_ft, x_coeff))
        return x


class Block(nn.Module):
    def __init__(self, configs, dim, index):
        super(Block, self).__init__()
        self.configs = configs
        # self.level = 4
        
        # self.dwt1d = DWT1D(wave='haar', J=self.level, mode="zero")
        # self.idwt1d = IDWT1D(wave='haar', mode="zero")
        # temp_seq = torch.rand(1, 1, configs.seq_len)
        # temp_seq_yl, temp_seq_yh = self.dwt1d(temp_seq)
        # temp_seq_len = [y.shpae[-1] for y in temp_seq_yh]

        self.filter = WaveConv1d(self.configs, index)

        self.conv = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        # x: [B, D, L]
        x1 = self.filter(x)
        x2 = self.conv(x)
        x = x1 + x2
        x = F.gelu(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.setup_seed(configs.seed)
        self.pred_len = configs.pred_len
        self.fc0 = nn.Linear(configs.enc_in, configs.d_model)

        self.blocks = nn.ModuleList([
            Block(configs, dim=configs.d_model, index=i)
            for i in range(configs.e_layers)])
        self.out_proj = nn.Linear(configs.seq_len, configs.pred_len)
        self.fc1 = nn.Linear(configs.d_model, 2*configs.d_model)
        self.fc2 = nn.Linear(2*configs.d_model, configs.enc_in)

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # [B, L, M]
        B = x_enc.shape[0]
        if 1:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev


        x = self.fc0(x_enc)

        x = x.permute(0, 2, 1)  # [B, D, L]

        for blk in self.blocks:
            x = blk(x)          # [B, D, L]

        x = self.out_proj(x)    # [B, D, H]
        x = x.permute(0, 2, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        if 1:
            # De-Normalization from Non-stationary Transformer
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return x, None