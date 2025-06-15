import torch
import torch.nn as nn

class RoPE1d(nn.Module):
    """
    Rotary Positional Embedding. This implementation is for 1D condition.
    """
    def __init__(self, feature_dim, reverse=False, base=10000):
        super(RoPE1d, self).__init__()
        # L, D
        self.reverse = reverse
        k_max = feature_dim // 2

        assert feature_dim % k_max == 0

        # angles: [10000, D/2]
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.arange(base).unsqueeze(-1) * theta_ks
        if reverse:
            angles = torch.flip(angles, dims=[0])
        # rotation: [10000, D/2, 2]
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        # x: [B, L, D] -> [B, L, D/2]
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
        if self.reverse:
            rotation = torch.view_as_complex(self.rotations[-x.shape[-2]:])
        else:
            rotation = torch.view_as_complex(self.rotations[:x.shape[-2]])
        pe_x = rotation * x
        return torch.view_as_real(pe_x).flatten(-2)
