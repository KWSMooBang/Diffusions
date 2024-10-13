import math
import numpy as np
import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn
from ldm.modules.attention import LinearAttention


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels,
                        eps=1e-6, affine=True)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels,
                                        kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x
    

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels,
                                  kernel_size=3, stride=2, padding=0)
        
    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self, 
        *, 
        in_channels, 
        out_channels=None, 
        conv_shortcut=False,
        dropout,
        temb_channels=512    
    ):
        super().__init__()
        
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        self.activation = Swish()
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels,
                                               kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels,
                                              kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.activation(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else: 
                self.nin_shortcut(x)
            
        return x + h


class LinearAttentionBlock(LinearAttention):
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, head_dim=in_channels, heads=1)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_channels, in_channels,
                                  kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        h = x
        h = self.norm(h)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, h, w = x.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)  # [b, hw, c]
        k = k.reshape(b, c, h*w)
        attn = torch.bmm(q, k)  # [b, hw(q), hw(k)]
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2)

        v = v.reshape(b, c, h*w)
        attn = attn.permute(0, 2, 1)  # [b, hw(k), hw(q)]
        out = torch.bmm(v, attn)  # [b, c, hw(q)]
        out = out.reshape(b, c, h, w)

        out = self.proj(out)

        return x + out


def make_attn(in_channels, attn_type='vanilla'):
    assert attn_type in ['vanilla', 'linear',' none'], f"attention type {attn_type} unknown"
    print(f"making attention of type '{attn_type} wtih {in_channels} in_channels'")
    if attn_type == 'vanilla':
        return AttentionBlock(in_channels)
    elif attn_type == 'none':
        return nn.Identity(in_channels)
    else:
        return LinearAttentionBlock(in_channels)

