import math
import torch
import torch.nn.functional as F

from typing import Optional, Union, Tuple, List
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        n_channels: int
    ):
        super().__init__()

        self.n_channels = n_channels
        self.embedding = nn.Sequential(
            nn.Linear(self.n_channels // 4, self.n_channels),
            nn.SiLU(),
            nn.Linear(self.n_channels, self.n_channels)
        )

    def fowrad(self, t: torch.Tensor):
        """
        Args:
            t: position values which has shape [height * width]
        """
        half_channels = self.n_channels // 8
        emb = math.log(10_000) / (half_channels - 1)
        emb = torch.exp(torch.arange(half_channels, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]    # [height * width, half_channels]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)    # [height * width, half_channels * 2]
        emb = self.embedding(emb)    # [height * width, n_channels ]

        return emb


class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_channels: int = None, 
        n_groups: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

        self.time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: input which has shape [batch_size, in_channels, height, width]
            t: time embedding which has shape [batch_size, time_channels]
        """
        h = self.conv1(x)
        h += self.time(t)
        h = self.conv2(x)

        return h + self.residual(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int = 4,
        d_k: int = None,
        n_groups: int = 32
    ):
        super().__init__()

        if d_k is None:
            d_k = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.proj = nn.Lienar(n_channels, n_heads * d_k * 3)
        self.linear = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor]):
        """
        Args:
            x: input which has shape [batch_size, in_channels, height, width]
            t: time embedding which has shape [batch_size, time_channels]
        """
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)    # [batch_size, height*width, n_channels]
        qkv = self.proj(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)    # [batch_size, height*width, n_heads, d_k * 3]
        q, k, v = torch.chunk(qkv, 3, dim=-1)    # [batch_size, height*width, n_heads, d_k]
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale    # [batch_size, height*width, height*width, n_heads]
        attn = attn.softmax(dim=2)
        h = torch.einsum('bijh,bjhd->bihd', attn, v)    # [batch_size, height*width, n_heads, d_k]
        h = h.view(batch_size, -1, self.n_heads * self.d_k)     # [batch_size. height_width, n_head * d_k]
        h = self.linear(h)    # [batch_size, height*width, n_channels]

        h += x
        h = h.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return h
    

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool
    ):
        super().__init__()

        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class DownSample(nn.Module):
    def __init__(
        self,
        n_channels: int
    ):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self,  x: torch.Tensor, t: torch.Tensor):
        return self.conv(x)
    

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool
    ):
        super().__init__()

        self.res = ResidualBlock(in_channels+out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x
    

class UpSample(nn.Module):
    def __init__(
        self,
        n_channels: int
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.conv(x)


class MiddleBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        time_channels: int
    ):
        super().__init__()

        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x
    

class UNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = [1, 2, 2, 2],
        is_attn: Union[Tuple[bool, ...], List[bool]] = [False, False, True, True],
        n_blocks: int = 2
    ):
        super().__init__()

        n_resolutions = len(ch_mults)
        
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)
        self.time_emb = TimeEmbedding(n_channels * 4)

        # Down stages
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(DownSample(in_channels))

        self.down = nn.ModuleList(down)

        # Middle stage
        self.middle = MiddleBlock(out_channels, n_channels * 4)

        # Up stage
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            if i > 0: up.append(UpSample(in_channels))

        self.up = nn.ModuleList(up)

        self.head = nn.Sequential(
            nn.GroupNorm(num_groups=8, n_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_emb(t)
        x = self.image_proj(x)

        h = [x]
        for down_block in self.down:
            x = down_block(x, t)
            h.append(x)

        x = self.middle(x, t)

        for up_block in self.up:
            if isinstance(up_block, UpSample):
                x = up_block(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = up_block(x, t)

        return self.head(x)

        