import math
import numpy as np
import torch
import torch.nn.functional as F

from typing import List
from einops import rearrange
from torch import nn
from ldm.modules.attention import LinearAttention
from ldm.modules.distribution import DiagonalGaussianDistribution


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
        dropout=0.0,
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
    def __init__(self, channels: int):
        super().__init__()

        self.norm = Normalize(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5
        
    def forward(self, x):
        x_norm = self.norm(x)
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        b, c, h, w = x.shape
        q = q.reshape(b, c, h*w)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.einsum('bci,bcj->bij', q, k)  * self.scale # [b, hw(q), hw(k)]
        attn = F.softmax(attn, dim=2)

        out = torch.einsum('bij,bcj->bci', attn, v)  # [b, c, hw(q)]
        out = out.reshape(b, c, h, w)

        out = self.proj_out(out)

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
    

class UNet(nn.Module):
    def __init__(
        self,
        *
        channels,
        out_channels,
        channel_multiplier=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolution,
        dropout=0.0,
        resample_with_conv=True,
        in_channels,
        resolution,
        use_timestep=True,
        use_linear_attn=False,
        attn_type='vanilla'
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = 'linear'
        self.channels = channels
        self.temb_channels = self.channels * 4
        self.num_resolutions = len(channel_multiplier)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                nn.Linear(self.channels, self.temb_channels),
                Swish(),
                nn.Linear(self.temb_channels, self.temb_channels)
            ])

        # Downsampling Layer
        self.conv_in = nn.Conv2d(in_channels, self.channels, kernel_size=3, stride=1, padding=1)

        current_resolution = resolution
        in_channel_multiplier = (1, ) + tuple(channel_multiplier)
        self.down = nn.ModuleList()
        for i in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_channel_multiplier[i]
            block_out = channels * channel_multiplier[i]
            for j in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in, 
                                        out_channels=block_out, 
                                        temb_channels=self.temb_channels,
                                        dropout=dropout)
                )
                block_in = block_out
                if current_resolution in attn_resolution:
                    attn.append(
                        make_attn(in_channels=block_in, attn_type=attn_type)
                    ) 
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resample_with_conv)
                current_resolution = current_resolution // 2
            self.down.append(down)
            
        # Middle Layer
        self.middle = nn.Module()
        self.middle.block_1 = ResnetBlock(in_channels=block_in,
                                                               out_channels=block_in,
                                                               temb_channels=self.temb_channels,
                                                               dropout=dropout)
        self.middle.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.middle.block_2 = ResnetBlock(in_channels=block_in,
                                                               out_channels=block_in,
                                                               temb_channels=self.temb_channels,
                                                               dropout=dropout)
        
        # Upsampling Layer
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channel_multiplier[i]
            skip_in = channels * channel_multiplier[i]
            for j in range(self.num_res_blocks + 1):
                if j == self.num_res_blocks:
                    skip_in = channels * in_channel_multiplier[i]
                block.append(
                    ResnetBlock(in_channels=block_in+skip_in,
                                        out_channels=block_out,
                                        temb_channels=self.temb_channels,
                                        dropout=dropout)
                )
                block_in = block_out
                if current_resolution in attn_resolution:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i != 0:
                up.upsample = Upsample(block_in, resample_with_conv)
                current_resolution = current_resolution * 2
            self.up.insert(0, up)

        # Head Lyaer
        self.norm_out = Normalize(block_in)
        self.activation = Swish()
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None, context=None):
        if context is not None:
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            assert t is not None
            temb = get_timestep_embedding(t, self.channels)
            temb = self.temb.dense[0](temb)
            temb = self.temb.dense[1](temb)
            temb = self.temb.dense[2](temb)
        else:
            temb = None

        # Downsampling
        hs = [self.conv_in(x)]
        for i in range(self.num_resolutions):
            for j in range(self.num_res_blocks):
                h = self.down[i].block[j](hs[-1], temb)
                if len(self.down[i].attn) > 0:
                    h = self.down[i].attn[j](h)
                hs.append(h)
            if i != self.num_resolutions - 1:
                hs.append(self.down[i].downsample(hs[-1]))
            
        # Middle
        h = hs[-1]
        h = self.middle.block_1(h, temb)
        h = self.middle.attn(h, temb)
        h = self.middle.block_2(h, temb)

        # Upsampling
        for i in reversed(range(self.num_resolutions)):
            for j in range(self.num_res_blocks + 1):
                h = self.up[i].block[j](torch.cat([h, hs.pop()]), temb)
                if len(self.up[i].attn) > 0:
                    h = self.up[i].attn[j](h)
            if i != 0:
                h = self.up[i].upsample(h)
            
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
            
        return h
    
    def get_last_layer(self):
        return self.conv_out.weight
    

class Encoder(nn.Module):
    def __init__(
        self, 
        *,
        channels: int,
        channel_multiplier: List[int],
        n_resnet_blocks: int,
        in_channels: int,
        latent_channels: int
    ):
        super().__init__()

        n_resolutions = len(channel_multiplier)
        channels_list = [m * channels for m in [1] + channel_multiplier]

        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        # Downsample Layers
        self.down = nn.ModuleList()
        for i in range(n_resolutions):
            resnet_blocks = nn.ModuleList()
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels  = channels_list[i + 1]
            down = nn.Module()
            down.block = resnet_blocks
            if i != n_resolutions - 1:
                down.downsample = Downsample(channels)
            else:
                down.downsample = nn.Identity()

            self.down.append(down)
        
        # Middle Layers
        self.middle = nn.Module()
        self.middle.block_1 = ResnetBlock(channels, channels)
        self.middle.attn = AttentionBlock(channels)
        self.middle.block_2 = ResnetBlock(channels, channels)
        
        # Head Layers
        self.norm_out = Normalize(channels)
        self.activation_out =Swish()
        self.conv_out = nn.Conv2d(channels, 2 * latent_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, image: torch.Tensor):
        x = self.conv_in(image)

        for down in self.down:
            for block in down.block:
                x = block(x)
            x = down.downsample(x)

        x = self.middle.block_1(x)
        x = self.middle.attn(x)
        x = self.middle.block_2(x)

        x = self.norm_out(x)
        x = self.activation_out(x)
        x = self.conv_out(x)

        return x
    

class Decoder(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        out_channels: int,
        latent_channels: int
    ):
        super().__init__()
        
        n_resolutions = len(channel_multipliers)
        channels_list = [m * channels for m in channel_multipliers]
        channels = channels_list[-1]

        # z to block_in
        self.conv_in = nn.Conv2d(latent_channels, channels, kernel_size=3, stride=1, padding=1)

        # Middle Layer
        self.middle = nn.Module()
        self.middle.block_1 = ResnetBlock(channels, channels)
        self.middle.attn = AttentionBlock(channels)
        self.middle.block_2 = ResnetBlock(channels)
        
        # Upsampling Layer  
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            resnet_blocks = nn.ModuleList()

            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i]))
                channels = channels_list[i]
            up = nn.Module()
            up.block = resnet_blocks
            if i != 0:
                up.upsample = Upsample(channels)
            else: 
                up.upsample = nn.Identity()
            self.up.insert(0, up)
        
        # Head
        self.norm_out = Normalize(channels)
        self.activation_out = Swish()
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor):
        h = self.conv_in(z)

        h = self.middle.block_1(h)
        h = self.middle.attn(h)
        h = self.middle.block_2(h)

        for up in reversed(self.up):
            for block in up.block:
                h = block(h)
            h = h.upsample(h)

        h = self.norm_out(h)
        h = self.activation_out(h)
        image = self.conv_out(h)

        return image
    

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 1),
            ResnetBlock(in_channels=in_channels,
                                 out_channels=in_channels * 2,
                                 temb_channels=0,
                                 dropout=0.0),
            ResnetBlock(in_channels=in_channels * 2,
                                 out_channels=in_channels * 4,
                                 temb_channels=0,
                                 dropout=0.0),
            ResnetBlock(in_channels=in_channels * 4,
                                 out_channels=in_channels * 2,
                                 temb_channels=0,
                                 dropout=0.0),
            nn.Conv2d(in_channels * 2, in_channels, 1),
            Upsample(in_channels, with_conv=True)
        ])
        
        self.norm_out = Normalize(in_channels)
        self.activation = Swish()
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1, 2, 3]:
                x = layer(x, None)
            else: 
                x = layer(x)
        
        h = self.norm_out(x)
        h = self.activation(x)
        h = self.conv_out(x)
        return x
    

class UpsampleDecoder(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        channels,
        num_res_blocks,
        resolution,
        channel_multiplier = (2, 2),
        dropout = 0.0
    ):
        super().__init__()

        # Upsampling
        self.temb_channels = 0
        self.num_resolutions = len(channel_multiplier)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        current_resolution = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i in range(self.num_resolutions):
            res_block = []
            block_out = channels * channel_multiplier[i]
            for j in range(self.num_res_blocks + 1):
                res_block.append(
                    ResnetBlock(in_channels=block_in,
                                        out_channels=block_out,
                                        temb_channels=self.temb_channels,
                                        dropout=dropout)
                )
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                current_resolution = current_resolution * 2
        
        # Head
        self.norm_out = Normalize(block_in)
        self.activation = Swish()
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = x
        for k, i in enumerate(range(self.num_resolutions)):
            for j in range(self.num_res_blocks + 1):
                h = self.res_blocks[i][j](h, None)
            if i != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        return h
    

class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, middle_channels, out_channels, depth=2):
        super().__init__()
        self.factor = factor
        self.conv_in = nn.Conv2d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1)
        self.res_block_1 = nn.ModuleList([
            ResnetBlock(in_channels=middle_channels,
                                out_channels=middle_channels,
                                temb_channels=0,
                                dropout=0.0) for _ in range(depth)
        ])
        self.attn = AttentionBlock(middle_channels)
        self.res_block_2 = nn.ModuleList([
            ResnetBlock(in_channels=middle_channels,
                                out_channels=middle_channels,
                                temb_channels=0,
                                dropout=0.0) for _ in range(depth)
        ])
        self.conv_out = nn.Conv2d(middle_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block_1:
            x = block(x, None)
        x = F.interpolate(x, size=(int(round(x.shape[2]*self.factor)), int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for block in self.res_block_2:
            x = block(x, None)
        x = self.conv_out(x)
        return x
    

class MergedRescaleEncoder(nn.Module):
    def __init__(
        self,
        in_channels, 
        channels,
        resolution,
        out_channels,
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resample_with_conv=True,
        channel_multiplier=(1, 2, 4, 8),
        rescale_factor=1.0,
        rescale_module_depth=1
    ):
        super().__init__()
        intermediate_channels = channels * channel_multiplier[-1]
        self.encoder = Encoder(
            in_channels=in_channels, num_res_blocks=num_res_blocks, channels=channels, channel_multiplier=channel_multiplier,
            z_channels=intermediate_channels, double_z=False, resolution=resolution, attn_resolution=attn_resolutions,
            dropout=dropout, resample_with_conv=resample_with_conv, out_channels=None
        )
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=intermediate_channels,
                                                       middle_channels=intermediate_channels, out_channels=out_channels,
                                                       depth =rescale_module_depth)
        
    def forward(self,  x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x
    

class MergedRescaleDecoder(nn.Module):
    def __init__(
        self,
        z_channels,
        out_channels,
        resolution,
        num_res_blocks,
        attn_resolutions,
        channels,
        channel_multiplier=(1, 2, 4, 8),
        dropout=0.0,
        resample_with_conv=True,
        rescale_factor=1.0,
        rescale_module_depth=1
    ):
        super().__init__()
        tmp_channels = z_channels * channel_multiplier[-1]
        self.decoder = Decoder(
            out_channels=out_channels, z_channels=tmp_channels, attn_resolutions=attn_resolutions,
            dropout=dropout, resample_with_conv=resample_with_conv, in_channels=None, num_res_blocks=num_res_blocks,
            channel_multiplier=channel_multiplier, resolution=resolution, channels=channels
        )
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=z_channels, middle_channels=tmp_cahnnels,
                                                       out_channels=tmp_channels, depth=rescale_module_depth)
        
    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x
    

class Upsampler(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, channel_multiplier=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size // in_size)) + 1
        factor_up = 1. + (out_size % in_size)
        print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, num_res_blocks=2,
                                                       out_channels=in_channels)
        self.decoder = Decoder(
            out_channels=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
            attn_resolutions=[], in_channels=None, channels=in_channels,
            channel_multiplier=[channel_multiplier for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.rescaler(x)
        x= self.decoder(x)
        return x
    

class Resize(nn.Module):
    def __init__(self, in_channels=None, learned=False, mode='bilinear'):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name__} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x, scale_factor=1.0):
        if scale_factor == 1.0:
            return x
        else:
            x = F.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x


class FirstStagePostProcessor(nn.Module):
    def __init__(
        self,
        channel_multiplier: list,
        in_channels,
        pretrained_model: nn.Module=None,
        reshape=False,
        n_channels=None,
        dropout=0.0,
        pretrained_config=None
    ):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, "Either 'pretrained_model' or 'pretrained_config' must not be None"
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, "Either 'pretrained_model' or 'pretrained_config' must not be None"
            self.instantiate_pretrained(pretrained_config)
        
        self.do_reshape = reshape

        if n_channels is None:
            n_channels = self.pretrained_model.encoder.channels

        self.proj_norm = Normalize(in_channels, num_groups=in_channels // 2)
        self.proj = nn.Conv2d(in_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.activation = Swish()

        blocks = []
        downs =[]
        in_channels = n_channels
        for m in channel_multiplier:
            blocks.append(ResnetBlock(in_channels=in_channels, out_channels=m*n_channels, dropout=dropout))
            in_channels = m * n_channels
            downs.append(Downsample(in_channels, with_conv=False))
        
        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)

    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_with_pretrained(self, x):
        c = self.pretrained_model.encode(x)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
        return c
    
    def forward(self, x):
        z_fs = self.encode_wtih_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = self.activation(z)

        for submodel, downmodel in zip(self.model, self.downsampler):
            z = submodel(z, temb=None)
            z = downmodel(z)

        if self.do_reshape:
            z = rearrange(z, 'b c h w -> b (h w) c')
        return z