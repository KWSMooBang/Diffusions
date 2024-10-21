import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from contextlib import contextmanager
from torch import nn
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.diffusion_modules.model import Encoder, Decoder
from ldm.modules.distribution import GaussianDistribution
from ldm.util import instantiate_from_config


class Autoencoder(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        embed_channels: int,
        latent_channels: int
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * embed_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_channels, latent_channels, kernel_size=1)

    def encode(self, image: torch.Tensor) -> GaussianDistribution:
        z = self.encoder(image)
        moments = self.quant_conv(z)
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        z = self.post_quant_conv(z)
        return self.decoder(z)


class VQModel(pl.LightningModule):
    def __init__(
        self,
        dd_config,
        loss_config,
        n_embed,
        embed_dim,
        checkpoint_path=None,
        ignore_keys=[],
        image_key='image',
        colorize_nlabels=None,
        monitor=None,
        batch_resize_range=None,
        scheduler_config=None,
        lr_g_factor=1.0,
        remap=None,
        sane_index_shape=False,
        use_ema=False
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**dd_config)
        self.decoder = Decoder(**dd_config)
        self.loss = instantiate_from_config(loss_config)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = nn.Conv2d(dd_config['z_channels'], embed_dim, 1)
        self.pos_quant_conv = nn.Conv2d(embed_dim, dd_config['z_channels'], 1)

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer('colorize', torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = None
            print(f"Kepping EMAs of {len(list(self.model_ema.buffers()))}.")
        
        if checkpoint_path is not None:
            self.init_from_checkpoint(checkpoint_path, ignore_keys=ignore_keys)
        
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters)
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_checkpoint(self, path, ignore_keys=list()):
        state_dict = torch.load(path, map_location='cpu')['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            for ignore_key in ignore_keys:
                if key.startwith(ignore_key):
                    print(f"Deleting key {key} from state_dict.")
                    del state_dict[key]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decoder(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
    
    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _ , indices) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, indices
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode='bicubic')
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, indices = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                            last_layer=self.get_last_layer(), split='train',
                                                            predicted_indices=indices)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                                last_layer=self.get_last_layer(), split='train')
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
        