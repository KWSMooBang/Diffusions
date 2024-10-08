import torch 
import torch.nn.functional  as F
import torch.utils.data

from typing import Tuple, Optional
from torch import nn


class DenoiseDiffusion:
    def __init__(
            self,
            eps_model: nn.Module,
            n_steps: int,
    ):
        super(DenoiseDiffusion, self).__init__()
        
        self.eps_model = eps_model
        self.n_steps = n_steps
        self.beta = torch.linspace(0.0001, 0.02, n_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x_0)
        
        mean = self._gather(self.alpha_bar, t) ** 0.5 * x_0
        var = 1 - self._gather(self.alpha_bar, t)

        return mean + (var ** 0.5) * eps

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor):
        eps_theta = self.eps_model(x_t, t)
        alpha_bar = self._gather(self,alpha_bar, t)
        alpha = self._gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5

        mean = 1 / (alpha ** 0.5) * (x_t - eps_coef * eps_theta)
        var = self._gather(self.sigma2, t)
        z = torch.randn(x_t.shape, device=x_t.device)

        return mean + (var ** 0.5) * z

    def loss(self, x_0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x_0.shape[0]

        t = torch.randint(0, self.n_steps, (batch_size, ), device=x_0.device, dtype=torch.long)
        if noise is None:
            noise = torch.rand_like(x_0)
        x_t = self.q_sample(x_0, t, eps=noise)
        eps_theta = self.eps_model(x_t, t)

        return F.mse_loss(noise, eps_theta)


    def _gather(self, consts: torch.Tensor, t: torch.Tensor):
        c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1, 1)


    