# https://github.com/w86763777/pytorch-ddpm/blob/master/main.py
from functools import partial
import math
from typing import List, Optional
import os
import copy
import torch
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl
from .modules import RefineNet
from .sampler import NCSNSampler
from torchvision.utils import make_grid
import wandb


class NCSN(pl.LightningModule):
    def __init__(
            self,
            ngf: int,
            num_classes: int,
            sigma_begin: float,
            sigma_end: float,
            step_size: float
    ):
        super().__init__()
        self.save_hyperparameters()

        self.score = RefineNet(
            in_channel=3, ngf=ngf, numb_classes=num_classes
        )
        self.register_buffer(
            'sigmas', torch.exp(torch.linspace(math.log(sigma_begin), math.log(sigma_end), num_classes))
        )
        self.register_buffer(
            'alphas', step_size * (self.sigmas / self.sigmas[-1]) ** 2
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.noise = torch.randn((36, 3, 32, 32))
        self.sample_dir = os.makedirs(os.path.join(self.trainer.log_dir, 'sample'), exist_ok=True)

    def forward(scorenet, samples, labels, sigmas, anneal_power=2.):
        used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
        perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
        target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
        scores = scorenet(perturbed_samples, labels)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        return loss

    def denoise(self, x_T):
        # Denoise the original sample
        sampler = NCSNSampler(
            score=self.score,
            sigmas=self.sigmas,
            alphas=self.alphas,
            n_steps_each=100,
        ).to(x_T.device)

        x_0 = sampler(x_T)
        return x_0

    def training_step(self, batch, batch_idx):
        x_0, y = batch
        loss = self(x_0).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_0, y = batch
        loss = self(x_0).mean()
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    @pl.utilities.rank_zero_only
    def on_validation_epoch_end(self) -> None:
        noise = self.noise.to(device=self.device)

        outs = self.denoise(noise)

        caption = "Generated images"
        self.logger.experiment[0].log(
            {"val/generated_images": [wandb.Image(make_grid(outs, nrow=6, normalize=True), caption=caption)]}
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=False)