# https://github.com/w86763777/pytorch-ddpm/blob/master/main.py
from functools import partial
from typing import List, Optional
import os
import copy
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from .modules import WideResnet
from .sampler import DDPMSampler
from torchvision.utils import make_grid
import wandb

from torchmetrics import FID, IS


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay)
        )


def normalize_0_255(tensor):
    batch, ch, s1, s2 = tensor.size()

    tensor = tensor.view(batch, -1)

    tensor -= tensor.min(1, keepdim=True)[0]
    tensor /= tensor.max(1, keepdim=True)[0]

    tensor = tensor.view(batch, ch, s1, s2)

    return (tensor * 255).type(torch.uint8)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class DDPM(pl.LightningModule):
    def __init__(
            self,
            T: int,
            ch: int,
            ch_mult: List[int],
            attn: List[int],
            num_res_blocks: int,
            dropout: float,
            beta_1: float,
            beta_T: float,
            warmup: int,
            ema_decay: float,
            sample_size: int,
            img_size: int,
            var_type: str = 'fixedlarge'
    ):
        """
        :param T: total diffusion steps
        :param ch: base channel of UNet
        :param ch_mult: channel multiplier
        :param attn: add attention to these levels
        :param num_res_blocks: resblock in each level
        :param dropout: dropout rate of resblock
        :param beta_1: start beta value
        :param beta_T: end beta value
        :param var_type: variance type ('fixedlarge', 'fixedsmall')
        """
        super().__init__()
        self.save_hyperparameters()

        self.score = WideResnet(
            T=T,
            ch=ch,
            ch_mult=ch_mult,
            attn=attn,
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )

        self.ema_score = copy.deepcopy(self.score)

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double()
        )
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar)
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)
        )

        self.x_T = torch.randn(sample_size, 3, img_size, img_size)

        self.is_score = IS(feature=2048, compute_on_step=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Evaluation
        #
        # from .score import evaluate
        # fid_cache = os.path.join(
        #     self.trainer.default_root_dir, 'stats/fid_cache_train.npz'
        # )
        # self.eval_IS_FID = partial(
        #     evaluate, fid_cache=fid_cache, sample_size=self.hparams.sample_size, img_size=self.hparams.img_size
        # )
        self.noise = torch.randn((36, 3, 32, 32))
        self.sample_dir = os.makedirs(os.path.join(self.trainer.log_dir, 'sample'), exist_ok=True)

    def forward(self, x_0):
        t = torch.randint(self.hparams.T, size=(x_0.shape[0],), device=self.device)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = (
                extract(v=self.sqrt_alphas_bar, t=t, x_shape=x_0.shape) * x_0 +
                extract(v=self.sqrt_one_minus_alphas_bar, t=t, x_shape=x_0.shape) * noise
        )
        loss = F.mse_loss(self.score(x_t, t), noise, reduction='none')
        return loss

    def denoise(self, x_T):
        # Denoise the original sample

        sampler = DDPMSampler(
            self.ema_score, beta_1=self.hparams.beta_1, beta_T=self.hparams.beta_T, T=self.hparams.T,
            var_type=self.hparams.var_type
        ).to(x_T.device)

        x_0 = sampler(x_T)
        return x_0

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss = self(X).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        with torch.no_grad():
            ema(self.score, self.ema_score, self.hparams.ema_decay)
        return loss

    def validation_step(self, batch, batch_idx):
        x_0, y = batch

        t = torch.randint(self.hparams.T, size=(x_0.shape[0],), device=self.device)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = (
                extract(v=self.sqrt_alphas_bar, t=t, x_shape=x_0.shape) * x_0 +
                extract(v=self.sqrt_one_minus_alphas_bar, t=t, x_shape=x_0.shape) * noise
        )
        loss = F.mse_loss(self.ema_score(x_t, t), noise, reduction='none').mean()

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    @pl.utilities.rank_zero_only
    def on_validation_epoch_end(self) -> None:
        noise = self.noise.to(device=self.device)

        outs = self.denoise(noise)

        caption = "Generated images"
        self.logger.experiment[0].log(
            {"val/generated_images": [wandb.Image(make_grid(outs, nrow=6, normalize=True), caption=caption)]}
        )

    def test_step(self, batch, batch_idx):
        x_0, y = batch
        noise = torch.randn(x_0.size(), device=self.device)

        outs = self.denoise(noise)

        outs = normalize_0_255(outs)
        is_score_mean, is_score_std = self.is_score(outs)

        self.log_dict({
            "test_is_score_mean": is_score_mean,
            "test_is_score_std": is_score_std
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)

        def warmup_lr(step):
            return min(step, self.hparams.warmup) / self.hparams.warmup

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
