# https://github.com/w86763777/pytorch-ddpm/blob/master/main.py
from functools import partial
from typing import List, Optional
import os
import copy
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

import pytorch_lightning as pl
from .modules import UNet
from .score import evaluate
from .sampler import DDPMSampler
from torchmetrics import MetricCollection, FID, IS

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay)
        )


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
            mean_type: str = 'epsilon',
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
        :param mean_type: predict variable ('xprev', 'xstart', 'epsilon')
        :param var_type: variance type ('fixedlarge', 'fixedsmall')
        """
        super().__init__()
        self.save_hyperparameters()

        self.unet = UNet(
            T=T,
            ch=ch,
            ch_mult=ch_mult,
            attn=attn,
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )

        self.ema_unet = copy.deepcopy(self.unet)

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

        self._fid = FID(feature=2048, compute_on_step=True, dist_sync_on_step=True)
        self._is = IS(feature=2048, compute_on_step=True, dist_sync_on_step=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Evaluation
        # fid_cache = os.path.join(
        #     self.trainer.default_root_dir, 'stats/fid_cache_train.npz'
        # )
        # self.eval_IS_FID = partial(
        #     evaluate, fid_cache=fid_cache, sample_size=self.hparams.sample_size, img_size=self.hparams.img_size
        # )

        self.sample_dir = os.makedirs(os.path.join(self.trainer.log_dir, 'sample'), exist_ok=True)

        self.unet_sampler = DDPMSampler(
            self.unet, beta_1=self.hparams.beta_1, beta_T=self.hparams.beta_T, T=self.hparams.T,
            mean_type=self.hparams.mean_type, var_type=self.hparams.var_type
        )
        self.ema_unet_sampler = DDPMSampler(
            self.ema_unet, beta_1=self.hparams.beta_1, beta_T=self.hparams.beta_T, T=self.hparams.T,
            mean_type=self.hparams.mean_type, var_type=self.hparams.var_type
        )

    def forward(self, x_0):
        t = torch.randint(self.hparams.T, size=(x_0.shape[0],), device=self.device)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        loss = F.mse_loss(self.unet(x_t, t), noise, reduction='none')
        return loss

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss = self(X).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        with torch.no_grad():
            ema(self.unet, self.ema_unet, self.hparams.ema_decay)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        loss = self(X).mean()
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        # Generate image and save
        x_T = self.x_T.to(self.device)

        x_0 = self.ema_unet_sampler(x_T)


        save_image(
            (make_grid(x_0).cpu() + 1) / 2,
            fp=os.path.join(self.sample_dir, '%d.png' % self.current_epoch)
        )

        # Evaluate FID and IS scores
        ema_fid = self._fid((x_0+1)*255/2)
        ema_is = self._is((x_0+1)*255/2)

        self.log_dict({
            'IS_EMA': ema_is,
            'FID_EMA': ema_fid
        }, prog_bar=False, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)

        def warmup_lr(step):
            return min(step, self.hparams.warmup) / self.hparams.warmup

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
