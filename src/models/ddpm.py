# https://github.com/w86763777/pytorch-ddpm/blob/master/main.py
from typing import List
import copy
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .modules import UNet


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


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
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                extract(
                    1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
                extract(
                    self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                    x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.hparams.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.hparams.mean_type == 'xprev':  # the model predicts x_{t-1}
            x_prev = self.unet(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.hparams.mean_type == 'xstart':  # the model predicts x_0
            x_0 = self.unet(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.hparams.mean_type == 'epsilon':  # the model predicts epsilon
            eps = self.unet(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.hparams.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.hparams.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)

        def warmup_lr(step):
            return min(step, self.hparams.warmup) / self.hparams.warmup

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
