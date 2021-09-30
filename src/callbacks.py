import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
from torchvision.utils import make_grid


# https://docs.wandb.ai/guides/integrations/lightning

class ImageCallback(pl.Callback):
    """Logs the input and output images of a module.

    Images are stacked into a mosaic, with output on the top
    and input on the bottom."""

    def __init__(self, img_size=32, sample_size=36):
        super().__init__()
        self.noise = torch.randn((sample_size, 3, img_size, img_size))

    def on_validation_end(self, trainer, pl_module):
        val_imgs = self.noise.to(device=pl_module.device)

        outs = pl_module.denoise(val_imgs)

        caption = "Generated images"
        pl_module.log_dict({
            "val/examples": [wandb.Image(make_grid(outs, nrow=6), caption=caption)],
            "global_step": trainer.global_step
        })
