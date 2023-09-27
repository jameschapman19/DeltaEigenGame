# Importing the required modules
import torch
from cca_zoo.deep import DCCA_EY as DCCA_EY_
from cca_zoo.linear import MCCA
from pytorch_lightning import Callback, LightningModule, Trainer
import numpy as np


# Defining a custom callback to log the correlation metrics
class CorrelationCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # Population Objective
        z = pl_module.transform(trainer.val_dataloaders[0])
        rew, pen = self.loss(z)
        corrs = MCCA(pl_module.latent_dims).fit(z).score(z)

        # Logging the sum of correlations and squared correlations
        pl_module.log("val/obj_reward", rew)
        pl_module.log("val/obj_penalty", pen)
        pl_module.log("val/obj", rew - pen)
        pl_module.log("val/corr", corrs.sum())
        pl_module.log("val/corr_squared", (corrs**2).sum())

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Population Objective
        z = pl_module.transform(trainer.train_dataloader)
        rew, pen = self.obj(z)
        corrs = MCCA(pl_module.latent_dims).fit(z).score(z)

        # Logging the sum of correlations and squared correlations
        pl_module.log("train/obj_reward", rew)
        pl_module.log("train/obj_penalty", pen)
        pl_module.log("train/obj", rew - pen)
        pl_module.log("train/corr", corrs.sum())
        pl_module.log("train/corr_squared", (corrs**2).sum())

    def obj(self, z):
        C = np.cov(np.hstack(z).T)
        N, D = z[0].shape
        A = C[:D, D:] + C[D:, :D]
        B = C[:D, :D] + C[D:, D:]
        return np.trace(2 * A), np.trace(B @ B)
