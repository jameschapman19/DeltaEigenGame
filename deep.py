import os

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from cca_zoo.data.deep import NumpyDataset
from cca_zoo.deepmodels import (
    DCCA,
    architectures,
    DCCA_NOI,
    DCCA_EigenGame,
)
from cca_zoo.deepmodels.objectives import CCA
from cca_zoo.models import MCCA
from multiviewdata.torchdatasets import NoisyMNIST, SplitMNIST, XRMB
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.cuda import device_count
from torch.utils.data import random_split


class CorrelationCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        corrs = pl_module.score(trainer.val_dataloaders[0])
        pl_module.log("val/corr", corrs.sum())
        pl_module.log("val/corr_squared", (corrs**2).sum())


WANDB_START_METHOD = "thread"
defaults = dict(
    data="SplitMNIST",
    mnist_type="MNIST",
    lr=0.001,
    batch_size=100,
    latent_dims=3,
    epochs=1000,
    model="DCCASVD_BIASED",
    architecture="linear",
    rho=0.1,
    random_seed=1,
    optimizer="sgd",
    project="DeepDeltaEigenGame",
    num_workers=0,
)


class DCCA_EY(DCCA_EigenGame):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(self, latent_dims: int, encoders=None, r: float = 0, **kwargs):
        super().__init__(latent_dims=latent_dims, encoders=encoders, **kwargs)
        self.batch_queue = []
        self.val_batch_queue = []

    def forward(self, views, **kwargs):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(views[i]))
        return z

    def training_step(self, batch, batch_idx):
        if len(self.batch_queue) < 1:
            self.batch_queue.append(batch)
            loss = {
                "objective": torch.tensor(0, requires_grad=True, dtype=torch.float32),
            }
        else:
            loss = self.loss(batch["views"], self.batch_queue[0]["views"])
        self.batch_queue.append(batch)
        self.batch_queue.pop(0)
        for k, v in loss.items():
            self.log("train/" + k, v, prog_bar=False)
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        if len(self.val_batch_queue) < 1:
            self.val_batch_queue.append(batch)
            loss = {
                "objective": torch.tensor(0, requires_grad=True, dtype=torch.float32),
            }
        else:
            loss = self.loss(batch["views"], self.val_batch_queue[0]["views"])
        # add current batch to queue and remove oldest batch
        self.val_batch_queue.append(batch)
        self.val_batch_queue.pop(0)
        for k, v in loss.items():
            self.log("val/" + k, v)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.loss(batch["views"])
        for k, v in loss.items():
            self.log("test/" + k, v)
        return loss["objective"]

    def get_AB(self, z):
        N, D = z[0].size()
        C = torch.cov(torch.hstack(z).T)
        A = C[:D, D:] + C[D:, :D]
        B = C[:D, :D] + C[D:, D:]
        return A, B

    def loss(self, views, views2=None, **kwargs):
        z = self(views)
        A, B = self.get_AB(z)
        if views2 is None:
            rewards = torch.trace(2 * A)
            penalties = torch.trace(B @ B)
        else:
            z2 = self(views2)
            A_, B_ = self.get_AB(z2)
            rewards = torch.trace(2 * A)
            penalties = torch.trace(B @ B_)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

    def configure_callbacks(self):
        return [CorrelationCallback()]


class DCCA_SVD(DCCA_EY):
    def loss(self, views, views2=None, **kwargs):
        z = self(views)
        C = torch.cov(torch.hstack(z).T)
        Cxy = C[: self.latent_dims, self.latent_dims :]
        Cxx = C[: self.latent_dims, : self.latent_dims]
        if views2 is None:
            Cyy = C[self.latent_dims :, self.latent_dims :]
        else:
            z2 = self(views2)
            Cyy = torch.cov(torch.hstack(z2).T)[self.latent_dims :, self.latent_dims :]
        rewards = torch.trace(2 * Cxy)
        penalties = torch.trace(Cxx @ Cyy)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }



MODEL_DICT = {
    "DCCA": DCCA,
    "DCCANOI": DCCA_NOI,
    "DCCAEY": DCCA_EY,
    "DCCASVD": DCCA_SVD,
}

if __name__ == "__main__":
    wandb.init(config=defaults, project="DeepDeltaEigenGame")
    config = wandb.config
    seed_everything(config.random_seed)
    wandb_logger = WandbLogger()
    if config.data == "XRMB":
        feature_size = [273, 112]
        train_dataset = XRMB(root=os.getcwd(), train=True, download=False)
        test_dataset = XRMB(root=os.getcwd(), train=False, download=False)
    elif config.data == "SplitMNIST":
        feature_size = [392, 392]
        train_dataset = SplitMNIST(
            root=os.getcwd(), mnist_type=config.mnist_type, train=True, download=False
        )
        test_dataset = SplitMNIST(
            root=os.getcwd(), mnist_type=config.mnist_type, train=False, download=False
        )
    elif config.data == "NoisyMNIST":
        feature_size = [784, 784]
        train_dataset = NoisyMNIST(
            root=os.getcwd(), mnist_type=config.mnist_type, train=True, download=False
        )
        test_dataset = NoisyMNIST(
            root=os.getcwd(), mnist_type=config.mnist_type, train=False, download=False
        )
    elif config.data == "sim":
        feature_size = [100, 100]
        X = np.random.rand(1000, 100)
        Y = np.random.rand(1000, 100)
        # scale data and split into train and test
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
        train_dataset = NumpyDataset((X_train, Y_train))
        test_dataset = NumpyDataset((X_test, Y_test))
    else:
        raise ValueError("dataset not supported")
    if config.num_workers == 0:
        persistent_workers = False
    else:
        persistent_workers = True
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    if config.architecture == "linear":
        encoder_1 = architectures.LinearEncoder(
            latent_dims=config.latent_dims, feature_size=feature_size[0]
        )
        encoder_2 = architectures.LinearEncoder(
            latent_dims=config.latent_dims, feature_size=feature_size[1]
        )
    elif config.architecture == "nonlinear":
        encoder_1 = architectures.Encoder(
            latent_dims=config.latent_dims,
            layer_sizes=(800, 800),
            feature_size=feature_size[0],
        )
        encoder_2 = architectures.Encoder(
            latent_dims=config.latent_dims,
            layer_sizes=(800, 800),
            feature_size=feature_size[1],
        )
    else:
        raise ValueError("architecture not supported")
    if config.model == "DCCANOI":
        dcca = DCCA_NOI(
            latent_dims=config.latent_dims,
            N=len(train_dataset),
            encoders=[encoder_1, encoder_2],
            lr=config.lr,
            rho=config.rho,
            optimizer=config.optimizer,
        )
    elif config.model == "DCCA":
        dcca = DCCA(
            latent_dims=config.latent_dims,
            encoders=[encoder_1, encoder_2],
            lr=config.lr,
            rho=config.rho,
            optimizer=config.optimizer,
            objective=CCA,
        )
    else:
        dcca = MODEL_DICT[config.model](
            latent_dims=config.latent_dims,
            encoders=[encoder_1, encoder_2],
            lr=config.lr,
            optimizer=config.optimizer,
        )
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        gpus=device_count(),
        default_root_dir=os.path.join(
            os.getcwd(), config.model, f"{config.batch_size}", f"{config.lr}"
        ),
        enable_checkpointing=False,
        log_every_n_steps=10,
        accelerator="gpu",
        enable_progress_bar=False,
    )
    trainer.fit(dcca, train_loader, test_loader)
