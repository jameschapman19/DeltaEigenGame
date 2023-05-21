import os

import pytorch_lightning as pl
import torch
import wandb
from cca_zoo.deepmodels import (
    DCCA,
    architectures,
    DCCA_NOI,
    DCCA_EigenGame,
)
from cca_zoo.deepmodels.objectives import CCA
from multiviewdata.torchdatasets import NoisyMNIST, SplitMNIST, XRMB
from cca_zoo.data.deep import NumpyDataset
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.cuda import device_count
from torch.utils.data import random_split
import numpy as np

WANDB_START_METHOD = "thread"
defaults = dict(
    data="SplitMNIST",
    mnist_type="MNIST",
    lr=0.0001,
    batch_size=8,
    latent_dims=50,
    epochs=50,
    model="DCCABT",
    architecture="nonlinear",
    rho=0.1,
    random_seed=1,
    optimizer="adam",
    project="DeepDeltaEigenGame",
    num_workers=8,
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
        if len(self.batch_queue) <2:
            self.batch_queue.append(batch)
            loss = {
                "objective": torch.tensor(0,requires_grad=True,dtype=torch.float32),
            }
        else:
            loss = self.loss(batch["views"], self.batch_queue[0]["views"],self.batch_queue[1]["views"])
        self.batch_queue.append(batch)
        self.batch_queue.pop(0)
        for k, v in loss.items():
            self.log("train/" + k, v, prog_bar=False)
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        if len(self.val_batch_queue) < 2:
            self.val_batch_queue.append(batch)
            loss = {
                "objective": torch.tensor(0, requires_grad=True,dtype=torch.float32),
            }
        else:
            loss = self.loss(batch["views"], self.val_batch_queue[0]["views"], self.val_batch_queue[1]["views"])
        # add current batch to queue and remove oldest batch
        self.batch_queue.append(batch)
        self.batch_queue.pop(0)
        for k, v in loss.items():
            self.log("val/" + k, v)
        return loss["objective"]

    def test_step(self, batch, batch_idx):
        loss = self.loss(batch["views"])
        for k, v in loss.items():
            self.log("test/" + k, v)
        return loss["objective"]

    def loss(self, views, views2=None, views3=None,  **kwargs):
        z = self(views)
        A, B = self.get_AB(z)
        # works better for DCCA_GH
        A = A - B
        if views2 is None:
            B2 = B
            B3 = B
        else:
            z2 = self(views2)
            A2, B2 = self.get_AB(z2)
            z3 = self(views3)
            A3, B3 = self.get_AB(z3)
        rewards = torch.trace(2 * A)
        penalties = torch.trace(B2@ B3)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

class DCCA_EY_BIASED(DCCA_EigenGame):
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

    def loss(self, views, **kwargs):
        z = self(views)
        A, B = self.get_AB(z)
        # works better for DCCA_GH
        A = A - B
        rewards = torch.trace(2 * A)
        penalties = torch.trace(B @ B)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

class DCCA_SVD(DCCA_EY):
    def loss(self,views, views2=None, views3=None, **kwargs):
        z = self(views)
        rewards=2*torch.trace(torch.cov(torch.hstack((z[0],z[1])).T)[:self.latent_dims,self.latent_dims:])
        penalties=torch.trace(torch.cov(z[0].T)@torch.cov(z[1].T))
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

class DCCA_BT(DCCA_EY):
    def loss(self,views, views2=None, views3=None,  **kwargs):
        z = self(views)
        # batchnorm the outputs z
        bn = [torch.nn.BatchNorm1d(self.latent_dims, affine=False).to(z[0].device) for _ in z]
        z = [bn_(z_) for bn_,z_ in zip(bn,z)]
        corr = torch.einsum("bi, bj -> ij", z[0], z[1]) / z[0].shape[0]

        diag = torch.eye(self.latent_dims, device=corr.device)
        cdif = (corr - diag).pow(2)
        lamb=5e-3
        cdif[~diag.bool()] *= lamb
        loss = cdif.sum()
        return {
            "objective": loss,
        }

MODEL_DICT = {
    "DCCA": DCCA,
    "DCCAEYUNBIASED": DCCA_EY,
    #"DCCAGH": DCCA_GH,
    "DCCANOI": DCCA_NOI,
    "DCCASVD": DCCA_SVD,
    "DCCABT": DCCA_BT,
    "DCCAEYBIASED": DCCA_EY_BIASED,
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
        train_dataset = NumpyDataset(
            (np.random.rand(1000, 100), np.random.rand(1000, 100))
            )
        test_dataset = NumpyDataset(
            (np.random.rand(1000, 100), np.random.rand(1000, 100))
            )
    else:
        raise ValueError("dataset not supported")
    n_train = int(0.8 * len(train_dataset))
    n_val = len(train_dataset) - n_train
    train_dataset, val_dataset = random_split(train_dataset, (n_train, n_val))
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
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
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
    trainer.fit(dcca, train_loader, val_loader)
