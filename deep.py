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
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.cuda import device_count
from torch.utils.data import random_split

WANDB_START_METHOD = "thread"
defaults = dict(
    data='SplitMNIST',
    mnist_type='MNIST',
    lr=0.0001,
    batch_size=100,
    latent_dims=50,
    epochs=50,
    model='DCCAEY',
    architecture='nonlinear',
    rho=0.1,
    random_seed=1,
    optimizer='adam',
    project='DeepDeltaEigenGame',
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
        self.previous_batch = None
        self.val_previous_batch = None

    def forward(self, views, **kwargs):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(views[i]))
        return z

    def training_step(self, batch, batch_idx):
        if self.previous_batch is None:
            self.previous_batch = batch
        loss = self.loss(batch["views"], self.previous_batch["views"])
        self.previous_batch = batch
        for k, v in loss.items():
            self.log("train/" + k, v, prog_bar=False)
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        if self.val_previous_batch is None:
            self.val_previous_batch = batch
        loss = self.loss(batch["views"], self.val_previous_batch["views"])
        for k, v in loss.items():
            self.log("val/" + k, v)
        return loss["objective"]

    def test_step(self, batch, batch_idx):
        loss = self.loss(batch["views"])
        for k, v in loss.items():
            self.log("test/" + k, v)
        return loss["objective"]

    def get_AB(self, z):
        # sum the pairwise covariances between each z and all other zs
        A = torch.zeros(self.latent_dims, self.latent_dims, device=z[0].device)
        B = torch.zeros(self.latent_dims, self.latent_dims, device=z[0].device)
        for i, zi in enumerate(z):
            for j, zj in enumerate(z):
                if i == j:
                    B += torch.cov(zi.T)
                A += torch.cov(torch.hstack((zi, zj)).T)[
                     self.latent_dims:, : self.latent_dims
                     ]
        return A, B

    def loss(self, views, views2=None, **kwargs):
        z = self(views)
        A, B = self.get_AB(z)
        if views2 is None:
            B2 = B
        else:
            z2 = self(views2)
            A2, B2 = self.get_AB(z2)
        rewards = torch.trace(2 * A)
        penalties = torch.trace(B @ B2)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }


class DCCA_GH(DCCA_EY):
    def loss(self, views, views2=None, **kwargs):
        z = self(views)
        A, B = self.get_AB(z)
        if views2 is None:
            B2 = B
        else:
            z2 = self(views2)
            A2, B2 = self.get_AB(z2)
        rewards = torch.trace(2 * A)
        penalties = torch.trace(A @ B2)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }


MODEL_DICT = {
    'DCCA': DCCA,
    'DCCAEY': DCCA_EY,
    'DCCAGH': DCCA_GH,
    'DCCANOI': DCCA_NOI,
}

if __name__ == '__main__':
    wandb.init(config=defaults, project='DeepDeltaEigenGame')
    config = wandb.config
    seed_everything(config.random_seed)
    wandb_logger = WandbLogger(
    )
    if config.data == 'XRMB':
        feature_size = [273, 112]
        train_dataset = XRMB(root=os.getcwd(), train=True, download=False)
        test_dataset = XRMB(root=os.getcwd(), train=False, download=False)
    elif config.data == 'SplitMNIST':
        feature_size = [392, 392]
        train_dataset = SplitMNIST(root=os.getcwd(), mnist_type=config.mnist_type, train=True, download=False)
        test_dataset = SplitMNIST(root=os.getcwd(), mnist_type=config.mnist_type, train=False, download=False)
    elif config.data == 'NoisyMNIST':
        feature_size = [784, 784]
        train_dataset = NoisyMNIST(root=os.getcwd(), mnist_type=config.mnist_type, train=True, download=False)
        test_dataset = NoisyMNIST(root=os.getcwd(), mnist_type=config.mnist_type, train=False, download=False)
    else:
        raise ValueError('dataset not supported')
    n_train = int(0.8 * len(train_dataset))
    n_val = len(train_dataset) - n_train
    train_dataset, val_dataset = random_split(train_dataset, (n_train, n_val))
    if config.num_workers == 0:
        persistent_workers = False
    else:
        persistent_workers = True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,
                                               num_workers=config.num_workers, pin_memory=True,
                                               persistent_workers=persistent_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=True, persistent_workers=persistent_workers)
    if config.architecture == 'linear':
        encoder_1 = architectures.LinearEncoder(latent_dims=config.latent_dims, feature_size=feature_size[0])
        encoder_2 = architectures.LinearEncoder(latent_dims=config.latent_dims, feature_size=feature_size[1])
    elif config.architecture == 'nonlinear':
        encoder_1 = architectures.Encoder(latent_dims=config.latent_dims, layer_sizes=(800, 800),
                                          feature_size=feature_size[0])
        encoder_2 = architectures.Encoder(latent_dims=config.latent_dims, layer_sizes=(800, 800),
                                          feature_size=feature_size[1])
    else:
        raise ValueError('architecture not supported')
    if config.model == 'DCCANOI':
        dcca = DCCA_NOI(latent_dims=config.latent_dims, N=len(train_dataset), encoders=[encoder_1, encoder_2],
                        lr=config.lr, rho=config.rho, optimizer=config.optimizer)
    elif config.model == 'DCCA':
        dcca = DCCA(latent_dims=config.latent_dims, encoders=[encoder_1, encoder_2],
                    lr=config.lr, rho=config.rho, optimizer=config.optimizer, objective=CCA)
    else:
        dcca = MODEL_DICT[config.model](latent_dims=config.latent_dims, encoders=[encoder_1, encoder_2], lr=config.lr,
                                        optimizer=config.optimizer)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        gpus=device_count(),
        default_root_dir=os.path.join(os.getcwd(), config.model, f"{config.batch_size}",
                                      f"{config.lr}"),
        enable_checkpointing=False,
        log_every_n_steps=1000,
        accelerator='gpu',
        enable_progress_bar=False,
    )
    trainer.fit(dcca, train_loader, val_loader)
