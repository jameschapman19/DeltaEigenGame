import os

import pytorch_lightning as pl
import torch
import wandb
from cca_zoo.deepmodels import (
    DCCA,
    architectures,
    DCCA_NOI,
    DCCA_SDL,
    BarlowTwins,
)
from cca_zoo.deepmodels.objectives import CCA
from multiviewdata.torchdatasets import NoisyMNIST, SplitMNIST, XRMB
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.cuda import device_count
from torch.utils.data import random_split

WANDB_START_METHOD="thread"
defaults = dict(
    data='SplitMNIST',
    mnist_type='MNIST',
    lr=0.001,
    batch_size=1000,
    latent_dims=50,
    epochs=30,
    model='DCCAGEPGD',
    architecture='nonlinear',
    rho=0.1,
    random_seed=42,
    optimizer='adam'
)

class DCCA_GEPGD(DCCA):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(self, latent_dims: int, encoders=None, r: float = 0, **kwargs):
        super().__init__(latent_dims=latent_dims, encoders=encoders, **kwargs)

    def forward(self, views, **kwargs):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(views[i]))
        return z

    def loss(self, views, **kwargs):
        z = self(views)
        A, B = self.get_AB(z)
        rewards = torch.trace(A)
        penalties = torch.trace(A.detach() @ B).sum()
        return {
            "objective": -rewards.sum() + penalties,
            "rewards": rewards.sum(),
            "penalties": penalties,
        }

    def get_AB(self, z):
        # sum the pairwise covariances between each z and all other zs
        A = torch.zeros(self.latent_dims, self.latent_dims, device=z[0].device)
        B = torch.zeros(self.latent_dims, self.latent_dims, device=z[0].device)
        for i, zi in enumerate(z):
            for j, zj in enumerate(z):
                if i == j:
                    B += torch.cov(zi.T)
                A += torch.cov(torch.hstack((zi, zj)).T)[
                    self.latent_dims :, : self.latent_dims
                ]
        return A / len(z), B / len(z)

MODEL_DICT = {
    'DCCA': DCCA,
    'DCCAGEPGD': DCCA_GEPGD,
    'DCCANOI': DCCA_NOI,
    'DCCASDL': DCCA_SDL,
    'BarlowTwins': BarlowTwins,
}

if __name__ == '__main__':
    wandb.init(config=defaults)
    config = wandb.config
    seed_everything(config.random_seed)
    wandb_logger = WandbLogger()
    if config.data == 'XRMB':
        feature_size = [273, 112]
        train_dataset = XRMB(root=os.getcwd(), train=True, download=True)
        test_dataset = XRMB(root=os.getcwd(), train=False, download=True)
    elif config.data == 'SplitMNIST':
        feature_size = [392, 392]
        train_dataset = SplitMNIST(root=os.getcwd(), mnist_type=config.mnist_type, train=True, download=True)
        test_dataset = SplitMNIST(root=os.getcwd(), mnist_type=config.mnist_type, train=False, download=True)
    elif config.data == 'NoisyMNIST':
        feature_size = [784, 784]
        train_dataset = NoisyMNIST(root=os.getcwd(), mnist_type=config.mnist_type, train=True, download=True)
        test_dataset = NoisyMNIST(root=os.getcwd(), mnist_type=config.mnist_type, train=False, download=True)
    else:
        raise ValueError('dataset not supported')
    n_train = int(0.8 * len(train_dataset))
    n_val = len(train_dataset) - n_train
    train_dataset, val_dataset = random_split(train_dataset, (n_train, n_val))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
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
        dcca = DCCA_NOI(N=len(train_dataset), latent_dims=config.latent_dims, encoders=[encoder_1, encoder_2],
                        lr=config.lr, rho=config.rho, optimizer=config.optimizer)
    elif config.model == 'DCCA':
        dcca = DCCA(latent_dims=config.latent_dims, encoders=[encoder_1, encoder_2],
                        lr=config.lr, rho=config.rho, optimizer=config.optimizer, objective=CCA)
    else:
        dcca = MODEL_DICT[config.model](latent_dims=config.latent_dims, encoders=[encoder_1, encoder_2], lr=config.lr, optimizer=config.optimizer)
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
    wandb.finish()