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
)
from cca_zoo.deepmodels.objectives import CCA
from multiviewdata.torchdatasets import NoisyMNIST, SplitMNIST, XRMB
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.cuda import device_count
from torch.utils.data import random_split

from dcca import DCCA_EY, DCCA_SVD, DCCA_BarlowTwins

WANDB_START_METHOD = "thread"

# Define default configuration parameters for wandb
defaults = dict(
    data="SplitMNIST",
    mnist_type="MNIST",
    lr=0.001,
    batch_size=100,
    latent_dims=3,
    epochs=1000,
    model="DCCASVD",
    architecture="linear",
    rho=0.1,
    random_seed=1,
    optimizer="sgd",
    project="DeepCCA",
    num_workers=0,
)

# Define a dictionary to map model names to classes
MODEL_DICT = {
    "DCCA": DCCA,
    "DCCANOI": DCCA_NOI,
    "DCCAEY": DCCA_EY,
    "DCCASVD": DCCA_SVD,
    "DCCABARLOWTWINS": DCCA_BarlowTwins,
}


def main():
    """Main function to run the experiment"""
    # Initialize wandb with the default configuration
    wandb.init(config=defaults, project="DeepCCA")
    config = wandb.config

    # Set the random seed for reproducibility
    seed_everything(config.random_seed)

    # Create a wandb logger to track the experiment
    wandb_logger = WandbLogger()

    # Load the data according to the configuration parameter
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

        # Generate random data for simulation
        X = np.random.rand(1000, 100)
        Y = np.random.rand(1000, 100)

        # Scale data and split into train and test sets
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

        # Create numpy datasets from the data arrays
        train_dataset = NumpyDataset((X_train, Y_train))
        test_dataset = NumpyDataset((X_test, Y_test))
    else:
        raise ValueError("dataset not supported")

    # Create data loaders for training and testing
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

    # Create encoders according to the configuration parameter
    if config.architecture == "linear":
        # Use linear encoders for each view
        encoder_1 = architectures.LinearEncoder(
            latent_dims=config.latent_dims, feature_size=feature_size[0]
        )

        encoder_2 = architectures.LinearEncoder(
            latent_dims=config.latent_dims, feature_size=feature_size[1]
        )
    elif config.architecture == "nonlinear":

        # Use nonlinear encoders with hidden layers for each view
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

    # Create the model according to the configuration parameter
    if config.model == "DCCANOI":

        # Use DCCA with noise injection regularization
        dcca = DCCA_NOI(
            latent_dims=config.latent_dims,
            N=len(train_dataset),
            encoders=[encoder_1, encoder_2],
            lr=config.lr,
            rho=config.rho,
            optimizer=config.optimizer,
        )
    elif config.model == "DCCA":

        # Use standard DCCA with CCA objective
        dcca = DCCA(
            latent_dims=config.latent_dims,
            encoders=[encoder_1, encoder_2],
            lr=config.lr,
            rho=config.rho,
            optimizer=config.optimizer,
            objective=CCA,
        )
    else:

        # Use a custom model from the model dictionary
        dcca = MODEL_DICT[config.model](
            latent_dims=config.latent_dims,
            encoders=[encoder_1, encoder_2],
            lr=config.lr,
            optimizer=config.optimizer,
        )

    # Create a trainer to train and evaluate the model
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


if __name__ == "__main__":
    main()
