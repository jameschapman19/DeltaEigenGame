import os

import pytorch_lightning as pl
import torch
from cca_zoo.data.deep import NumpyDataset
from cca_zoo.deep import DCCA, DCCA_NOI, DCCA_EY, architectures
from cca_zoo.deep.callbacks import (
    BatchTrainCorrelationCallback,
    BatchValidationCorrelationCallback,
)
from cca_zoo.deep.objectives import CCA
from multiviewdata.torchdatasets import NoisyMNIST, SplitMNIST, XRMB
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split

import wandb

WANDB_START_METHOD = "thread"

# Define default configuration parameters for wandb
defaults = dict(
    data="sim",
    mnist_type="MNIST",
    lr=0.00001,
    batch_size=100,
    latent_dims=50,
    epochs=25,
    model="DCCAEY",
    architecture="nonlinear",
    rho=0.1,
    random_seed=1,
    optimizer="adam",
    num_workers=4,
)

# Define a dictionary to map model names to classes
MODEL_DICT = {
    "DCCA": DCCA,
    "DCCANOI": DCCA_NOI,
    "DCCAEY": DCCA_EY,
}

# class EYCallback(pl.Callback):
#     def __init__(self, train_views):
#         self.train_views = train_views
#         self.train_views = [torch.Tensor(view) for view in self.train_views]
#
#     def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         self.train_views = [view.to(pl_module.device) for view in self.train_views]
#         with torch.no_grad():
#             loss=pl_module.loss(self.train_views)
#             for k, v in loss.items():
#                 # Use f-string instead of concatenation
#                 self.log(
#                     f"batch/{k}",
#                     v,
#                 )


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
        train_dataset = XRMB(root=os.getcwd(), train=True, download=True)
        test_dataset = XRMB(root=os.getcwd(), train=False, download=True)
    elif config.data == "SplitMNIST":
        feature_size = [392, 392]
        train_dataset = SplitMNIST(
            root=os.getcwd(), mnist_type=config.mnist_type, train=True, download=True
        )
        test_dataset = SplitMNIST(
            root=os.getcwd(), mnist_type=config.mnist_type, train=False, download=True
        )
    elif config.data == "NoisyMNIST":
        feature_size = [784, 784]
        train_dataset = NoisyMNIST(
            root=os.getcwd(), mnist_type=config.mnist_type, train=True, download=True
        )
        test_dataset = NoisyMNIST(
            root=os.getcwd(), mnist_type=config.mnist_type, train=False, download=True
        )
    elif config.data == "sim":
        import numpy as np
        feature_size = [784,784]
        X=np.random.randn(1000,784)
        Y=np.random.randn(1000,784)
        train_dataset = NumpyDataset((X,Y))
        test_dataset = NumpyDataset((X,Y))
    else:
        raise ValueError("dataset not supported")

    # if running on windows then set config.num_workers = 0
    if os.name == "nt":
        num_workers = 0
    else:
        num_workers = config.num_workers

    # Create data loaders for training and testing
    if num_workers == 0:
        persistent_workers = False
    else:
        persistent_workers = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    # Create encoders according to the configuration parameter
    if config.architecture == "linear":
        # Use linear encoders for each view
        encoder_1 = architectures.LinearEncoder(
            latent_dimensions=config.latent_dims, feature_size=feature_size[0]
        )

        encoder_2 = architectures.LinearEncoder(
            latent_dimensions=config.latent_dims, feature_size=feature_size[1]
        )
    elif config.architecture == "nonlinear":

        # Use nonlinear encoders with hidden layers for each view
        encoder_1 = architectures.Encoder(
            latent_dimensions=config.latent_dims,
            layer_sizes=(800, 800),
            feature_size=feature_size[0],
        )
        encoder_2 = architectures.Encoder(
            latent_dimensions=config.latent_dims,
            layer_sizes=(800, 800),
            feature_size=feature_size[1],
        )
    else:
        raise ValueError("architecture not supported")

    # Create the model according to the configuration parameter
    if config.model == "DCCANOI":

        # Use DCCA with noise injection regularization
        dcca = DCCA_NOI(
            latent_dimensions=config.latent_dims,
            N=len(train_dataset),
            encoders=[encoder_1, encoder_2],
            lr=config.lr,
            rho=config.rho,
            optimizer=config.optimizer,
        )
    elif config.model == "DCCA":

        # Use standard DCCA with CCA objective
        dcca = DCCA(
            latent_dimensions=config.latent_dims,
            encoders=[encoder_1, encoder_2],
            lr=config.lr,
            rho=config.rho,
            optimizer=config.optimizer,
            objective=CCA,
        )
    else:

        # Use a custom model from the model dictionary
        dcca = MODEL_DICT[config.model](
            latent_dimensions=config.latent_dims,
            encoders=[encoder_1, encoder_2],
            lr=config.lr,
            optimizer=config.optimizer,
        )

    # Create a trainer to train and evaluate the model
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        default_root_dir=os.path.join(
            os.getcwd(), config.model, f"{config.batch_size}", f"{config.lr}"
        ),
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[BatchTrainCorrelationCallback(), BatchValidationCorrelationCallback()],#, EYCallback((X,Y))],
    )

    trainer.fit(dcca, train_loader, test_loader)
    wandb.finish()

if __name__ == "__main__":
    main()
