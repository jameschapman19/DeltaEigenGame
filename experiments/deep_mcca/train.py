import os

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from cca_zoo.deep import DCCA, DCCA_EY, architectures
from cca_zoo.deep.callbacks import (
    BatchTrainCorrelationCallback,
    BatchValidationCorrelationCallback,
)
from cca_zoo.deep.objectives import GCCA, MCCA
from cca_zoo.linear._gradient._ey import DoubleNumpyDataset
from multiviewdata.torchdatasets import Twitter
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split

from src.src.data_utils import MFeat

WANDB_START_METHOD = "thread"

# Define default configuration parameters for wandb
defaults = dict(
    data="mfeat",
    mnist_type="MNIST",
    lr=0.001,
    batch_size=10,
    epochs=50,
    model="DCCAEY",
    architecture="nonlinear",
    rho=0.9,
    random_seed=1,
    optimizer="adam",
    num_workers=4,
)


class IndependentMixin:
    random_state = np.random.RandomState(0)

    def __getitem__(self, index):
        views = super().__getitem__(index)
        independent_index = self.random_state.randint(0, len(self))
        independent_views = super().__getitem__(independent_index)
        return {
            "views": views["views"],
            "independent_views": independent_views["views"],
        }


class MFeat_(IndependentMixin, MFeat):
    pass


class Twitter_(IndependentMixin, Twitter):
    pass


# Define a dictionary to map model names to classes
MODEL_DICT = {
    "DCCAEY": DCCA_EY,
}


def main():
    """Main function to run the experiment"""
    # Initialize wandb with the default configuration
    wandb.init(config=defaults, project="DeepMCCA")
    config = wandb.config

    # Set the random seed for reproducibility
    seed_everything(config.random_seed)

    # Create a wandb logger to track the experiment
    wandb_logger = WandbLogger()

    if config.data == "mfeat":
        feats= ["fac", "fou", "kar", "pix", "zer"]
        feature_size = [216, 76, 64, 240, 47]
        dataset = MFeat(root=os.getcwd(), download=True, feats=feats)
        import numpy as np
        dataset.dataset["fac"] = dataset.dataset["fac"]
        dataset.dataset["fou"] = dataset.dataset["fou"]
        dataset.dataset["kar"] = dataset.dataset["kar"]
        dataset.dataset["mor"] = dataset.dataset["mor"]
        dataset.dataset["pix"] = dataset.dataset["pix"]
        dataset.dataset["zer"] = dataset.dataset["zer"]
        # split into train and test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
        )
        latent_dims = 50
    elif config.data == "twitter":
        feature_size = [100, 100, 100, 100, 100, 100]
        dataset = Twitter_(root=os.getcwd(), download=True, maxrows=120000)
        # split into train and test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
        )
        latent_dims = 5
    elif config.data == "sim":
        import numpy as np

        feature_size = [128, 128, 128,  128]
        X = np.random.randn(1000, 128)
        Y = np.random.randn(1000, 128)
        Z = np.random.randn(1000, 128)
        A = np.random.randn(1000, 128)
        train_dataset = DoubleNumpyDataset((X, Y, Z, A))
        test_dataset = DoubleNumpyDataset((X, Y, Z, A))
        latent_dims = 10
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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    # Create encoders according to the configuration parameter
    if config.architecture == "linear":
        # Use linear encoders for each view
        encoders = [architectures.LinearEncoder(
            latent_dimensions=latent_dims, feature_size=f
        ) for f in feature_size]
    elif config.architecture == "nonlinear":
        # Use nonlinear encoders with hidden layers for each view
        encoders = [architectures.Encoder(
            latent_dimensions=latent_dims,
            layer_sizes=(800,800,),
            feature_size=f
        ) for f in feature_size]
    else:
        raise ValueError("architecture not supported")

    if config.model == "DGCCA":
        # Use standard DCCA with CCA objective
        dcca = DCCA(
            latent_dimensions=latent_dims,
            encoders=encoders,
            lr=config.lr,
            optimizer=config.optimizer,
            objective=GCCA,
        )
    elif config.model == "DMCCA":
        # Use standard DCCA with CCA objective
        dcca = DCCA(
            latent_dimensions=latent_dims,
            encoders=encoders,
            lr=config.lr,
            optimizer=config.optimizer,
            objective=MCCA,
        )
    elif config.model == "DCCAEY":
        # Use a custom model from the model dictionary
        dcca = DCCA_EY(
            latent_dimensions=latent_dims,
            encoders=encoders,
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
        enable_progress_bar=False,
        callbacks=[
            BatchTrainCorrelationCallback(),
            BatchValidationCorrelationCallback(),
        ],  # , EYCallback((X,Y))],
    )

    trainer.fit(dcca, train_loader, test_loader)
    wandb.finish()


if __name__ == "__main__":
    main()
