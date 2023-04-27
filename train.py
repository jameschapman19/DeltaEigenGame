import argparse

import numpy as np
from cca_zoo.models import CCA, PLS
from scipy.linalg import svdvals

import cca
import pls
import wandb
from data_utils import cifar_dataset, mnist_dataset, mediamill_dataset, xrmb_dataset


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Train Models with Delta-EigenGame", add_help=False
    )

    # Experiment
    parser.add_argument(
        "--model", type=str, default="delta", help="Model to train"
    )
    parser.add_argument(
        "--data", type=str, default="mnist", help="Data directory"
    )
    parser.add_argument(
        "--objective", type=str, default="pls", help="Objective function"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed"
    )
    parser.add_argument(
        "--components", type=int, default=5, help="Number of components"
    )

    # Parameters
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--momentum", type=bool, default=0.9, help="Use Nesterov momentum"
    )

    # GammaEigenGame
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="Gamma"
    )

    return parser


MODEL_DICT = {
    "cca": {
        "sgha": cca.SGHA,
        "gamma": cca.GammaEigenGame,
        "delta": cca.DeltaEigenGame,
        "ghagep": cca.GHAGEP,
        "saa": CCA,
    },
    "pls": {
        "sgha": pls.SGHA,
        "gamma": pls.GammaEigenGame,
        "delta": pls.DeltaEigenGame,
        "ghagep": pls.GHAGEP,
        "sp": pls.StochasticPower,
        "saa": PLS,
    },
}


def tvc(weights, views):
    z = [view @ weight for view, weight in zip(views, weights)]
    m = z[0].T @ z[1] / (z[0].shape[0]-1)
    return np.diag(m)


def main():
    np.random.seed(wandb.config.seed)
    model = MODEL_DICT[wandb.config.objective][wandb.config.model](
        batch_size=wandb.config.batch_size,
        epochs=wandb.config.epochs,
        learning_rate=wandb.config.lr,
        latent_dims=wandb.config.components,
        momentum=wandb.config.momentum,
    )

    if wandb.config.data == "synthetic":
        X = np.random.rand(100, 10)
        Y = np.random.rand(100, 10)
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
        Y = StandardScaler().fit_transform(Y)
        X_test = np.random.rand(100, 10)
        Y_test = np.random.rand(100, 10)
    elif wandb.config.data == "cifar":
        X, Y, X_test, Y_test = cifar_dataset()
    elif wandb.config.data == "mnist":
        X, Y, X_test, Y_test = mnist_dataset()
    elif wandb.config.data == "xrmb":
        X, Y, X_test, Y_test = xrmb_dataset()
    elif wandb.config.data == "mediamill":
        X, Y, X_test, Y_test = mediamill_dataset()
    else:
        raise NotImplementedError

    if wandb.config.data == "synthetic":
        true = {"train": svdvals(X.T@Y)[:5].sum(), "val": tvc([np.eye(10), np.eye(10)], [X_test, Y_test]).sum()}
    else:
        true = {"train": np.load(f'./results/{wandb.config.data}_{wandb.config.objective}_score_train.npy').sum(), "val": np.load(f'./results/{wandb.config.data}_{wandb.config.objective}_score_test.npy').sum()}
    # log every 10% of an epoch for a given dataset and batch size
    log_every = int(X.shape[0] / wandb.config.batch_size / 10)
    model.fit([X, Y], val_views=[X_test, Y_test], true=true, log_every=log_every)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Train Models with Delta-EigenGame", parents=[get_arguments()]
    )
    args = parser.parse_args()
    args = vars(args)
    wandb.init(config=args, mode='offline')
    main()
    wandb.finish()
