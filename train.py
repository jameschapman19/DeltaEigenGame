import argparse

import numpy as np
from cca_zoo.models import rCCA, PLS
from scipy.linalg import svdvals

import cca
import pls
import wandb

from data_utils import load_mnist, load_mediamill, load_cifar


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Train Models with Delta-EigenGame", add_help=False
    )

    # Experiment
    parser.add_argument("--model", type=str, default="gamma", help="Model to train")
    parser.add_argument("--data", type=str, default="cifar", help="Data directory")
    parser.add_argument(
        "--objective", type=str, default="cca", help="Objective function"
    )
    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument(
        "--components", type=int, default=4, help="Number of components"
    )

    # Parameters
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--momentum", type=bool, default=0, help="Use Nesterov momentum"
    )

    # GammaEigenGame
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma")

    return parser


MODEL_DICT = {
    "cca": {
        "sgha": cca.SGHA,
        "gamma": cca.GammaEigenGame,
        "delta": cca.DeltaEigenGame,
        "gha": cca.GHAGEP,
        "saa": rCCA,
        "subspace": cca.Subspace,
    },
    "pls": {
        "sgha": pls.SGHA,
        "gamma": pls.GammaEigenGame,
        "delta": pls.DeltaEigenGame,
        "gha": pls.GHAGEP,
        "sp": pls.StochasticPower,
        "saa": PLS,
        "subspace": pls.Subspace,
    },
}


def tvc(weights, views):
    z = [view @ weight for view, weight in zip(views, weights)]
    m = z[0].T @ z[1] / (z[0].shape[0] - 1)
    return np.diag(m)


def main():
    np.random.seed(wandb.config.seed)
    model = MODEL_DICT[wandb.config.objective][wandb.config.model](
        batch_size=wandb.config.batch_size,
        epochs=wandb.config.epochs,
        learning_rate=wandb.config.lr,
        latent_dims=wandb.config.components,
        momentum=wandb.config.momentum,
        random_state=wandb.config.seed,
        scale=False,
        centre=False,
    )
    if wandb.config.model == "gamma":
        model.gamma = wandb.config.gamma

    if wandb.config.data == "synthetic":
        X = np.random.rand(100, 10)
        Y = np.random.rand(100, 10)
        X_test = np.random.rand(100, 10)
        Y_test = np.random.rand(100, 10)
    elif wandb.config.data == "cifar":
        X, Y, X_test, Y_test = load_cifar()
    elif wandb.config.data == "mnist":
        X, Y, X_test, Y_test = load_mnist()
    elif wandb.config.data == "mediamill":
        X, Y, X_test, Y_test = load_mediamill()
    else:
        raise NotImplementedError

    from sklearn.preprocessing import StandardScaler

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)
    Y = y_scaler.fit_transform(Y)
    if X_test is not None:
        X_test = x_scaler.transform(X_test)
        Y_test = y_scaler.transform(Y_test)

    if wandb.config.data == "synthetic":
        true = {
            "train": svdvals(X.T @ Y)[:5].sum(),
            "val": tvc([np.eye(10), np.eye(10)], [X_test, Y_test]).sum(),
        }
    else:
        true = {
            "train": np.load(
                f"./results/{wandb.config.data}_{wandb.config.objective}_score_train.npy"
            )[: wandb.config.components].sum(),
            "val": np.load(
                f"./results/{wandb.config.data}_{wandb.config.objective}_score_test.npy"
            )[: wandb.config.components].sum(),
        }
    # log every 5% of an epoch for a given dataset and batch size
    # log_every = int((X.shape[0] / 20))
    # round down X.shape[0] to the nearest 100
    log_every = int((X.shape[0] // 100) / 20)
    if X_test is not None:
        model.fit([X, Y], val_views=[X_test, Y_test], true=true, log_every=log_every)
    else:
        model.fit([X, Y], true=true, log_every=log_every)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Train Models with Delta-EigenGame", parents=[get_arguments()]
    )
    args = parser.parse_args()
    args = vars(args)
    wandb.init(config=args, mode="online")
    main()
    wandb.finish()
