import argparse

import numpy as np
from cca_zoo.models import CCA, PLS

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
        "--batch_size", type=int, default=8, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument(
        "--momentum", type=bool, default=0.5, help="Use Nesterov momentum"
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
        scale=False,
        centre=False,
    )

    if wandb.config.data == "synthetic":
        X = np.random.rand(100, 10)
        Y = np.random.rand(100, 10)
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

    from cca_zoo.models import PLSEigenGame

    # ppp=PLSEigenGame(latent_dims=wandb.config.components, scale=False, centre=False, epochs=100, learning_rate=1e-2).fit(((X, Y)))
    # m=tvc(ppp,(X,Y))
    # print()

    # S = np.linalg.svd(X.T @ Y / X.shape[0], compute_uv=False)
    if wandb.config.objective == "cca":
        C = CCA(latent_dims=wandb.config.components).fit(((X, Y)))
        true = {"train": C.score((X, Y)).sum(), "val": C.score((X_test, Y_test)).sum()}
    elif wandb.config.objective == "pls":
        U, S, Vt = np.linalg.svd(X.T @ Y / (X.shape[0] - 1), full_matrices=False)
        true = {"train": S[:wandb.config.components].sum(),
                "val": tvc((U[:, :wandb.config.components], Vt.T[:, :wandb.config.components]), (X_test, Y_test)).sum()}
    else:
        raise NotImplementedError
    model.fit([X, Y], val_views=[X_test, Y_test], true=true)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Train Models with Delta-EigenGame", parents=[get_arguments()]
    )
    args = parser.parse_args()
    args = vars(args)
    wandb.init(project="delta-eigengame", config=args, mode='offline')
    main()
    wandb.finish()
