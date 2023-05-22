# This script trains different models for CCA or PLS objectives on various datasets
# using Delta-EigenGame algorithm

import argparse

import numpy as np
import wandb  # module for logging and tracking experiments
from cca_zoo.models import rCCA, PLS, CCA

import cca  # custom module for CCA models
from data_utils import (
    load_mnist,
    load_mediamill,
    load_cifar,
)  # custom module for loading datasets


def get_arguments():
    # This function parses the command-line arguments and returns a parser object
    parser = argparse.ArgumentParser(
        description="Train Stochastic CCA Models", add_help=False
    )

    # Experiment
    parser.add_argument(
        "--model", type=str, default="gepey", help="Model to train"
    )
    parser.add_argument("--data", type=str, default="cifar", help="Data directory")
    parser.add_argument(
        "--objective", type=str, default="cca", help="Objective function"
    )
    parser.add_argument("--seed", type=int, default=5, help="Random seed")
    parser.add_argument(
        "--components", type=int, default=10, help="Number of components"
    )

    # Parameters
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
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
        "saa": rCCA,
        "gepgh": cca.GEPGH,
        "gepghbiased": cca.GEPGHBiased,
        "gepey": cca.GEPEY,
        "gepeybiased": cca.GEPEYBiased,
        "svd": cca.SVD,
    },
}


# This function computes the total variance captured (TVC) by the weights of the views
def tvc(weights, views):
    z = [view @ weight for view, weight in zip(views, weights)]
    m = z[0].T @ z[1] / (z[0].shape[0] - 1)
    return np.diag(m)


def main():
    np.random.seed(wandb.config.seed)

    # Initialize the model based on the objective and model name
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

    # Set the gamma parameter if using GammaEigenGame model
    if wandb.config.model == "gamma":
        model.gamma = wandb.config.gamma

    # Load the data based on the data name
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

    elif wandb.config.data == "synthetic":
        X = np.random.rand(100, 10)
        Y = np.random.rand(100, 10)
        X_test = np.random.rand(100, 10)
        Y_test = np.random.rand(100, 10)

    else:
        raise NotImplementedError

    from sklearn.preprocessing import StandardScaler

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Scale the data using standard scaler
    X = x_scaler.fit_transform(X)
    Y = y_scaler.fit_transform(Y)

    if X_test is not None:
        X_test = x_scaler.transform(X_test)
        Y_test = y_scaler.transform(Y_test)

    try:
        true = {
            "train": np.load(
                f"./results/{wandb.config.data}_{wandb.config.objective}_score_train.npy"
            )[: wandb.config.components].sum(),
            "val": np.load(
                f"./results/{wandb.config.data}_{wandb.config.objective}_score_test.npy"
            )[: wandb.config.components].sum(),
        }
    except FileNotFoundError:
        cca = CCA(latent_dims=4, scale=False, centre=False).fit((X, Y))
        cca_score_train = cca.score((X, Y))
        np.save(f"./results/{wandb.config.data}_cca_score_train.npy", cca_score_train)
        if X_test is not None:
            cca_score_test = cca.score((X_test, Y_test))
            np.save(f"./results/{wandb.config.data}_cca_score_test.npy", cca_score_test)
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
        "Train Stochastic CCA Models", parents=[get_arguments()]
    )
    args = parser.parse_args()
    args = vars(args)
    wandb.init(config=args)
    main()
    wandb.finish()
