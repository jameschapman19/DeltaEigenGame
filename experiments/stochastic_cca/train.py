import numpy as np
import pytorch_lightning as pl
from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.linear import rCCA, CCA_GHA, CCA_EY, CCA, MCCA
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import Callback

import wandb
from src.cca import GammaEigenGame
from src.data_utils import load_mnist, load_mediamill, load_cifar

# Define default hyperparameters
defaults = {
    "model": "gamma",
    "data": "cifar",
    "objective": "cca",
    "seed": 5,
    "components": 10,
    "batch_size": 100,
    "epochs": 5,
    "lr": 1e-4,
    "gamma": 1e-3,
    "optimizer": "SGD",
}

MODEL_DICT = {
    "cca": {
        "gamma": GammaEigenGame,
        "saa": rCCA,
        "gha": CCA_GHA,
        "ey": CCA_EY,
    },
}


class SampleCounterCallback(Callback):
    def __init__(self):
        self.samples_seen = 0

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int
    ):
        self.samples_seen += pl_module.batch_size
        pl_module.log("samples_seen", self.samples_seen)


class CorrelationCapturedCallback(Callback):
    def __init__(self, true_tcc, train_views, val_views=None):
        self.true_tcc = true_tcc
        self.train_views = train_views
        self.val_views = val_views

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        mcca = MCCA(pl_module.latent_dimensions)
        mcca.weights = [w.detach().cpu().numpy() for w in pl_module.torch_weights]
        mcca.n_views_ = len(self.train_views)
        z_train= mcca.transform(self.train_views)
        if self.val_views is not None:
            z_val = mcca.transform(self.val_views)
        train_tcc = mcca.fit(z_train).score(z_train)
        pl_module.log("train/TCC", train_tcc.sum())
        train_pcc = train_tcc.sum() / self.true_tcc["train"]
        pl_module.log("train/PCC", train_pcc)
        if self.val_views is not None:
            val_tcc = mcca.score(z_val)
            pl_module.log("val/TCC", val_tcc.sum())


def fit_cca(data_config, train_views):
    """Fit a CCA model."""
    cca = CCA(latent_dimensions=data_config.components).fit(train_views)
    return cca


def load_scores(filename):
    """Load and sum up the latent component scores."""
    return np.load(filename)


def try_load_or_fit_cca(data_config, train_views, X_test=None, Y_test=None):
    """Try loading existing scores, fit a new CCA model otherwise."""
    train_filename = f"results/{data_config.data}_{data_config.objective}_score_train.npy"
    val_filename = f"results/{data_config.data}_{data_config.objective}_score_test.npy"

    if data_config.data == "synthetic":
        cca = fit_cca(data_config, train_views)
        return {
            "train": cca.score(train_views),
            "val": cca.score([X_test, Y_test]),
        }
    elif data_config.data == "mediamill":
        try:
            # Try to load existing scores
            return {
                "train": load_scores(train_filename)
            }
        except FileNotFoundError:
            cca = fit_cca(data_config, train_views)
            np.save(train_filename, cca.score(train_views))
            return {
                "train": load_scores(train_filename)
            }
    else:
        try:
            # Try to load existing scores
            return {
                "train": load_scores(train_filename),
                "val": load_scores(val_filename),
            }
        except FileNotFoundError:
            cca = fit_cca(data_config, train_views)
            np.save(train_filename, cca.score(train_views))

            if X_test is not None:
                # Only save if the data is not synthetic
                save_score = cca.score((X_test, Y_test))
                np.save(f"results/{data_config.data}_cca_score_test.npy", save_score)

            return {
                "train": load_scores(train_filename),
                "val": load_scores(val_filename),
            }

def main():
    # Initialize wandb with the default configuration
    wandb.init(config=defaults, project="StochasticCCA")
    config = wandb.config

    np.random.seed(config.seed)

    # Load the data based on the data name
    if wandb.config.data == "synthetic":
        dataset=LinearSimulatedData(view_features=[10, 10], latent_dims=3)
        X, Y = dataset.sample(10000)
        X_test, Y_test = dataset.sample(1000)

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

    # Scale the data using standard scaler
    X = x_scaler.fit_transform(X)
    Y = y_scaler.fit_transform(Y)

    if X_test is not None:
        X_test = x_scaler.transform(X_test)
        Y_test = y_scaler.transform(Y_test)

    train_views = [X, Y]
    if X_test is not None:
        val_views = [X_test, Y_test]
    else:
        val_views = [X[:1000], Y[:1000]]

    true = try_load_or_fit_cca(wandb.config, train_views, X_test, Y_test)
    true["train"] = true["train"][:wandb.config.components].sum()
    if X_test is not None:
        true["val"] = true["val"][:wandb.config.components].sum()

    # Initialize the model based on the objective and model name
    model = MODEL_DICT[config.objective][config.model](
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.lr,
        latent_dimensions=config.components,
        random_state=config.seed,
        trainer_kwargs={"logger": WandbLogger(),
                        "callbacks": [CorrelationCapturedCallback(true, train_views, val_views=val_views),
                                      SampleCounterCallback()],
                        "enable_progress_bar": False,
                        "val_check_interval": 0.1,
                        "log_every_n_steps": 20},
        optimizer_kwargs={"optimizer": wandb.config.optimizer},

    )
    # Set the gamma parameter if using GammaEigenGame model
    if config.model == "gamma":
        model.gamma = config.gamma

    model.fit(train_views, validation_views=val_views, true=true)
    wandb.finish()


if __name__ == "__main__":
    main()
