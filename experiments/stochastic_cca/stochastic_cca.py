import numpy as np
import pytorch_lightning as pl
from cca_zoo.linear import rCCA, CCA_GHA, CCA_EY, CCA, MCCA
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import Callback

import wandb
from src import cca
from src.data_utils import load_mnist, load_mediamill, load_cifar

# Define default hyperparameters
defaults = {
    "model": "gamma",
    "data": "cifar",
    "objective": "cca",
    "seed": 5,
    "components": 4,
    "batch_size": 100,
    "epochs": 10,
    "lr": 1e-1,
    "gamma": 0.1,
}

MODEL_DICT = {
    "cca": {
        "gamma": cca.GammaEigenGame,
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
    def __init__(self, true_tcc,train_views, val_views=None):
        self.true_tcc = true_tcc
        self.train_views=train_views
        self.val_views=val_views

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        mcca=MCCA(pl_module.latent_dimensions)
        mcca.weights = [w.detach().cpu().numpy() for w in pl_module.torch_weights]
        mcca.n_views_ = len(self.train_views)
        train_tcc = mcca.score(self.train_views)
        pl_module.log("train/TCC", train_tcc.sum())
        train_pcc = train_tcc.sum()/self.true_tcc["train"]
        pl_module.log("train/PCC", train_pcc)
        if self.val_views is not None:
            val_tcc = mcca.score(self.val_views)
            pl_module.log("val/TCC", val_tcc.sum())


def main():
    # Initialize wandb with the default configuration
    wandb.init(config=defaults, project="StochasticCCA")
    config = wandb.config

    np.random.seed(config.seed)

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
        
    train_views = [X, Y]
    val_views = [X_test, Y_test]

    try:
        true = {
            "train": np.load(
                f"results/{wandb.config.data}_{wandb.config.objective}_score_train.npy"
            )[: wandb.config.components].sum(),
            "val": np.load(
                f"results/{wandb.config.data}_{wandb.config.objective}_score_test.npy"
            )[: wandb.config.components].sum(),
        }
    except FileNotFoundError:
        cca = CCA(latent_dimensions=wandb.config.components).fit(train_views)
        cca_score_train = cca.score(train_views)
        np.save(f"results/{wandb.config.data}_cca_score_train.npy", cca_score_train)
        if X_test is not None:
            cca_score_test = cca.score((X_test, Y_test))
            np.save(f"results/{wandb.config.data}_cca_score_test.npy", cca_score_test)
        true = {
            "train": np.load(
                f"results/{wandb.config.data}_{wandb.config.objective}_score_train.npy"
            )[: wandb.config.components].sum(),
            "val": np.load(
                f"results/{wandb.config.data}_{wandb.config.objective}_score_test.npy"
            )[: wandb.config.components].sum(),
        }



    # we would like to log every 5% of an epoch as measured in batches
    log_every = int(X.shape[0] / wandb.config.batch_size / 20)
    # Initialize the model based on the objective and model name
    model = MODEL_DICT[config.objective][config.model](
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.lr,
        latent_dimensions=config.components,
        random_state=config.seed,
        trainer_kwargs={"logger": WandbLogger(),
                        "callbacks": [CorrelationCapturedCallback(true, train_views, val_views=val_views), SampleCounterCallback()],
                        "enable_progress_bar": True,
                        "val_check_interval": 0.1}
    )
    # Set the gamma parameter if using GammaEigenGame model
    if config.model == "gamma":
        model.gamma = config.gamma

    if X_test is not None:
        model.fit(train_views, validation_views=val_views, true=true)
    else:
        model.fit(train_views, true=true)


if __name__ == "__main__":
    main()
