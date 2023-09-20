import numpy as np
import wandb
from cca_zoo.linear import rCCA, CCA_GHA, CCA_EY
from pytorch_lightning import Callback
from sklearn.preprocessing import StandardScaler

# Define default hyperparameters
defaults = {
    "model": "ey",
    "data": "cifar",
    "objective": "cca",
    "seed": 5,
    "components": 4,
    "batch_size": 100,
    "epochs": 10,
    "lr": 1e-3,
}

MODEL_DICT = {
    "cca": {
        "saa": rCCA,
        "gha": CCA_GHA,
        "ey": CCA_EY,
    },
}

class TrueCallback(Callback):
    def __init__(self, true):
        self.true = true

    def on_epoch_end(self, epoch, logs=None):
        pass
def main():
    # Initialize wandb with the default configuration
    wandb.init(config=defaults, project="StochasticMCCA")
    config = wandb.config

    np.random.seed(config.seed)

    # Initialize the model based on the objective and model name
    model = MODEL_DICT[config.objective][config.model](
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.lr,
        latent_dimensions=config.components,
        random_state=config.seed,
    )

    # Set the gamma parameter if using GammaEigenGame model
    if config.model == "gamma":
        model.gamma = config.gamma

    # Load the data based on the data name
    # Dummy synthetic data for illustration
    if config.data == "synthetic":
        views = [np.random.rand(100, 10) for _ in range(3)]
    else:
        raise NotImplementedError

    # Apply standard scaling to each view
    scaled_views = []
    for view in views:
        scaler = StandardScaler()
        scaled_view = scaler.fit_transform(view)
        scaled_views.append(scaled_view)

    # Determine when to log the progress
    log_every = int(scaled_views[0].shape[0] / config.batch_size / 20)

    # Fit the model
    model.fit(scaled_views, log_every=log_every)

    print()


if __name__ == "__main__":
    main()