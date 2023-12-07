# %%
# Imports
# -------

import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything

import numpy as np
from tqdm import tqdm
from cca_zoo.deep import BarlowTwins, architectures, DCCA_EY
from cca_zoo.deep._discriminative import VICReg
from docs.source.examples import example_mnist_data
from cca_zoo.utils import cross_corrcoef
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.5)

seed_everything(42)
LATENT_DIMS = 10  # The dimensionality of the latent space
EPOCHS = 50  # The number of epochs to train the models
N_TRAIN = 1000  # The number of training samples
N_VAL = 200  # The number of validation samples
LR = 1e-4
train_loader, val_loader, train_labels, val_labels = example_mnist_data(N_TRAIN, N_VAL)

encoder_1 = architectures.LinearEncoder(latent_dimensions=LATENT_DIMS, feature_size=392)
encoder_2 = architectures.LinearEncoder(latent_dimensions=LATENT_DIMS, feature_size=392)


def latent_spectrum(latent_vectors_corr):
    """
    Calculate the latent spectrum of a set of latent vectors.
    """
    # Calculate the eigenvalues of the covariance matrix
    eigvals = np.linalg.svd(latent_vectors_corr)[1]

    # Sort the eigenvalues in descending order
    eigvals = np.sort(eigvals)[::-1]

    return eigvals


def plot_latent_spectrum(latent_vectors, title=None, save_path=None):
    """
    Plot the latent spectrum of a set of latent vectors using Seaborn.
    """
    # Calculate the latent spectrum
    eigvals = latent_spectrum(cross_corrcoef(*latent_vectors, rowvar=False))

    # Set Seaborn style
    sns.set(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # Create a Seaborn plot
    plt.figure()
    plt.title(title)
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    sns.lineplot(x=range(1, len(eigvals) + 1), y=eigvals)
    # Add text with the sum of the eigenvalues
    plt.text(
        0.5,
        0.9,
        f"Sum of Eigenvalues: {np.sum(eigvals):.2f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show(block=False)


def experiment(train_loader, val_loader, encoder_1, encoder_2, lamb_values, cov_values):

    # Train EY
    model = DCCA_EY(
        latent_dimensions=LATENT_DIMS, encoders=[encoder_1, encoder_2], lr=LR
    )
    trainer = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False)
    trainer.fit(model, train_loader, val_loader)

    # Get latent vectors and calculate sparsity
    latent_vectors = model.transform(val_loader)
    plot_latent_spectrum(
        latent_vectors, title="EY Latent Spectrum", save_path="plots/spectrum/ey_latent_spectrum.pdf"
    )

    for cov in tqdm(cov_values, desc="VICReg Experiment"):
        # Train model
        model = VICReg(
            latent_dimensions=LATENT_DIMS,
            N=N_TRAIN,
            encoders=[encoder_1, encoder_2],
            sim_loss_weight=25,
            cov_loss_weight=cov,
            var_loss_weight=25,
            lr=LR,
        )
        trainer = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False)
        trainer.fit(model, train_loader, val_loader)

        # Get latent vectors and calculate sparsity
        latent_vectors = model.transform(val_loader)
        plot_latent_spectrum(
            latent_vectors,
            title=f"VICReg Latent Spectrum (Cov={cov})",
            save_path=f"plots/spectrum/vicreg_latent_spectrum_cov_{cov}.pdf",
        )

    for lamb in tqdm(lamb_values, desc="Barlow Twins Experiment"):
        # Train model
        model = BarlowTwins(
            latent_dimensions=LATENT_DIMS,
            encoders=[encoder_1, encoder_2],
            lamb=lamb,
            lr=LR,
        )
        trainer = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False)
        trainer.fit(model, train_loader, val_loader)

        # Get latent vectors and calculate sparsity
        latent_vectors = model.transform(val_loader)
        plot_latent_spectrum(
            latent_vectors,
            title=f"Barlow Twins Latent Spectrum (Lamb={lamb})",
            save_path=f"plots/spectrum/barlow_twins_latent_spectrum_lamb_{lamb}.pdf",
        )


# Hyperparameters for the experiment
lamb_values = [1e-7,1e-5,1e-3, 1e-1, 10]
cov_values = [1e-5,1e-3, 1e-1, 10]

# Run the experiment
sparsity_results_barlow, sparsity_results_vicreg = experiment(
    train_loader, val_loader, encoder_1, encoder_2, lamb_values, cov_values
)

# Visualization of Results
plt.figure()
plt.title("Effect of Lamb and Covariance Penalty on Sparsity of Latent Representations")
plt.xlabel("Penalty Value")
plt.ylabel("Mean Pairwise Distance (Sparsity)")
plt.plot(
    list(sparsity_results_barlow.keys()),
    list(sparsity_results_barlow.values()),
    label="Barlow Twins",
)
plt.plot(
    list(sparsity_results_vicreg.keys()),
    list(sparsity_results_vicreg.values()),
    label="VICReg",
)
plt.legend()
plt.show()
