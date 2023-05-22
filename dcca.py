# Importing the required modules
import torch
from cca_zoo.deepmodels import DCCA_EigenGame
from pytorch_lightning import Callback, LightningModule, Trainer


# Defining a custom callback to log the correlation metrics
class CorrelationCallback(Callback):
    def on_validation_epoch_end(
            self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # Computing the correlations for each view pair
        corrs = pl_module.score(trainer.val_dataloaders[0])
        # Logging the sum of correlations and squared correlations
        pl_module.log("val/corr", corrs.sum())
        pl_module.log("val/corr_squared", (corrs ** 2).sum())


# Defining a subclass of DCCA_EigenGame with a custom loss function
class DCCA_EY(DCCA_EigenGame):
    def __init__(self, latent_dims: int, encoders=None, r: float = 0, **kwargs):
        super().__init__(latent_dims=latent_dims, encoders=encoders, **kwargs)
        # Initializing two queues to store batches for computing the loss
        self.batch_queue = []
        self.val_batch_queue = []

    def forward(self, views, **kwargs):
        # Encoding each view with its corresponding encoder
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(views[i]))
        return z

    def training_step(self, batch, batch_idx):
        # Checking if the queue has at least one batch
        if len(self.batch_queue) < 1:
            # Adding the current batch to the queue
            self.batch_queue.append(batch)
            # Returning a zero loss
            loss = {
                "objective": torch.tensor(0, requires_grad=True, dtype=torch.float32),
            }
        else:
            # Computing the loss with the current batch and the oldest batch in the queue
            loss = self.loss(batch["views"], self.batch_queue[0]["views"])
            # Adding the current batch to the queue and removing the oldest batch
            self.batch_queue.append(batch)
            self.batch_queue.pop(0)
        # Logging the loss components
        for k, v in loss.items():
            self.log("train/" + k, v, prog_bar=False)
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        # Similar logic as training_step but for validation data
        if len(self.val_batch_queue) < 1:
            self.val_batch_queue.append(batch)
            loss = {
                "objective": torch.tensor(0, requires_grad=True, dtype=torch.float32),
            }
        else:
            loss = self.loss(batch["views"], self.val_batch_queue[0]["views"])
            self.val_batch_queue.append(batch)
            self.val_batch_queue.pop(0)
        for k, v in loss.items():
            self.log("val/" + k, v)
        return loss

    def test_step(self, batch, batch_idx):
        # Computing the loss with the test batch only
        loss = self.loss(batch["views"])
        # Logging the loss components
        for k, v in loss.items():
            self.log("test/" + k, v)
        return loss["objective"]

    def get_AB(self, z):
        # Getting the dimensions of the encoded views
        N, D = z[0].size()
        # Computing the covariance matrix of the concatenated views
        C = torch.cov(torch.hstack(z).T)
        # Extracting the cross-covariance and auto-covariance matrices
        A = C[:D, D:] + C[D:, :D]
        B = C[:D, :D] + C[D:, D:]
        return A, B

    def loss(self, views, views2=None, **kwargs):
        # Encoding the views with the forward method
        z = self(views)
        # Getting A and B matrices from z
        A, B = self.get_AB(z)

        if views2 is None:
            # Computing rewards and penalties using A and B only
            rewards = torch.trace(2 * A)
            penalties = torch.trace(B @ B)

        else:
            # Encoding another set of views with the forward method
            z2 = self(views2)
            # Getting A' and B' matrices from z2
            A_, B_ = self.get_AB(z2)
            # Computing rewards and penalties using A and B'
            rewards = torch.trace(2 * A)
            penalties = torch.trace(B @ B_)

        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

    def configure_callbacks(self):
        # Returning an instance of CorrelationCallback as a callback
        return [CorrelationCallback()]


# Defining another subclass of DCCA_EY with a different loss function
class DCCA_SVD(DCCA_EY):
    def loss(self, views, views2=None, **kwargs):
        z = self(views)
        C = torch.cov(torch.hstack(z).T)

        Cxy = C[: self.latent_dims, self.latent_dims:]
        Cxx = C[: self.latent_dims, : self.latent_dims]

        if views2 is None:
            Cyy = C[self.latent_dims:, self.latent_dims:]
        else:
            z2 = self(views2)
            Cyy = torch.cov(torch.hstack(z2).T)[self.latent_dims:, self.latent_dims:]

        rewards = torch.trace(2 * Cxy)
        penalties = torch.trace(Cxx @ Cyy)

        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

class DCCA_BarlowTwins(DCCA_EY):
    def loss(self, views, views2=None, **kwargs):
        z = self(views)
        N, D = z[0].size()
        bn = torch.nn.BatchNorm1d(D, affine=False).to(z[0].device)
        z = [bn(zi) for zi in z]

        corr = torch.einsum("bi, bj -> ij", z[0], z[1]) / N

        diag = torch.eye(D, device=corr.device)
        cdif = (corr - diag).pow(2)
        cdif[~diag.bool()] *= 5e-3
        loss = 0.025 * cdif.sum()
        return {
            "objective": loss,
        }