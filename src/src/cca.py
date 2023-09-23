import torch
from cca_zoo.linear import CCA_EY, CCA_GHA

class CCA_GHAObj(CCA_GHA):
    def loss(self, views, independent_views=None, **kwargs):
        # Encoding the views with the forward method
        z = self(views)
        # Getting A and B matrices from z
        A, B = self.get_AB(z)
        rewards = torch.trace(2 * A)
        if independent_views is None:
            # Hebbian
            penalties = torch.trace(A @ B)
            # penalties = torch.trace(A @ B)
        else:
            # Encoding another set of views with the forward method
            independent_z = self(independent_views)
            # Getting A' and B' matrices from independent_z
            independent_A, independent_B = self.get_AB(independent_z)
            # Hebbian
            penalties = torch.trace(independent_A @ B)
            # penalties = torch.trace(A @ independent_B)
        return {
            "loss": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

class SGHA(CCA_GHA):
    def loss(self, views, **kwargs):
        # Encoding the views with the forward method
        z = self(views)
        # Getting A and B matrices from z
        A, B = self.get_AB(z)
        rewards = torch.trace(2 * A)
        penalties = torch.trace(A.detach() @ B)
        return {
            "loss": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }



class GammaEigenGame(CCA_EY):
    """
    Implement the Stochastic γ-EigenGame algorithm based on the following paper:

    Gemp, Ian, Charlie Chen, and Brian McWilliams.
    "The generalized eigenvalue problem as a Nash equilibrium."
    arXiv preprint arXiv:2206.04993 (2022).

    Attributes:
        gamma (float): Step size for Bv updates.
        Bv (tensor): Initialized as None and will be updated during training.
        rho (float): Small scalar lower bounding σmin(B).
        manual_optimization (bool): Flag to indicate manual optimization.
    """

    gamma = 0.1
    Bv = None
    rho = 1e-10
    manual_optimization = True

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (dict): The current batch of data.
            batch_idx (int): The index of the current batch.
        """
        if self.batch_size is None:
            batch = self.batch
        self._update_grads(batch["views"])
        self.v = self.v + self.learning_rate * self.v.grad
        self.v /= torch.norm(self.v, dim=0)
        for w, v in zip(self.torch_weights, torch.split(self.v, self.n_features_)):
            w.data = v

    def validation_step(self, batch, batch_idx):
        pass

    def on_train_start(self) -> None:
        """Initialize v from torch_weights at the start of training."""
        self.v = torch.vstack([w.data for w in self.torch_weights])
        self.v /= torch.norm(self.v, dim=0)

    def _update_grads(self, views):
        """
        Update gradients based on the current views.

        Args:
            views (list): List of current views.
        """
        A, B = self._AB(views)
        Av = A @ self.v
        Bv = B @ self.v
        check = torch.diag(self.v.T @ Av) / torch.diag(self.v.T @ Bv)
        if self.Bv is None:
            self.Bv = Bv
        denominator = torch.diag(self.v.T @ self.Bv)
        denominator = torch.where(denominator > self.rho, torch.sqrt(denominator), self.rho)
        y = self.v / denominator[None, :]
        By = self.Bv / denominator
        Ay = A @ y
        rewards = Av * torch.diag(self.v.T @ Bv) - Bv * torch.diag(self.v.T @ Av)
        penalties = By @ torch.triu(Ay.T @ self.v * torch.diag(self.v.T @ Bv), 1) - Bv * torch.diag(
            torch.tril(self.v.T @ By, -1) @ Ay.T @ self.v
        )
        self.Bv += self.gamma * (Bv - self.Bv)
        grads = rewards - penalties
        self.v.grad = grads

    def _AB(self, views):
        """
        Compute A and B matrices.

        Args:
            views (list): List of current views.

        Returns:
            tuple: A tuple containing A and B matrices.
        """
        all_views = torch.hstack(views)
        A = torch.cov(all_views.T)
        B = torch.block_diag(*[torch.cov(view.T) for view in views])
        A -= B
        return A, B

