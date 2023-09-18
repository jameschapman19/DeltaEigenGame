from cca_zoo.linear import CCA_EY
import torch


class GammaEigenGame(CCA_EY):
    gamma = 1e-1
    Bu = None
    rho = 1e-10
    manual_optimization = True

    def training_step(self, batch, batch_idx):
        if self.batch_size is None:
            batch = self.batch
        loss = self.loss(batch["views"], batch.get("independent_views", None))
        # Logging the loss components
        for k, v in loss.items():
            self.log(k, v, prog_bar=True)
        self._update_grads(batch["views"], batch.get("independent_views", None))
        self.optimizer_step(self.optimizers())
        return loss

    def grads(self, views, u=None):
        # stack self.torch_weights and take the norm
        # combined weights
        u = torch.stack(self.torch_weights)
        norm = torch.linalg.norm(u, dim=0, keepdim=True)
        weights = [weights / norm for weights in self.torch_weights]

        z = self(views)
        wAw, wBw = self.get_AB(self, z)
        Aw = torch.vstack(views[0].T @ views[1] @ weights[0], views[1].T @ views[0] @ weights[1])
        Bw = torch.vstack(views[0].T @ views[0] @ weights[0], views[1].T @ views[1] @ weights[1])

        if self.Bu is None:
            self.Bu = u

        denominator = torch.diag(torch.mm(u.T, self.Bu))
        denominator = torch.where(denominator > self.rho, torch.sqrt(denominator), torch.tensor([self.rho]))
        y = u / denominator
        By = self.Bu / denominator
        Ay, _, _, _ = self._get_terms(views, y, unbiased=False)

        rewards = Aw * torch.diag(wBw) - Bw * torch.diag(wAw)
        penalties = By.mm(
            torch.triu(torch.mm(Ay.T, u) * torch.diag(wBw), diagonal=1)
        ) - Bw * torch.diag(
            torch.tril(u.T.mm(By), diagonal=-1).mm(Ay.T.mm(u))
        )

        self.Bu = self.Bu + self.gamma * (Bw - self.Bu)

        grads = rewards - penalties

        self.torch_weights[0].grad = grads[:views[0].shape[1]]
        self.torch_weights[1].grad = grads[views[0].shape[1]:]


if __name__ == '__main__':
    import numpy as np

    views = [np.random.rand(100, 10), np.random.rand(100, 10)]

    model = GammaEigenGame()
    model.fit(views)
