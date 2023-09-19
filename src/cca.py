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
        for i in range(len(self.torch_weights)):
            self.torch_weights[i].data -= self.learning_rate * self.torch_weights[i].grad
            self.torch_weights[i].data /= torch.norm(self.torch_weights[i], dim=0)

    def loss(self, views, independent_views=None, **kwargs):
        # measure the correlation between the views
        z = self(views)
        return {'correlation': torch.corrcoef(torch.hstack(z).T)[0, 1]}

    def _update_grads(self, views, independent_views=None):
        A, B = self._AB(views)
        u = torch.vstack([w.data for w in self.torch_weights])
        Aw = A @ u
        Bw = B @ u
        wAw = torch.diag(u.T @ Aw)
        wBw = torch.diag(u.T @ Bw)
        rewards = Aw * torch.diag(wBw) - Bw * torch.diag(wAw)
        for i in range(len(self.torch_weights)):
            self.torch_weights[i].grad =- rewards[:views[i].shape[1]]

    def _AB(self, views):
        all_views = torch.hstack(views)
        A = torch.cov(all_views.T)
        B = torch.block_diag(*[torch.cov(view.T) for view in views])
        A = A - B
        return A, B

    def _get_Aw(self, views):
        n = views[0].shape[0]
        Aw = [views[0].T @ views[1] @ self.torch_weights[1] / n, views[1].T @ views[0] @ self.torch_weights[0] / n]
        Bw = [views[0].T @ views[0] @ self.torch_weights[0] / n, views[1].T @ views[1] @ self.torch_weights[1] / n]
        return Aw, Bw


if __name__ == '__main__':
    import numpy as np
    from cca_zoo.linear import CCA

    views = [np.random.rand(100, 12), np.random.rand(100, 10)]
    cca = CCA().fit(views).score(views)

    # from cca_zoo.linear import CCA_EY
    # model = CCA_EY(epochs=1000, learning_rate=1e-1, optimizer_kwargs={'optimizer': 'SGD', 'nesterov':False}).fit(views)
    # model_score = model.score(views)
    # print()
    model = GammaEigenGame(epochs=1000, learning_rate=1, optimizer_kwargs={'optimizer': 'SGD', 'nesterov': False}).fit(
        views)
    model_score = model.score(views)
    print()
