from typing import Iterable

import numpy as np
from cca_zoo.models import PLSEigenGame, PLSGHAGEP
from cca_zoo.models._iterative._base import _default_initializer

import wandb


class Tracker:
    def fit(self, views: Iterable[np.ndarray], y=None, val_views: Iterable[np.ndarray] = None, log_every=100,
            true=None):
        views = self._validate_inputs(views)
        self._check_params()
        train_dataloader, val_dataloader = self.get_dataloader(views)
        initializer = _default_initializer(
            views, self.initialization, self.random_state, self.latent_dims
        )
        self.weights = initializer.fit(views).weights
        self.weights = [weights.astype(np.float32) for weights in self.weights]
        self.track=[]
        for _ in range(self.epochs):
            for i, sample in enumerate(train_dataloader):
                self._update(sample["views"])
                if i % log_every == 0:
                    #self.weights_=[self.qr_weights(weights) for weights in self.weights]
                    tvc = -self.objective(views,u=self.weights)
                    wandb.log({"Train TVC": tvc})
                    if true is not None:
                        wandb.log({"Train PVC": tvc / true['train']})
                    if val_views is not None:
                        tvc= -self.objective(val_views,u=self.weights)
                        wandb.log({"Val TVC": tvc})
                        if true is not None:
                            wandb.log({"Val PVC": tvc / true['val']})
        return self

    def tvc(self, views):
        z = [view @ w for view, w in zip(views, self.weights)]
        m = np.cov(np.concatenate(z, axis=1).T)[self.latent_dims:, :self.latent_dims]
        return np.trace(m)

    @staticmethod
    def qr_weights(weights):
        Q, R = np.linalg.qr(weights)
        S = np.sign(np.sign(np.diag(R)) + 0.5)
        return Q @ np.diag(S)


class DeltaEigenGame(Tracker, PLSEigenGame):
    pass


class GHAGEP(Tracker, PLSGHAGEP):
    pass


class StochasticPower(Tracker, PLSGHAGEP):
    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u)
        # TODO


class SGHA(Tracker, PLSGHAGEP):
    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u)
        return Bw @ wAw - Aw


class GammaEigenGame(Tracker, PLSEigenGame):
    def __init__(self, **kwargs):
        self.gamma = kwargs.pop('gamma', 1)
        self.rho = kwargs.pop('rho', 1)
        self.BU = None
        super().__init__(**kwargs)

    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u)
        if self.BU is None:
            self.BU = Bw
        denominator = np.diag(u.T @ self.BU)
        denominator_y = np.sqrt(np.where(denominator > self.rho, denominator, self.rho))
        denominator_By = np.sqrt(np.clip(denominator, self.rho, np.inf))
        y = u / denominator_y
        By = self.BU / denominator_By
        Ay, _, _, _ = self._get_terms(views, y)
        rewards = Aw * np.diag(wBw) - Bw * np.diag(wAw)
        penalties = By @ np.triu(Ay.T @ u * np.diag(wBw), 1) - Bw * np.diag(np.tril(u.T @ By, -1) @ Ay.T @ u)
        self.BU = self.BU + self.learning_rate * (Bw - self.BU)
        grads = rewards - penalties
        return -grads
