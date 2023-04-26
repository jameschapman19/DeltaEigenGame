from typing import Iterable

import numpy as np
import wandb
from cca_zoo.models import CCAEigenGame, CCAGHAGEP, MCCA
from cca_zoo.models._iterative._base import _default_initializer


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
        for _ in range(self.epochs):
            for i, sample in enumerate(train_dataloader):
                self._update(sample["views"])
                if i % log_every == 0:
                    tcc = self.tcc(views)
                    wandb.log({"Train TCC": tcc})
                    if true is not None:
                        pcc = tcc / true['train']
                        wandb.log({"Train PCC": pcc})
                    if val_views is not None:
                        tcc= self.tcc(val_views)
                        wandb.log({"Val TCC": tcc})
                        if true is not None:
                            pcc=tcc / true['val']
                            wandb.log({"Val PCC": pcc})
        return self

    def tcc(self, views):
        z = self.transform(views)
        tcc = MCCA(latent_dims=self.latent_dims).fit(z).score(z)
        return tcc.sum()


class DeltaEigenGame(Tracker, CCAEigenGame):
    pass


class GHAGEP(Tracker, CCAGHAGEP):
    pass


class SGHA(Tracker, CCAGHAGEP):
    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u)
        return Bw @ wAw - Aw


class GammaEigenGame(Tracker, CCAEigenGame):
    def __init__(self, gamma=1e-1, rho=1e-3, **kwargs):
        self.gamma = gamma
        self.rho = rho
        self.BU = None
        super().__init__(**kwargs)

    def grads(self, views, u=None):
        u /= np.linalg.norm(u, axis=0, keepdims=True)
        self.weights /= np.linalg.norm(self.weights, axis=0, keepdims=True)
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
        self.BU = self.BU + self.gamma * (Bw - self.BU)
        grads = rewards - penalties
        return -grads
