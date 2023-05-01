from typing import Iterable

import numpy as np
import wandb
from cca_zoo.models import CCAEigenGame, CCAGHAGEP, MCCA
from cca_zoo.models._iterative._base import _default_initializer


class Tracker:
    def fit(self, views: Iterable[np.ndarray], y=None, val_views: Iterable[np.ndarray] = None, log_every=1,
            true=None):
        views = self._validate_inputs(views)
        self._check_params()
        train_dataloader, val_dataloader = self.get_dataloader(views)
        initializer = _default_initializer(
            views, self.initialization, self.random_state, self.latent_dims
        )
        self.weights = initializer.fit(views).weights
        self.weights = [weights.astype(np.float32) for weights in self.weights]
        i = 0
        for e in range(self.epochs):
            for s, sample in enumerate(train_dataloader):
                self._update(sample["views"])
                i += self.batch_size
                if i % log_every == 0:
                    tcc = self.tcc(views)
                    wandb.log({"Train TCC": tcc}, step=i)
                    if true is not None:
                        pcc = tcc / true['train']
                        wandb.log({"Train PCC": pcc}, step=i)
                    if val_views is not None:
                        tcc= self.tcc(val_views)
                        wandb.log({"Val TCC": tcc}, step=i)
                        if true is not None:
                            pcc=tcc / true['val']
                            wandb.log({"Val PCC": pcc}, step=i)
        return self

    def tcc(self, views):
        z = self.transform(views)
        tcc = MCCA(latent_dims=self.latent_dims, scale=False, centre=False).fit(z).score(z)
        return tcc.sum()


class DeltaEigenGame(Tracker, CCAEigenGame):
    pass

class DeltaEigenGameSO(Tracker, CCAEigenGame):
    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u)
        grads = 2 * Aw - (Aw @ wBw * np.sign(np.diag(wAw)) + Bw @ wAw)
        return -grads

class GHAGEP(Tracker, CCAGHAGEP):
    pass


class SGHA(Tracker, CCAGHAGEP):
    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u)
        return Bw @ wAw - Aw


class GammaEigenGame(Tracker, CCAEigenGame):
    def __init__(self, **kwargs):
        self.gamma = kwargs.pop('gamma', 1e-1)
        self.Bu = None
        super().__init__(**kwargs)
        self.rho=1e-10

    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u)
        check=np.diag(wAw)/np.diag(wBw)
        if self.Bu is None:
            self.Bu = u
        denominator = np.diag(u.T @ self.Bu)
        denominator = np.where(denominator > self.rho, np.sqrt(denominator), self.rho)
        y = u / denominator
        By = self.Bu / denominator
        Ay, _, _, _ = self._get_terms(views, y)
        rewards = Aw * np.diag(wBw) - Bw * np.diag(wAw)
        penalties = By @ np.triu(Ay.T @ u * np.diag(wBw), 1) - Bw * np.diag(np.tril(u.T @ By, -1) @ Ay.T @ u)
        self.Bu = self.Bu + self.gamma * (Bw - self.Bu)
        grads = rewards - penalties
        return -grads

    def _gradient_step(self, weights, velocity):
        weights= weights + velocity
        return weights/np.linalg.norm(weights, axis=0, keepdims=True)