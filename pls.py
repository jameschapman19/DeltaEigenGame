from typing import Iterable

import numpy as np
import scipy.linalg
from cca_zoo.models import PLSEigenGame, PLSGHAGEP
from cca_zoo.models._iterative._base import _default_initializer

import wandb


class Tracker:
    def fit(
        self,
        views: Iterable[np.ndarray],
        y=None,
        val_views: Iterable[np.ndarray] = None,
        log_every=1,
        true=None,
    ):
        views = self._validate_inputs(views)
        self._check_params()
        train_dataloader, val_dataloader = self.get_dataloader(views)
        initializer = _default_initializer(
            views, self.initialization, self.random_state, self.latent_dims
        )
        self.weights = initializer.fit(views).weights
        self.weights = [weights.astype(np.float32) for weights in self.weights]
        self.weights = [weight/(2*np.linalg.norm(weight)) for weight in self.weights]
        i = 0
        for e in range(self.epochs):
            for s, sample in enumerate(train_dataloader):
                self._update(sample["views"])
                i += 1
                if i % log_every == 0:
                    u = [self.qr_weights(w) for w in self.weights]
                    tvc = self.tvc(views, u=u)
                    wandb.log({"Train TVC": tvc}, step=i * self.batch_size)
                    if true is not None:
                        pvc = tvc / true["train"]
                        wandb.log({"Train PVC": pvc}, step=i * self.batch_size)
                    if val_views is not None:
                        tvc = self.tvc(val_views, u=u)
                        wandb.log({"Val TVC": tvc}, step=i * self.batch_size)
                        if true is not None:
                            pvc = tvc / true["val"]
                            wandb.log({"Val PVC": pvc}, step=i * self.batch_size)
        return self

    def tvc(self, views, u):
        z = [view @ w for view, w in zip(views, u)]
        s = scipy.linalg.svdvals(
            np.cov(*z, rowvar=False)[0 : self.latent_dims, self.latent_dims :]
        )
        return s.sum()

    @staticmethod
    def qr_weights(weights):
        Q, R = np.linalg.qr(weights)
        S = np.sign(np.sign(np.diag(R)) + 0.5)
        return Q @ np.diag(S)


class DeltaEigenGame(Tracker, PLSEigenGame):
    pass


class Subspace(Tracker, PLSEigenGame):
    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u, unbiased=True)
        grads = 2 * Aw - (Aw @ wBw + Bw @ wAw)
        return -grads


class GHAGEP(Tracker, PLSGHAGEP):
    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u, unbiased=True)
        grads = Aw - Bw @ np.triu(wAw)
        return -grads


class StochasticPower(Tracker, PLSGHAGEP):
    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u)
        return -2 * Aw

    def _gradient_step(self, weights, velocity):
        weights = weights + velocity
        # QR decomposition to orthogonalize weights
        Q, R = np.linalg.qr(weights)
        # SVD decomposition to get the sign of the weights
        U, _, V = np.linalg.svd(R)
        S = np.sign(np.sign(np.diag(U)) + 0.5)
        weights = Q @ np.diag(S)
        return weights

class Eckhart(Tracker, PLSEigenGame):
    def grads(self, views, u=None):
        if self.previous_views is None:
            self.previous_views = views
        projections = self.projections(self.previous_views, u)
        Bw_ = self._Bw(self.previous_views, projections, u)
        projections = self.projections(views, u)
        Bw = self._Bw(views, projections, u)
        Aw = self._Aw(views, projections)
        self.previous_views = views
        wBw = u.T @ Bw
        wBw_ = u.T @ Bw_
        grads = 2 * Aw - (Bw_ @ wBw + Bw @ wBw_)
        return -grads

class SGHA(Tracker, PLSGHAGEP):
    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u, unbiased=True)
        return -2*Aw + Bw @ wAw


class GammaEigenGame(Tracker, PLSEigenGame):
    def __init__(self, **kwargs):
        self.gamma = kwargs.pop("gamma", 1e-1)
        self.BU = None
        super().__init__(**kwargs)
        self.rho = 1e-10

    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u, unbiased=True)
        if self.BU is None:
            self.BU = Bw
        denominator = np.diag(u.T @ self.BU)
        denominator = np.where(denominator > self.rho, np.sqrt(denominator), self.rho)
        y = u / denominator
        By = self.BU / denominator
        Ay, _, _, _ = self._get_terms(views, y)
        rewards = Aw * np.diag(wBw) - Bw * np.diag(wAw)
        penalties = By @ np.triu(Ay.T @ u * np.diag(wBw), 1) - Bw * np.diag(
            np.tril(u.T @ By, -1) @ Ay.T @ u
        )
        self.BU = self.BU + self.gamma * (Bw - self.BU)
        grads = rewards - penalties
        return -grads

    def _gradient_step(self, weights, velocity):
        weights = weights + velocity
        return weights / np.linalg.norm(weights, axis=0, keepdims=True)
