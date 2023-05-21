# This script defines different classes for CCA models that inherit from Tracker and CCAEigenGame or CCAGHAGEP
# The classes implement different gradient update rules based on different variants of EigenGame algorithm

from typing import Iterable

import numpy as np
import wandb
from cca_zoo.models import CCAEigenGame, CCAGHAGEP, MCCA
from cca_zoo.models._iterative._base import _default_initializer


class Tracker:
    # This class is a base class for all the CCA models that use Tracker
    # It implements the fit and tcc methods that are common to all the models

    def fit(
            self,
            views: Iterable[np.ndarray],
            y=None,
            val_views: Iterable[np.ndarray] = None,
            log_every=1,
            true=None,
    ):
        # This method trains the model on the given views and logs the results using wandb

        views = self._validate_inputs(views)  # validate the input views
        self._check_params()  # check the model parameters
        train_dataloader, val_dataloader = self.get_dataloader(
            views
        )  # get the data loaders for training and validation
        initializer = _default_initializer(
            views, self.initialization, self.random_state, self.latent_dims
        )  # get the initializer for the weights
        self.weights = initializer.fit(views).weights  # initialize the weights
        self.weights = [
            weights.astype(np.float32) / 4 for weights in self.weights
        ]  # scale the weights by 1/4
        i = 0  # initialize the iteration counter
        for e in range(self.epochs):  # loop over the epochs
            for s, sample in enumerate(train_dataloader):  # loop over the batches
                self._update(sample["views"])  # update the weights using the gradients
                i += 1  # increment the iteration counter
                if i % log_every == 0:  # log every log_every iterations
                    tcc = self.tcc(
                        views
                    )  # compute the total correlation captured (TCC) on the training views
                    wandb.log(
                        {"Train TCC": tcc}, step=i * self.batch_size
                    )  # log the train TCC using wandb
                    if true is not None:
                        pcc = (
                                tcc / true["train"]
                        )  # compute the percentage correlation captured (PCC) on the training views using true values
                        wandb.log(
                            {"Train PCC": pcc}, step=i * self.batch_size
                        )  # log the train PCC using wandb
                    if val_views is not None:
                        tcc = self.tcc(
                            val_views
                        )  # compute the TCC on the validation views
                        wandb.log(
                            {"Val TCC": tcc}, step=i * self.batch_size
                        )  # log the val TCC using wandb
                        if true is not None:
                            pcc = (
                                    tcc / true["val"]
                            )  # compute the PCC on the validation views using true values
                            wandb.log(
                                {"Val PCC": pcc}, step=i * self.batch_size
                            )  # log the val PCC using wandb
        return self

    def tcc(self, views):
        # This method computes the TCC of the model on the given views

        z = self.transform(views)  # transform the views using the weights
        tcc = (
            MCCA(latent_dims=self.latent_dims).fit(z).score(z)
        )  # compute the TCC using MCCA model from cca_zoo package
        return tcc.sum()  # return the sum of TCC


class GEPGH(Tracker, CCAEigenGame):
    # This class implements GEPGH model which uses a gradient update rule based on generalized eigenvalue problem (GEP)
    # and a history term (H) to avoid oscillations

    def grads(self, views, u=None):
        # This method computes the gradients of GEPGH model

        if self.previous_views is None:
            self.previous_views = (
                views  # initialize previous_views with current views if None
            )

        projections_ = self.projections(
            self.previous_views, u
        )  # project previous_views using u (default to weights)

        Bw = self._Bw(
            self.previous_views, projections_, u
        )  # compute Bw term using previous_views and projections_

        projections = self.projections(views, u)  # project current views using u

        Aw = self._Aw(
            views, projections
        )  # compute Aw term using current views and projections

        self.previous_views = views  # update previous_views with current views

        wBw = u.T @ Bw  # compute wBw term

        wAw = u.T @ Aw  # compute wAw term

        grads = 2 * Aw - (
                Aw @ wBw + Bw @ wAw
        )  # compute gradients using GEP formula with history term

        return -grads


class GEPGHBiased(Tracker, CCAEigenGame):
    # This class implements GEPGHBiased model which uses a gradient update rule based on generalized eigenvalue problem (GEP)
    # and a history term (H) to avoid oscillations but without removing bias terms

    def grads(self, views, u=None):
        projections = self.projections(views, u)
        Bw = self._Bw(views, projections, u)
        Aw = self._Aw(views, projections)
        self.previous_views = views
        wBw = u.T @ Bw
        wAw = u.T @ Aw
        grads = 2 * Aw - (Aw @ wBw + Bw @ wAw)
        return -grads


class GEPEY(Tracker, CCAEigenGame):
    def grads(self, views, u=None):
        if self.previous_views is None:
            self.previous_views = views
        projections = self.projections(views, u)
        Bw = self._Bw(views, projections, u)
        Aw = self._Aw(views, projections)
        projections_ = self.projections(self.previous_views, u)
        Bw_ = self._Bw(self.previous_views, projections_, u)
        self.previous_views = views
        wBw = u.T @ Bw
        wBw_ = u.T @ Bw_
        grads = 2 * Aw - (Bw_ @ wBw + Bw @ wBw_)
        return -grads


class GEPEYBiased(Tracker, CCAEigenGame):
    def grads(self, views, u=None):
        projections = self.projections(views, u)
        Bw = self._Bw(views, projections, u)
        Aw = self._Aw(views, projections)
        wBw = u.T @ Bw
        grads = 2 * Aw - (Bw @ wBw + Bw @ wBw)
        return -grads


class SVD(Tracker, CCAEigenGame):
    def grads(self, views, u=None):
        if self.previous_views is None:
            self.previous_views = views
        projections = self.projections(self.previous_views, u)
        covs = [np.cov(projection.T) for projection in projections]
        projections = self.projections(views, u)
        Bws = [view.T @ projection for view, projection in zip(views, projections)]
        Bws = [Bws[0] @ covs[1], Bws[1] @ covs[0]]
        Aw = [views[0].T @ projections[1], views[1].T @ projections[0]]
        self.previous_views = views
        grads = 2 * np.vstack(Aw) - 2 * np.vstack(Bws)
        return -grads


class SGHA(Tracker, CCAGHAGEP):
    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u, unbiased=True)
        grads = Aw - Bw @ wAw
        return -grads


class GammaEigenGame(Tracker, CCAEigenGame):
    def __init__(self, **kwargs):
        self.gamma = kwargs.pop("gamma", 1e-1)
        self.Bu = None
        super().__init__(**kwargs)
        self.rho = 1e-10

    def grads(self, views, u=None):
        u /= np.linalg.norm(u, axis=0, keepdims=True)
        Aw, Bw, wAw, wBw = self._get_terms(views, u, unbiased=False)
        check = np.diag(wAw) / np.diag(wBw)
        if self.Bu is None:
            self.Bu = u
        denominator = np.diag(u.T @ self.Bu)
        denominator = np.where(denominator > self.rho, np.sqrt(denominator), self.rho)
        y = u / denominator
        By = self.Bu / denominator
        Ay, _, _, _ = self._get_terms(views, y, unbiased=False)
        rewards = Aw * np.diag(wBw) - Bw * np.diag(wAw)
        penalties = By @ np.triu(Ay.T @ u * np.diag(wBw), 1) - Bw * np.diag(
            np.tril(u.T @ By, -1) @ Ay.T @ u
        )
        self.Bu = self.Bu + self.gamma * (Bw - self.Bu)
        grads = rewards - penalties
        return -grads

    def _gradient_step(self, weights, velocity):
        weights = weights + velocity
        return weights / np.linalg.norm(weights, axis=0, keepdims=True)
