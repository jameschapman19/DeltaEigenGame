"""
Get Baseline Scores for PLS and CCA on each dataset and save them to memory
"""
import os

import numpy as np
from cca_zoo.models import CCA, PLS

from data_utils import load_cifar, load_mediamill, load_mnist

os.makedirs('./results', exist_ok=True)

for data in ['cifar','mnist','mediamill']:
    if data == 'cifar':
        X, Y, X_test, Y_test = load_cifar()
    elif data == 'mnist':
        X, Y, X_test, Y_test = load_mnist()
    elif data == 'mediamill':
        X, Y, X_test, Y_test = load_mediamill()
    else:
        raise ValueError('Dataset not found')

    from sklearn.preprocessing import StandardScaler

    components = 5

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)
    Y = y_scaler.fit_transform(Y)
    if X_test is not None:
        X_test = x_scaler.transform(X_test)
        Y_test = y_scaler.transform(Y_test)

    cca = CCA(latent_dims=components, scale=True, centre=True).fit((X, Y))
    cca_score_train = cca.score((X, Y))
    np.save(f'./results/{data}_cca_score_train.npy', cca_score_train)
    if X_test is not None:
        cca_score_test = cca.score((X_test, Y_test))
        np.save(f'./results/{data}_cca_score_test.npy', cca_score_test)

    pls = PLS(latent_dims=components, scale=False, centre=False).fit((X, Y))
    z = pls.transform((X, Y))
    pls_score_train = np.diag(np.cov(z[0].T, z[1].T)[:components, components:])
    np.save(f'./results/{data}_pls_score_train.npy', pls_score_train)
    if X_test is not None:
        z = pls.transform((X_test, Y_test))
        pls_score_test = np.diag(np.cov(z[0].T, z[1].T)[:components, components:])
        np.save(f'./results/{data}_pls_score_test.npy', pls_score_test)
