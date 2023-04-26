import time

import numpy as np
from cca_zoo.models import CCA

from cca import DeltaEigenGame, GammaEigenGame, GHAGEP

np.random.seed(1)

X = np.random.rand(100, 10)
Y = np.random.rand(100, 10)

epochs = 100
lr = 1e-1
batch_size = 100
momentum = 0.5

gt = CCA(latent_dims=5).fit((X, Y)).score((X, Y)).sum()

# BLS
t = time.time()
delta_bls = DeltaEigenGame(latent_dims=5, epochs=epochs, learning_rate=1, line_search=True, momentum=0.1,
                           random_state=0, batch_size=batch_size).fit((X, Y))
print(time.time() - t)

# nesterov
t = time.time()
delta_acc = GHAGEP(latent_dims=5, epochs=epochs, learning_rate=lr, momentum=momentum, random_state=0,
                   batch_size=batch_size).fit((X, Y))
print(time.time() - t)

t = time.time()
gamma_acc = GammaEigenGame(latent_dims=5, epochs=epochs, learning_rate=lr, momentum=momentum, random_state=0,
                           batch_size=batch_size).fit((X, Y))
print(time.time() - t)

t = time.time()
ghagep_acc = GHAGEP(latent_dims=5, epochs=epochs, learning_rate=lr, momentum=momentum, random_state=0,
                    batch_size=batch_size).fit((X, Y))
print(time.time() - t)

# sgd
t = time.time()
ghagep = GHAGEP(latent_dims=5, epochs=epochs, learning_rate=lr, random_state=0, batch_size=batch_size).fit((X, Y))
print(time.time() - t)

t = time.time()
gamma = GammaEigenGame(latent_dims=5, epochs=epochs, learning_rate=lr, random_state=0, batch_size=batch_size).fit(
    (X, Y))
print(time.time() - t)

t = time.time()
delta = DeltaEigenGame(latent_dims=5, epochs=epochs, learning_rate=lr, random_state=0, batch_size=batch_size).fit(
    (X, Y))
print(time.time() - t)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(ghagep.track, label='ghagep')
plt.plot(ghagep_acc.track, label='ghagep acc')
plt.plot(gamma.track, label='gamma')
plt.plot(gamma_acc.track, label='gamma acc')
plt.plot(delta_acc.track, label='delta acc')
plt.plot(delta.track, label='delta')
plt.plot(delta_bls.track, label='delta bls')
plt.legend()
plt.show()

print()
