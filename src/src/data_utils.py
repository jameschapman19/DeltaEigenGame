import os
from typing import Tuple, Optional

import numpy as np
from scipy.io import loadmat

def set_project_dir() -> str:
    """
    Set the project directory based on the existing path.

    Returns:
        str: The project directory path.
    """
    if os.path.exists("C:/Users/chapm/PycharmProjects/GEP-GD/data/"):
        return "C:/Users/chapm/PycharmProjects/GEP-GD/data/"
    return "/cluster/project9/CCA_public/data/"


PROJECT_DIR = set_project_dir()


def load_cifar() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR dataset.

    Returns:
        Tuple of Numpy arrays: (train_x, train_y, test_x, test_y)
    """
    train_x = loadmat(os.path.join(PROJECT_DIR, "CIFAR", "train_x.mat"))["cifar_train_x"]
    train_y = loadmat(os.path.join(PROJECT_DIR, "CIFAR", "train_y.mat"))["cifar_train_y"]
    test_x = loadmat(os.path.join(PROJECT_DIR, "CIFAR", "test_x.mat"))["test_x"]
    test_y = loadmat(os.path.join(PROJECT_DIR, "CIFAR", "test_y.mat"))["test_y"]

    # Reshape data to 2D
    return train_x.reshape(train_x.shape[0], -1), train_y.reshape(train_y.shape[0], -1), \
        test_x.reshape(test_x.shape[0], -1), test_y.reshape(test_y.shape[0], -1)


def load_mediamill() -> Tuple[np.ndarray, np.ndarray, Optional[None], Optional[None]]:
    """
    Load Mediamill dataset.

    Returns:
        Tuple of Numpy arrays: (train_x, train_y, None, None)
    """
    train_x = loadmat(os.path.join(PROJECT_DIR, "Mediamill", "mediamill_trainX.mat"))["trainX"]
    train_y = loadmat(os.path.join(PROJECT_DIR, "Mediamill", "mediamill_trainY.mat"))["trainY"]

    return train_x, train_y, None, None


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset and add Gaussian noise.

    Returns:
        Tuple of Numpy arrays: (train_x, train_y, test_x, test_y)
    """
    mnist = loadmat(os.path.join(PROJECT_DIR, "MNIST", "mnist.mat"))
    train = mnist["trainX"]
    test = mnist["testX"]

    # Add Gaussian noise
    train = train + np.random.normal(0, 100, train.shape)
    test = test + np.random.normal(0, 100, test.shape)

    # Split data
    train_x, train_y = train[:, :392], train[:, 392:]
    test_x, test_y = test[:, :392], test[:, 392:]

    return train_x, train_y, test_x, test_y
