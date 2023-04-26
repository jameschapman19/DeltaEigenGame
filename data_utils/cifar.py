import numpy as np
from scipy.io import loadmat

PROJECT_DIR = "C:/Users/chapm/PycharmProjects/DeltaEigenGame/"
CLUSTER_DIR = "/home/jchapman/projects/DeltaEigengame/"


def cifar():
    """
    Download, parse and process MediaMill data
    Examples
    --------
    from ccagame import datasets

    train_view_1, train_view_2, test_view_1, test_view_2 = datasets.cifar()

    Returns
    -------
    train_view_1, train_view_2, test_view_1, test_view_2
    """
    try:
        x_tr, y_tr, x_te, y_te = load_cifar(PROJECT_DIR + "data/CIFAR/")
    except:
        x_tr, y_tr, x_te, y_te = load_cifar(CLUSTER_DIR + "data/CIFAR/")
    return x_tr, y_tr, x_te, y_te


def load_cifar(project_dir):
    x_tr = loadmat(project_dir + "train_x.mat")["cifar_train_x"]
    x_te = loadmat(project_dir + "test_x.mat")["test_x"]
    y_tr = loadmat(project_dir + "train_y.mat")["cifar_train_y"]
    y_te = loadmat(project_dir + "test_y.mat")["test_y"]
    return (
        x_tr.reshape((50000, -1)),
        y_tr.reshape((50000, -1)),
        x_te.reshape((10000, -1)),
        y_te.reshape((10000, -1)),
    )


def cifar_dataset():
    X, Y, X_te, Y_te = cifar()
    X = X.astype(np.float32) / 255
    Y = Y.astype(np.float32) / 255
    X_te = X_te.astype(np.float32) / 255
    Y_te = Y_te.astype(np.float32) / 255
    return X, Y, X_te, Y_te


def cifar_true(cca=False):
    if cca:
        try:
            U = np.load(PROJECT_DIR + "data/CIFAR/CCAU.npy")
            V = np.load(PROJECT_DIR + "data/CIFAR/CCAV.npy")
        except:
            U = np.load(CLUSTER_DIR + "data/CIFAR/CCAU.npy")
            V = np.load(CLUSTER_DIR + "data/CIFAR/CCAV.npy")
    else:
        try:
            U = np.load(PROJECT_DIR + "data/CIFAR/U.npy")
            V = np.load(PROJECT_DIR + "data/CIFAR/V.npy")
        except:
            U = np.load(CLUSTER_DIR + "data/CIFAR/U.npy")
            V = np.load(CLUSTER_DIR + "data/CIFAR/V.npy")
    return U, V
