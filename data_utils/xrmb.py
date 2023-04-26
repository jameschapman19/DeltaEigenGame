import numpy as np
from scipy.io import loadmat

PROJECT_DIR = "C:/Users/chapm/PycharmProjects/DeltaEigenGame/"
CLUSTER_DIR = "/home/jchapman/projects/DeltaEigengame/"


def xrmb():
    """
    Download, parse and process xrmb data
    Examples
    --------
    from ccagame import datasets

    train_view_1, train_view_2, test_view_1, test_view_2 = datasets.xrmb()

    Returns
    -------
    train_view_1, train_view_2, test_view_1, test_view_2
    """
    try:
        view_1 = loadmat(PROJECT_DIR + "data/XRMB/XRMBf2KALDI_window7_single1.mat")
        view_2 = loadmat(PROJECT_DIR + "data/XRMB/XRMBf2KALDI_window7_single2.mat")
    except:
        view_1 = loadmat(CLUSTER_DIR + "data/XRMB/XRMBf2KALDI_window7_single1.mat")
        view_2 = loadmat(CLUSTER_DIR + "data/XRMB/XRMBf2KALDI_window7_single2.mat")
    return view_1["X1"], view_2["X2"], view_1["XTe1"], view_2["XTe2"]


def xrmb_dataset():
    X, Y, X_te, Y_te = xrmb()
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    X_te = X_te.astype(np.float32)
    Y_te = Y_te.astype(np.float32)
    return X, Y, X_te, Y_te


def xrmb_true(cca=False):
    if cca:
        try:
            U = np.load(PROJECT_DIR + "data/XRMB/CCAU.npy")
            V = np.load(PROJECT_DIR + "data/XRMB/CCAV.npy")
        except:
            U = np.load(CLUSTER_DIR + "data/XRMB/CCAU.npy")
            V = np.load(CLUSTER_DIR + "data/XRMB/CCAV.npy")
    else:
        try:
            U = np.load(PROJECT_DIR + "data/XRMB/U.npy")
            V = np.load(PROJECT_DIR + "data/XRMB/V.npy")
        except:
            U = np.load(CLUSTER_DIR + "data/XRMB/U.npy")
            V = np.load(CLUSTER_DIR + "data/XRMB/V.npy")
    return U, V
