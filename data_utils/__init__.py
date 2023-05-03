import numpy as np
from scipy.io import loadmat

PROJECT_DIR = "C:/Users/chapm/PycharmProjects/DeltaEigenGame/data/"
CLUSTER_DIR = "/cluster/project9/CCA_public/"


def load_cifar():
    train_x = loadmat(PROJECT_DIR + "CIFAR/train_x.mat")['cifar_train_x']
    train_y = loadmat(PROJECT_DIR + "CIFAR/train_y.mat")['cifar_train_y']
    test_x = loadmat(PROJECT_DIR + "CIFAR/test_x.mat")['test_x']
    test_y = loadmat(PROJECT_DIR + "CIFAR/test_y.mat")['test_y']
    #reshape to 2D
    train_x = train_x.reshape(train_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)
    train_y = train_y.reshape(train_y.shape[0], -1)
    test_y = test_y.reshape(test_y.shape[0], -1)
    return train_x, train_y, test_x, test_y

def load_mediamill():
    train_x = loadmat(PROJECT_DIR + "Mediamill/mediamill_trainX.mat")['trainX']
    train_y = loadmat(PROJECT_DIR + "Mediamill/mediamill_trainY.mat")['trainY']
    return train_x, train_y, None, None

def load_mnist():
    mnist = loadmat(PROJECT_DIR + "MNIST/mnist.mat")
    train = mnist['trainX']
    test = mnist['testX']
    train_x = train[:, :392]
    train_y = train[:, 392:]
    test_x = test[:, :392]
    test_y = test[:, 392:]
    return train_x, train_y, test_x, test_y