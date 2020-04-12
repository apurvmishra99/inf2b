#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io


def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def task2_sNeuron(W, X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    #  W : (D+1)-by-1 vector of weights (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    bias = W[0]
    W = np.delete(W, 0, 0)
    # W.T shape == (1 x D)
    # X.T shape == (D x N)
    Y = (W.T @ X.T).T + bias
    # Y shape == (D+1 x 1)
    return sigmoid(Y)
