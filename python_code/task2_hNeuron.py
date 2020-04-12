#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io


def step(y):
    return 1 * (y > 0)


def task2_hNeuron(W, X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    #  W : (D+1)-by-1 vector of weights (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    bias = W[0]
    W = np.delete(W, 0, 0)
    # W.T shape == (1 x D)
    # X.T shape == (D x N)
    Y = (W.T @ X) + bias
    # Y shape == (1 x N)
    return step(Y.T)
