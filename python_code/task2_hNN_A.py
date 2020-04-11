#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io

def task2_hNN_A(X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # Weights for first layer
    W1 = np.asanyarray([-3.325, -0.05, 1])
    W2 = np.asanyarray([8.5, -1, 0])
    W3 = np.asanyarray([303.0/56, 11.0/28, -1])
    W4 = np.asanyarray([-7.875, 1.25, 1])

    # Outputs from first layer
    Y1 = task2_hNeuron(W1, X)
    Y2 = task2_hNeuron(W2, X)
    Y3 = task2_hNeuron(W3, X)
    Y4 = task2_hNeuron(W4, X)
    
    #  Weights for second layer
    W5 = np.array([-3.5, 1, 1, 1, 1])

    X_out = np.vstack((Y1,Y2,Y3,Y4))

    Y = task2_hNeuron(W5, X_out)

    return Y
