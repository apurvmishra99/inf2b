#
# Version 0.9  (HS 09/03/2020)
#
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import numpy as np
import scipy.io
from task2_hNeuron import task2_hNeuron

def task2_hNN_A(X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # First layer weights
    W1, W2, W3, W4 = np.asanyarray([
                            [-2.0096131 ,  0.33077714,  1.        ],
                            [ 1.        , -0.45212818,  0.06827846],
                            [ 1.        , -0.18659937, -0.27743793],
                            [-1.20418437,  1.        , -0.40741555]])
       
    # Applying weights and calculating the outputs
    Y1 = task2_hNeuron(W1.reshape((3,1)), X)
    Y2 = task2_hNeuron(W2.reshape((3,1)), X)
    Y3 = task2_hNeuron(W3.reshape((3,1)), X)
    Y4 = task2_hNeuron(W4.reshape((3,1)), X)
    
    #  Second layer weight
    W5 = np.array([-3.5, 1, 1, 1, 1]).reshape((5,1))

    X_out = np.hstack((Y1,Y2,Y3,Y4))

    Y = task2_hNeuron(W5, X_out)

    return Y
