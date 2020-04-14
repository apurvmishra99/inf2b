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
from task2_sNeuron import task2_sNeuron

def sNN(X, flag):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)
    
    if flag == "A":
        # First layer weights
        multiplication_factor = 350
        W1, W2, W3, W4 = multiplication_factor *  np.asanyarray([
                                [-2.0096131 ,  0.33077714,  1.        ],
                                [ 1.        , -0.45212818,  0.06827846],
                                [ 1.        , -0.18659937, -0.27743793],
                                [-1.20418437,  1.        , -0.40741555]])
        
    else:
        # First layer weights
        multiplication_factor = 300
        W1, W2, W3, W4 = multiplication_factor * np.asanyarray([
                                [ 1.        ,  0.98977494, -0.53907994],
                                [ 1.        , -0.27213978, -0.03654737],
                                [ 1.        , -0.18637246, -0.17595583],
                                [-0.78936275,  0.2933849 ,  1.        ]])
        
                      
    # Applying weights and calculating the outputs
    Y1 = task2_sNeuron(W1.reshape((3,1)), X)
    Y2 = task2_sNeuron(W2.reshape((3,1)), X)
    Y3 = task2_sNeuron(W3.reshape((3,1)), X)
    Y4 = task2_sNeuron(W4.reshape((3,1)), X)
    
    #  Second layer weight
    W5 = np.array([-3.5, 1, 1, 1, 1]).reshape((5,1))

    X_out = np.hstack((Y1,Y2,Y3,Y4))

    Y = task2_sNeuron(multiplication_factor *  W5, X_out)

    # For all outputs greater than 0.5 we classify them class 1
    return Y > 0.5

def task2_sNN_AB(X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # Output from our network for points in A
    Y_A = sNN(X, "A")
    # Output from our network for points in B
    Y_B = sNN(X, "B")

    Y_A_B = np.hstack((Y_A,Y_B))

    # we are modelling a neuron with binary inputs x1 and x2 which
    # outputs 1 if and only if  x1 = 0 and x2 = 1, as seen in lectures
    # the line -x + y = 0 will be sufficient 
    # therefore,  w = [0, -1, 1]
    
    Y = task2_sNeuron(300 * np.array([0,-1,1]).reshape((3,1)), Y_A_B)
    
    return Y > 0.5
    