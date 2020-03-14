#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io

def task1_1(X, Y):
    # Input:
    #  X : N-by-D data matrix (np.double)
    #  Y : N-by-1 label vector (np.int32)
    # Variables to save
    #  S : D-by-D covariance matrix (np.double) to save as 't1_S.mat'
    #  R : D-by-D correlation matrix (np.double) to save as 't1_R.mat'

    scipy.io.savemat('t1_S.mat', mdict={'S': S})
    scipy.io.savemat('t1_R.mat', mdict={'R': R})
