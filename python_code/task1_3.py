#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io

def task1_3(Cov):
    # Input:
    #  Cov : D-by-D covariance matrix (np.double)
    # Variales to save:
    #  EVecs : D-by-D matrix of column vectors of eigen vectors (np.double)  
    #  EVals : D-by-1 vector of eigen values (np.double)  
    #  Cumvar : D-by-1 vector of cumulative variance (np.double)  
    #  MinDims : 4-by-1 vector (np.int32)  
    
    scipy.io.savemat('t1_EVecs.mat', mdict={'EVecs': EVecs})
    scipy.io.savemat('t1_EVals.mat', mdict={'EVals': EVals})
    scipy.io.savemat('t1_Cumvar.mat', mdict={'Cumvar': Cumvar})
    scipy.io.savemat('t1_MinDims.mat', mdict={'MinDims': MinDims})


