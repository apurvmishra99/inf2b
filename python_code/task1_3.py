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
    
    
    eigen_vals, eigen_vecs = np.linalg.eig(Cov)
    idx = np.argsort(eigen_vals)[::-1]
    EVals = eigen_vals[idx]
    EVecs = eigen_vecs[:, idx]

    # Change direction of negative evecs
    np.apply_along_axis(lambda x : x*(-1) if x[0] < 0 else x, 0, EVecs)
    # caluclate the cumulative variance
    Cumvar = np.cumsum(EVals)
    # calculate min number of PCA dimensions required for the following percentages
    req_cov_lst = [0.70, 0.80, 0.90, 0.95]
    MinDims = np.zeros((len(req_cov_lst),), dtype=np.int32)
    for i, cov_for in enumerate(req_cov_lst):
        for k in range(EVals.size):
            if Cumvar[k]/Cumvar[EVals.size - 1] >= cov_for:
                MinDims[i] = k
                break

    scipy.io.savemat('t1_EVecs.mat', mdict={'EVecs': EVecs})
    scipy.io.savemat('t1_EVals.mat', mdict={'EVals': EVals})
    scipy.io.savemat('t1_Cumvar.mat', mdict={'Cumvar': Cumvar})
    scipy.io.savemat('t1_MinDims.mat', mdict={'MinDims': MinDims})
