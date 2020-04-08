#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io

def _split(X, k, mc):
    curr = 1
    sub_arrays = []
    for i in range(k):
        if curr != k:
            start = i*mc
            end = i*mc + mc
            sub_arrays.append(X[start:end])
        else:
            start = i*mc
            sub_arrays.append(X[start:])
        curr += 1
    return sub_arrays

def my_k_folds(X, Y, k_folds):
    # Assuming classes follow the pattern 1..C;
    # C becomes the total no. of classes
    no_of_classes = np.max(Y)
    p_map = np.zeros((len(X), 1))
    
    for c in range(1, no_of_classes+1):
        Nc = np.count_nonzero(Y == c)
        Mc = int(np.floor(Nc / k_folds))
        # find indexes of all places where the element is c
        idxs = np.apply_along_axis(lambda x: x[0], 1, np.argwhere(Y == c))
        # split the idxs based on k_folds and Mc
        sub_arrs = _split(idxs, k_folds, Mc)
        for i in range(len(sub_arrs)):
            p_map[sub_arrs[i]] = i + 1
            
    return p_map


def task1_mgc_cv(X, Y, CovKind, epsilon, Kfolds):
    # Input:
    #  X : N-by-D matrix of feature vectors (np.double)
    #  Y : N-by-1 label vector (np.int32)
    #  CovKind : scalar (np.int32)
    #  epsilon : scalar (np.double)
    #  Kfolds  : scalar (np.int32)
    #
    # Variables to save
    #  PMap   : N-by-1 vector of partition numbers (np.int32)
    #  Ms     : C-by-D matrix of mean vectors (np.double)
    #  Covs   : C-by-D-by-D array of covariance matrices (np.double)
    #  CM     : C-by-C confusion matrix (np.double)
    
    # scipy.io.savemat('t1_mgc_<Kfolds>cv_PMap.mat', mdict={'PMap':PMap})
    # For each <p> and <CovKind>
    #  scipy.io.savemat('t1_mgc_<Kfolds>cv<p>_Ms.mat', mdic={'Ms':Ms})
    #  scipy.io.savemat('t1_mgc_<Kfolds>cv<p>_ck<CovKind>_Covs.mat', mdict={'Covs': Cov})
    #  scipy.io.savemat('t1_mgc_<Kfolds>cv<p>_ck<CovKind>_CM.mat', mdict={'CM':CM});
    #  scipy.io.savemat('t1_mgc_<Kfolds>cv<L>_ck<CovKind>_CM.mat', mdict={'CM':CM});
    # NB: replace <Kfolds>, <p>, and <CovKind> properly.

    

