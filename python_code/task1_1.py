#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io

def my_mean(matrix):
    """
    Calculates mean values in a matrix
    Parameters:
      matrix: N-by-D data matrix
    Returns:
      mu: D-by-1 column vector of sample mean values, where mu(i) = mean(matrix(:,i)).
    """
    
    # Check if the matrix is not empty to make sure we do not divide by 0.
    if matrix.shape[0] == 0:
        s = 1
    else: 
        s = matrix.shape[0]
    
    # Compute sample mean vector
    mu = (np.sum(matrix, axis=0) / s).T 

    return mu

def my_cov(X):
    """
    Calculates covariance of a NxD matrix.
    Parameters:
      matrix: N-by-D data matrix
    Returns:
      cov: D-by-D covariance matrix.
    """
    X_mean = my_mean(X)
    X_shifted = X - X_mean
    # Using Maximum Likelihood Estimation so divide by N
    cov = np.matmul(X_shifted.T, X_shifted) / X.shape[0]
    return cov

def my_corr(X):
    """
    Calculates correlation of a NxD matrix.
    Parameters:
      matrix: N-by-D data matrix
    Returns:
      cov: D-by-D correlation matrix.
    """
    c = my_cov(X)
    
    try:
        d = np.diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    std_dev = np.sqrt(d.real)
    c /= (std_dev[:, None] * std_dev[None, :])

    return c

def task1_1(X, Y):
    # Input:
    #  X : N-by-D data matrix (np.double)
    #  Y : N-by-1 label vector (np.int32)
    # Variables to save
    #  S : D-by-D covariance matrix (np.double) to save as 't1_S.mat'
    #  R : D-by-D correlation matrix (np.double) to save as 't1_R.mat'

    S = my_cov(X)
    R = my_corr(X)

    scipy.io.savemat('t1_S.mat', mdict={'S': S})    
    scipy.io.savemat('t1_R.mat', mdict={'R': R})


if __name__ == "__main__":
    data = scipy.io.loadmat('../data/dset.mat')
    X_vec = data.get('X')
    Y_species = data.get('Y_species')
    task1_1(X=X_vec, Y=Y_species)