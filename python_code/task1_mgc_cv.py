#
# Verscipy.ion 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
from scipy.stats import multivariate_normal


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
    cov = (X_shifted.T @ X_shifted) / X.shape[0]
    return cov


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


def my_confusion_matrix(y_actual, y_pred):
    '''
    Parameters:
        y_actual: actual labels
        y_pred: predicted labels of data
    Returns:
        CM: Confusion matrix for the data as a 2d numpy array
    '''
    if(y_actual.shape[0] != y_pred.shape[0]):
        raise ValueError("Vectors supplied must be of same length")

    # Convert the arrays to int32 to avoid indexing errors
    y_actual = y_actual.astype(np.int32, copy=False)
    y_pred = y_pred.astype(np.int32, copy=False)

    # Minimum and maximum values in order to know the size of the confusion matrix and how to access it
    # Problems come from classes indexing beginning with 1 and data array indexing with 0
    max_actual = np.max(y_actual)
    max_predicted = np.max(y_pred)
    min_actual = np.min(y_actual)
    min_predicted = np.min(y_pred)

    N = max_actual + 1 if min_actual == 0 else max_actual
    M = max_predicted + 1 if min_predicted == 0 else max_predicted

    confusion_matrix = np.zeros((N, M))

    for index in range(y_pred.shape[0]):
        i = y_actual[index] if min_actual == 0 else y_actual[index] - 1
        j = y_pred[index] if min_predicted == 0 else y_pred[index] - 1
        confusion_matrix[i, j] += 1

    return confusion_matrix


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
    #  CM     : C-by-C confuscipy.ion matrix (np.double)

    PMap = my_k_folds(X, Y, Kfolds)
    scipy.io.savemat(f"t1_mgc_{Kfolds}cv_PMap.mat", {"PMap": PMap})
    no_of_classes = np.max(Y)
    total_cov_mat = np.zeros((no_of_classes, no_of_classes))
    D = X[0].size
    for p in range(1, Kfolds+1):
        Ms = np.zeros((no_of_classes, D))
        Covs = np.zeros((no_of_classes, D, D))
        priors = np.zeros((1, no_of_classes))

        for c in range(1, no_of_classes+1):
            train_data = X[np.where(np.logical_and(PMap != p, Y == c))[0]]
            Ms[c-1] = my_mean(train_data)

            if CovKind == 1:
                # Full covariance mat
                Cov = my_cov(train_data)
            elif CovKind == 2:
                # Diagonal covariance mat
                Cov = np.diag(np.diag(my_cov(train_data)))
            else:
                # Shared covariance mat
                Cov = my_cov(X)

            standardised_cov = Cov + epsilon * np.eye(D)
            Covs[c-1] = standardised_cov

            priors[:, c-1] = len(train_data) / len(X[np.where(PMap != p)[0]])

        scipy.io.savemat(f"t1_mgc_{Kfolds}cv{p}_Ms.mat", {"Ms": Ms})
        scipy.io.savemat(
            f"t1_mgc_{Kfolds}cv{p}_ck{CovKind}_Covs.mat", {"Covs": Covs})

        test_data = X[np.where(PMap == p)[0]]
        test_likelihoods = np.zeros((len(test_data), no_of_classes))

        for i in range(len(test_data)):
            for c in range(1, no_of_classes+1):
                test_likelihoods[i, c-1] = multivariate_normal.pdf(test_data[i], Ms[c-1], np.squeeze(Covs[c-1])
                                                                   )
        test_probs = test_likelihoods * priors
        test_preds = np.zeros((len(test_data), 1))

        for i in range(len(test_data)):
            curr_row_vec = test_probs[i]
            test_preds[i] = np.argmax(curr_row_vec, axis=0)

        CM = my_confusion_matrix(Y[np.where(PMap == p)[0]], test_preds)

        scipy.io.savemat(
            f"t1_mgc_{Kfolds}cv{p}_ck{CovKind}_CM.mat", {"CM": CM})

        total_cov_mat = total_cov_mat + (CM / (Kfolds * len(test_data)))
    L = Kfolds + 1
    scipy.io.savemat(f"t1_mgc_{Kfolds}cv{L}_ck{CovKind}_CM.mat", {
                     "CM": total_cov_mat})
