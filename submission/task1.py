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
    cov = (X_shifted.T @ X_shifted) / X.shape[0]
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
    c /= (std_dev[:, None] @ std_dev[None, :])

    return c

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
    Calculates the confusion matrix given true and predicted labels.
    Parameters:
        y_actual: actual labels
        y_pred: predicted labels of data
    Returns:
        CM: Confusion matrix for the data as a 2d numpy array
    '''

    # Convert the arrays to int32 to avoid indexing errors
    y_actual = y_actual.astype(np.int32, copy=False)
    y_pred = y_pred.astype(np.int32, copy=False)

    k = np.max(y_actual)
    confusion_matrix = np.zeros((k, k))

    for actual, pred in zip(y_actual, y_pred):
        confusion_matrix[actual-1, pred-1] += 1

    return confusion_matrix


def gaussian_mvn_pdf(X, mean, cov):
    '''
    Return posterior probabilities approximated by a Gaussian with provided mean and covariance.
    Params:
        X: Data to be classified (Dx1)
        mean: Mean vector of the data (Dx1)
        cov: Covariance matrix of the data (DxD)
    Returns:
        p: posterior probabilities
    '''
    D = det(cov)
    inv_cov = inv(cov)
    X_shift = X - mean
    p_1 = 1 / (((2 * np.pi)**(len(mean)/2)) * (D**(1/2)))
    p_2 = (-1/2) * ((X_shift.T) @ (inv_cov) @ (X_shift))
    prior_prob = p_1 * np.exp(p_2)
    return prior_prob

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
    prop = Evals / sum(Evals)
    Cumvar = np.cumsum(prop)
    # calculate min number of PCA dimensions required for the following percentages
    req_cov_lst = [0.70, 0.80, 0.90, 0.95]
    MinDims = np.zeros((len(req_cov_lst),), dtype=np.int32)
    for i, cov_for in enumerate(req_cov_lst):
        for k in range(EVals.size):
            if Cumvar[k]/Cumvar[EVals.size - 1] >= cov_for:
                MinDims[i] = k+1
                break

    scipy.io.savemat('t1_EVecs.mat', mdict={'EVecs': EVecs})
    scipy.io.savemat('t1_EVals.mat', mdict={'EVals': EVals})
    scipy.io.savemat('t1_Cumvar.mat', mdict={'Cumvar': Cumvar})
    scipy.io.savemat('t1_MinDims.mat', mdict={'MinDims': MinDims})


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
                test_likelihoods[i, c-1] = gaussian_mvn_pdf(test_data[i], Ms[c-1], np.squeeze(Covs[c-1])
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
    
    # Final confusion matrix
    scipy.io.savemat(f"t1_mgc_{Kfolds}cv{Kfolds+1}_ck{CovKind}_CM.mat", {
                     "CM": total_cov_mat})

if __name__ == "__main__":
    data = scipy.io.loadmat('dset.mat')
    X_vec = data.get('X')
    Y_species = data.get('Y_species')

    # Task 1.1
    task1_1(X=X_vec, Y=Y_species)

    # Task 1.3
    cov = my_cov(X_vec)
    task1_3(cov)

    # Task 1.4
    for i in [1,2,3]:
        task1_mgc_cv(X_vec, Y, i, 0.05, 5)