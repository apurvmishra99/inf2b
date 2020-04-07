import task1_1
import task1_3
import scipy.io
import numpy as np


if __name__ == "__main__":
    data = scipy.io.loadmat('../data/dset.mat')
    X_vec = data.get('X')
    Y_species = data.get('Y_species')
    Cov = task1_1.my_cov(X=X_vec)
    task1_3.task1_3(Cov)