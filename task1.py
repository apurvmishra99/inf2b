import scipy.io as sio

dat = sio.loadmat('data/dset.mat')
print(dat.keys())
print(dat['list_family'][0][0])