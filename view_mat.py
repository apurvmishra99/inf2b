#! /usr/bin/env python

import scipy.io
import pandas as pd
from sys import argv

file_name = argv[1]
mat = scipy.io.loadmat(file_name)
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
data.to_csv(f"csvs/{file_name.split('.')[0]}.csv")