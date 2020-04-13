#! /usr/bin/env python
#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
from numpy.linalg import lstsq

def find_weights(point_1, point_2):
    points = [point_1, point_2]
    xs, ys = zip(*points)
    A = np.vstack([xs, np.ones(len(xs))]).T
    m, c = lstsq(A, ys)[0]

    weights = np.array([-1*c,-1*m,1])
    norm_weights = weights / weights[abs(weights).argmax()]
    return norm_weights

def load_data(filename, flag='A'):
    if flag == 'A':
        A, _ = np.genfromtxt(filename, usecols=range(1,9))
        points = list(zip(A[::2], A[1::2]))
        return points
    else:
        _, B = np.genfromtxt(filename, usecols=range(1,9))
        points = list(zip(B[::2], B[1::2]))
        return points

def calculate_hNN_A_weights(filename="data/task2_data.txt"):
    points = load_data(filename)
    weight_vecs = np.zeros((len(points), 3))
    w_count = 0
    for i in range(len(points)):
        point_1, point_2 = points[i], points[(i+1) % len(points)]
        weight_vecs[w_count] = find_weights(point_1, point_2)
        w_count += 1

    print(weight_vecs)

if __name__ == "__main__":
    calculate_hNN_A_weights()
    
