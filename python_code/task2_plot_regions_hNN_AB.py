#! /usr/bin/env python
#
# Version 0.9  (HS 09/03/2020)
#
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm
from task2_hNN_AB import task2_hNN_AB

# Generating points between -2 and 7 to classify:
xs = np.linspace(-2, 7, 1000)
ys = np.linspace(-2, 7, 1000)
xx, yy = np.meshgrid(xs, ys)
grid = np.vstack((xx.ravel(), yy.ravel()))


# Calling task2_hNN_AB on the generated points
data = task2_hNN_AB(grid.T)

# Reshape the result to fit the plt function
data = data.reshape((len(xs), len(ys)))

# Setup the plot title, label and axis
plt.title('Task 2.7 Plot')
plt.xticks(np.arange(-2, 10, 0.5), fontsize=7)
plt.yticks(np.arange(-2, 10, 0.5), fontsize=7)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)

# Setup the legend
blue_patch_legend = mpatches.Patch(color='lightblue', label='Class 0')
purple_patch_legend = mpatches.Patch(color='purple', label='Class 1')
plt.legend(loc='best', fancybox=True, framealpha=0.1, handles=[
           blue_patch_legend, purple_patch_legend], facecolor='black', fontsize=12)

# Plot, save as pdf and show result
plt.contourf(xx, yy, data, cmap=cm.BuPu)
plt.savefig('t2_regions_hNN_AB.pdf')
plt.show()