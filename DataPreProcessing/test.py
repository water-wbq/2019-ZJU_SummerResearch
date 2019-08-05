import pandas as pd
import numpy as np
distance_path = '../data/weight_adj.csv'


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    W = pd.read_csv(file_path, header=None).values
    return W

w = weight_matrix(distance_path)
print(w.shape)



import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

plt.matshow(w, cmap='hot')
plt.colorbar()
plt.show()