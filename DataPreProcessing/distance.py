import pandas as pd
import numpy as np
distance_path = '../OriginalData/30PointsDistance.csv'


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print('input file was not found')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W

w = weight_matrix(distance_path)



import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

plt.matshow(w, cmap='hot')
plt.colorbar()
plt.show()

df = pd.DataFrame(w)
df.to_csv('../data/dis_weight.csv')