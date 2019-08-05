from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="/Users/mac/PycharmProjects/ZJUsummer/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # load all data
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # idx_features_labels.shape (2708, 1435)
    # idx_features_labels[0:5]\
    # ([['31336', '0', '0', ..., '0', '0', 'Neural_Networks'],
    #    ['1061127', '0', '0', ..., '0', '0', 'Rule_Learning'],
    #    ['1106406', '0', '0', ..., '0', '0', 'Reinforcement_Learning'],
    #    ['13195', '0', '0', ..., '0', '0', 'Reinforcement_Learning'],
    #    ['37879', '0', '0', ..., '0', '0', 'Probabilistic_Methods']],
    # dtype='|S22')

    # form a feature matrix
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # features.shape: (2708, 1433)
    # (0, 0) 0.0
    # (0, 1) 0.0
    # (0, 2) 0.0
    # (0, 3) 0.0
    # (0, 4) 0.0
    # (0, 5)
    # ::
    # (2707, 1408) 0.0
    # (2707, 1409) 0.0
    # ::

    labels = encode_onehot(idx_features_labels[:, -1])
    # labels.shape (2708, 7)
    # labels[0:5]
    # array([[0, 0, 0, 1, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 1],
    #        [0, 1, 0, 0, 0, 0, 0],
    #        [0, 1, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 1, 0, 0]], dtype=int32)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    # list(enumerate(idx))[0:5] >>> [(0, 31336), (1, 1061127), (2, 1106406), (3, 13195), (4, 37879)]
    # idx_map
    # {(31336, 0), (1061127, 1), (1106406, 2), (13195, 3), (37879, 4)}

    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj.shape (2708, 2708)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    # features.todense() Return a dense matrix representation of this matrix.
    # [[0. 0. 0.... 0. 0. 0.]
    #  [0. 0. 0.... 0. 0. 0.]
    #  ...]]

    # features.todense(): features of each paper (2708, 1433)
    # adj: adjacent matrix (2708, 2708)
    # labels: labels of all papers after encode_onehot (2708, 7)
    return features.todense(), adj, labels


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
        # transpose
        # C = AB, D = B^T * A^T, C^T = (AB)^T = B^T * A^T = D^T
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    # The normalized adjacency matrix is D^-1/2 A D^-1/2 : https://people.orie.cornell.edu/dpw/orie6334/lecture7.pdf
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    # sp.eye(): Return a 2-D array with ones on the diagonal and zeros elsewhere.
    # the reason to do this is to alleviate the problem that repeated application
    # of this operator (D^-1/2 A D^-1/2) can lead to numerical instabilities
    # and exploding/vanishing gradients when used in a deep neural network model. [P3 in paper]
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    # choose 0-139 in labels as train set
    y_val[idx_val] = y[idx_val]
    # choose 200-499 in labels as val set
    y_test[idx_test] = y[idx_test]
    # choose 500-1499 in labels as test set
    train_mask = sample_mask(idx_train, y.shape[0])
    # train_mask is an array [true, true , ... , false, false, .., false],  0-139 is true, 140-2707 is false
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    # extract: Return the elements of an array that satisfy some condition.
    # mean: The arithmetic mean is the sum of the elements along the axis divided by the number of elements.
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape