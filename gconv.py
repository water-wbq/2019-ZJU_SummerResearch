import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse import linalg


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(
        d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum(adj_mx, adj_mx.T)
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


class gconv(tf.keras.Model):
    def __init__(self,
                 num_units,
                 adj_mx,
                 max_diffusion_step,
                 num_nodes,
                 num_proj=None,
                 activation=tf.nn.tanh,
                 reuse=None,
                 filter_type="laplacian"):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".

        """
        super(gconv, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx,
                                                       lambda_max=None))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def __call__(self, inputs, bias_start=0.0):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input to (batch_size, num_nodes, input_dim)
        output_size = self._num_units
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, [batch_size, self._num_nodes, -1])
        input_size = inputs.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs
        x0 = tf.transpose(x, perm=[1, 2,0])  # (num_nodes, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            if self._max_diffusion_step == 0:
                pass
            else:
                for support in self._supports:
                    x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    x = self._concat(x, x1)

                    for _ in range(2, self._max_diffusion_step + 1):
                        x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
            x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(
                x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable("biases", [output_size],
                                     dtype=dtype,
                                     initializer=tf.constant_initializer(
                                         bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to: (batch_size, num_node, state_dim)
        return tf.reshape(x, [batch_size, self._num_nodes, output_size])
