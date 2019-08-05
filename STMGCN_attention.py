# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os

from gconv import gconv

TIME_STEPS = 12
BATCH_SIZE = 32
HIDDEN_UNITS = 16
LEARNING_RATE = 0.01
EPOCH = 50
NODES = 30
KEEP_DROP = 0.2
INPUT_SIZE = 30
OUTPUT_SIZE = 30
REGULARIZAER = 0.003

TRAIN_EXAMPLES = 6400
VAL_EXAMPLES = 800
TEST_EXAMPLES = 1600
NUM_LAYER = 3

scaler = MinMaxScaler(feature_range=(0, 1))

def generate(seq, vol):
    X = []
    y = []
    for i in range(vol):
        X.append([seq[i:i + TIME_STEPS]])
        y.append([seq[i + TIME_STEPS]])

    X = np.array(X, dtype=np.float32)
    X = X.reshape(-1, TIME_STEPS, NODES)

    Y = np.array(y, dtype=np.float32)
    Y = Y.reshape(-1, NODES)
    return X, Y


def load_data():
    # load dataset
    dataset = pd.read_csv('data/data.csv', header=None, index_col=None)
    values = dataset.values.astype('float32')

    # normalize features
    scaled = scaler.fit_transform(values)

    # split into train and test sets
    n_train = int(len(scaled) * 0.7) + 1
    n_val = int(len(scaled) * 0.1)
    n_test = int(len(scaled) * 0.2)

    train = scaled[:n_train, :]
    val = scaled[n_train:n_train + n_val, :]
    test = scaled[n_train + n_val:, :]

    X_train, y_train = generate(train, TRAIN_EXAMPLES)
    X_val, y_val = generate(val, VAL_EXAMPLES)
    X_test, y_test = generate(test, TEST_EXAMPLES)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_adj():
    simi_adj = pd.read_csv('data/weight_simi.csv', header=None, index_col=None)
    simi_adj = np.mat(simi_adj)

    dis_adj = pd.read_csv('data/weight_dis.csv', header=None, index_col=None)
    dis_adj = np.mat(dis_adj)

    cont_adj = pd.read_csv('data/weight_adj.csv', header=None, index_col=None)
    cont_adj = np.mat(cont_adj)

    return simi_adj, dis_adj, cont_adj


def get_W(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    w=tf.get_variable('W', dtype=tf.float32,shape=shape,initializer=initializer)
    return w


def get_b(shape):
    initializer = tf.constant(0.,shape=shape,dtype=tf.float32)
    b=tf.get_variable('b',dtype=tf.float32,initializer=initializer)
    return b


def attention(x, adj):
#   x_pool_input = tf.reshape(x, [BATCH_SIZE, TIME_STEPS, NODES, 1])
#   x_pool = tf.layers.average_pooling2d(x_pool_input, [TIME_STEPS, NODES], [1,1], 'VALID')
    print ('----attention----')
    x_pool = tf.reduce_sum(x,2)

    x_gcn_input = tf.transpose(x, perm=[0, 2, 1])
    x_gcn_input = tf.reshape (x_gcn_input, [BATCH_SIZE, NODES, TIME_STEPS])
    GCN = gconv(TIME_STEPS, adj, 2, NODES)
    x_gcn_output = GCN(x_gcn_input)
    x_gcn_pool = tf.reduce_sum(x_gcn_output,1)

    x_hat =tf.add(x_pool,x_gcn_pool)
    z = tf.divide(x_hat,NODES)

    with tf.variable_scope('w1_attention',reuse=tf.AUTO_REUSE):
        w1 = get_W([TIME_STEPS,TIME_STEPS])
        b1 = get_b([TIME_STEPS])
        tmp_s =tf.nn.relu(tf.add(tf.matmul(z,w1),b1))

    with tf.variable_scope('w2_attention', reuse=tf.AUTO_REUSE):
        w2 = get_W([TIME_STEPS, TIME_STEPS])
        b2 = get_b([TIME_STEPS])
        s= tf.nn.sigmoid(tf.add(tf.matmul(tmp_s,w2),b2))

    s = tf.reshape(s, [BATCH_SIZE,TIME_STEPS,1])
    x_reweight = tf.multiply(x,s)

    return x_reweight


def lstm_network(x):

    def lstm_cell():
        return tf.nn.rnn_cell.LSTMCell(HIDDEN_UNITS)

    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(NUM_LAYER)])

    outputs, state_tuple = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=x,
        dtype=tf.float32)
    # outputs: (?, 12, 16)
    network_output = outputs[:, -1, :]
    return network_output

def stmgcn(x, adj):
    x_reweight = attention(x, adj)  # shape=(32, 12, 30)

    split = tf.split(x_reweight, NODES, 2)  # [32, 12, 1]

    with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
        tmp = [0] * NODES
        for i in range(len(split)):
            output_each_node = lstm_network(split[i])
            tmp[i] = tf.reshape(output_each_node, [1, BATCH_SIZE, HIDDEN_UNITS])
        lstm_output = tf.concat(tmp, 0)
        lstm_output = tf.transpose(lstm_output, perm=[1, 0, 2])
        print('lstm_output', lstm_output)

    with tf.variable_scope('gcn'):
        GCN = gconv(HIDDEN_UNITS, adj, 2, NODES)
        gcn_output = GCN(lstm_output)

    return gcn_output

def get_batch(X, y):
    idx = np.random.randint(X.shape[0] - BATCH_SIZE)
    x_batch = X[idx: idx + BATCH_SIZE]
    y_batch = y[idx: idx + BATCH_SIZE]
    return x_batch, y_batch


simi_adj, dis_adj, cont_adj = load_adj()
X_train, y_train, X_val, y_val, X_test, y_test = load_data()
print ('X_train:{}, y_train:{}'.format(X_train.shape,y_train.shape))
print ('X_val:{}, y_val:{}'.format(X_val.shape,y_val.shape))

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, TIME_STEPS, INPUT_SIZE])  # [None, 12, 30]
y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])  # [None, 30]

with tf.variable_scope('simi_adj'):
    gcn_output1 = stmgcn(x,simi_adj)

with tf.variable_scope('dis_adj'):
    gcn_output2 = stmgcn(x,simi_adj)

with tf.variable_scope('cont_adj'):
    gcn_output3 = stmgcn(x,simi_adj)

network_output = tf.add(tf.add(gcn_output1, gcn_output2), gcn_output3)  # (32, 30, 16)
all_output = tf.layers.dense(
    inputs=network_output,
    units=1,
    activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZAER))

all_output = tf.reshape(all_output, shape=[-1, NODES])
print('all_output', all_output)  # (32, 30)


loss = tf.losses.mean_squared_error(labels=y,predictions=all_output)
training_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
sess = tf.InteractiveSession()
sess.run(init)

loss_cache = []
val_loss_cache = []

train_batch_number = int(X_train.shape[0] / BATCH_SIZE)
val_batch_number = int(X_val.shape[0] / BATCH_SIZE)


for e in range(EPOCH):
    for _ in range(train_batch_number):
        x_batch, y_batch = get_batch(
            X=X_train,
            y=y_train
        )

        _, train_loss = sess.run(
            [training_optimizer, loss],
            feed_dict={
                x: x_batch,
                y: y_batch})

        loss_cache.append(train_loss)

    res_variance_unscaled = []
    #
    # y_pred_all_batch = [[0] * NODES for i in range(BATCH_SIZE)]
    # y_val_all_batch = [[0] * NODES for i in range(BATCH_SIZE)]

    for _ in range(val_batch_number):
        x_batch_val, y_batch_val = get_batch(
            X=X_val,
            y=y_val
        )

        val_loss, y_pred_val = sess.run(
            [loss,all_output],
            feed_dict={
                x: x_batch_val,
                y: y_batch_val})

        y_pred_val_unscaled = scaler.inverse_transform(y_pred_val)
        y_batch_val_unscaled = scaler.inverse_transform(y_batch_val)

        # print('y_batch_val_unscaled',len(y_batch_val_unscaled))
        # y_pred_all_batch = y_pred_all_batch + y_pred_val_unscaled
        # y_val_all_batch = y_pred_all_batch + y_batch_val_unscaled

        variance_one_batch = (y_pred_val_unscaled - y_batch_val_unscaled) ** 2
        res_variance_unscaled.append(variance_one_batch)

        val_loss_cache.append(val_loss)

    # print ('y_pred_all_batch',len(y_pred_all_batch))
    mse = np.mean(res_variance_unscaled)
    rmse = np.sqrt(mse)

    print('Epoch: {}/{}\ttrain_loss: {}\tval_loss: {}'.format(e + 1, EPOCH, str(train_loss)[:8], str(val_loss)[:8]))
    print('Val Set Performance: Unscaled MSE: {} Unscaled RMSE: {}'.format(mse, rmse))


# best rmse: 23
