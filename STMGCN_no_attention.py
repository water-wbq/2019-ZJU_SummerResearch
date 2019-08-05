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
LEARNING_RATE = 0.001
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


def lstm_network(x):

    def lstm_cell():
        return tf.nn.rnn_cell.LSTMCell(HIDDEN_UNITS)

    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(NUM_LAYER)])

    outputs, state_tuple = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=x,
        dtype=tf.float32)
    print (outputs.shape)
    network_output = outputs[:, -1, :]
    return network_output


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

split = tf.split(x, NODES, 2) # [32, 12, 1]
print ('split:{},split_len:{},split[0]:{},split[0]_len:{}'.format(type(split),len(split),type(split[0]),split[0].shape))


with tf.variable_scope('lstm',reuse=tf.AUTO_REUSE):
    tmp = [0] * NODES
    for i in range(len(split)):
        output_each_node = lstm_network(split[i])
        # print('output_each_node',output_each_node.shape)
        # tmp[i] = tf.expand_dims(output_each_node, 1)
        # print ('type tmp[i]',type(tmp[i]))
        tmp[i] = tf.reshape(output_each_node, [1, BATCH_SIZE, HIDDEN_UNITS])
    # print ('type tmp', type(tmp))
    lstm_output = tf.concat(tmp,0)
    # print('lstm_output', lstm_output)
    lstm_output = tf.transpose(lstm_output, perm=[1, 0, 2])

    print('lstm_output', lstm_output)


with tf.variable_scope('simi_adj'):
    GCN = gconv(HIDDEN_UNITS,simi_adj,2,NODES)
    gcn_output1 = GCN(lstm_output)
    print("simi_adj", gcn_output1)

with tf.variable_scope('dis_adj'):
    GCN = gconv(HIDDEN_UNITS,dis_adj,2,NODES)
    gcn_output2 = GCN(lstm_output)
    print("dis_adj", gcn_output2)

with tf.variable_scope('cont_adj'):
    GCN = gconv(HIDDEN_UNITS,cont_adj,2,NODES)
    gcn_output3 = GCN(lstm_output)
    print("cont_adj", gcn_output3)

network_output = tf.add(tf.add(gcn_output1, gcn_output2), gcn_output3)
all_output = tf.layers.dense(inputs=network_output, units=1, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

all_output = tf.reshape(all_output, shape=[-1, NODES])
print('all_output', all_output)

# print ('y',y)
# y_loss = scaler.inverse_transform(y)
# print ('y_loss',y_loss)
#
# all_output_loss = scaler.inverse_transform(all_output)
# print('all_output_loss',all_output_loss)


# loss = tf.losses.mean_squared_error(labels=y_loss, predictions=all_output_loss)
# training_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

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


# path = 'out/'
# if not os.path.exists(path):
#     os.makedirs(path)


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

    # print (len(y_val_all_batch), len(y_pred_all_batch))
    # y_val_table = pd.DataFrame(y_val_all_batch)
    # y_val_table.to_csv('/Users/mac/Desktop/materials/data/y_val_table.csv', index=False)
    # y_pred_table = pd.DataFrame(y_pred_all_batch)
    # y_pred_table.to_csv('/Users/mac/Desktop/materials/data/y_pred_table.csv', index=False)



    # if (e % 1 == 0):
    #     saver.save(sess, path + '/stmgcn_%r' % e, global_step=e)


    # res_variance_unscaled = []
    # for _ in range(50):
    #     x_batch_test, y_batch_test = get_batch(
    #         X=X_test,
    #         y=y_test
    #     )
    #
    #     y_pred = sess.run(
    #         all_output,
    #         feed_dict={x: x_batch_test})
    #
    #     y_pred_unscaled = scaler.inverse_transform(y_pred)
    #     test_y = scaler.inverse_transform(y_test)
    #
    #     variance_one_batch = (y_batch_test - y_pred_unscaled) ** 2
    #     res_variance_unscaled.append(variance_one_batch)
    #
    # mse = np.mean(res_variance_unscaled)
    # rmse = np.sqrt(mse)
    #
    # print('Test Set Performance: Unscaled MSE: {} Unscaled RMSE: {}'.format(mse, rmse))

    # if (e != 0 and e % 5 == 0):
    #     save_model(checkpoint_path, model_name, e, rmse)

# best rmse: 20
