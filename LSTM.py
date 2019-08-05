import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

seq_length = 12
batch_vol = 5000


def get_batch(data, seq_length, batch_vol):
    x = []
    y = []
    for i in range(batch_vol):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)


# load dataset
dataset = pd.read_csv('data/data.csv', header=None, index_col=None)
values = dataset.values.astype('float64')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# split into train and test sets
n_train = int(len(scaled)*0.7)+1
n_val = int(len(scaled)*0.1)
n_test = int(len(scaled)*0.2)

train = scaled[:n_train, :]
val = scaled[n_train:n_train+n_val, :]
test = scaled[n_train+n_val:, :]

train_X, train_Y = get_batch(train, seq_length, 6400)
test_X, test_Y = get_batch(test, seq_length, 1600)

# design network
# use default parameters
model = Sequential()
# model.add(LSTM (unit, input_shape(input_length,input_dimension)))
model.add(LSTM(16, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(30))
model.compile(loss='mse', optimizer='adam')

# fit network
# here the batch_size is the number of instances to be trained in the network at a time
history = model.fit(train_X, train_Y, epochs=50, batch_size=32,
                    validation_data=(test_X, test_Y), verbose=2, shuffle=False)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model.predict(test_X)
inv_yhat = scaler.inverse_transform(yhat)
inv_test_Y = scaler.inverse_transform(test_Y)

rmse = sqrt(mean_squared_error(inv_yhat, inv_test_Y))
print('Test RMSE: %.3f' % rmse)

y_val_table = pd.DataFrame(inv_test_Y)
y_val_table.to_csv('data/y_val_table.csv', index=False)
y_pred_table = pd.DataFrame(inv_yhat)
y_pred_table.round(decimals=0)
y_pred_table[y_pred_table < 0] = 0
y_pred_table.to_csv('data/y_pred_table.csv', index=False)

rmse = sqrt(mean_squared_error(y_val_table, y_pred_table))
print('Test RMSE: %.3f' % rmse)

# Test RMSE: 29.434