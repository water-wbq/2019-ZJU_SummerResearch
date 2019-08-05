import pandas as pd
import numpy as np

# use the demand of next time t+1 as the predicted demand of t

dataset = pd.read_csv('data/data.csv', header=None, index_col=None)
values = dataset.values.astype('float32')

res = []

for i in range(len(values)-1):
    tmp = (values[i+1] -values[i])**2
    res.append(tmp)

mse = np.mean(res)
rmse = np.sqrt(mse)

print (rmse)

# rmse: 35.804382