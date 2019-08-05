#coding:utf8
import os
import numpy as np
import pandas as pd

PATH = '../OriginalData/30Points40DaysDemand'

array = []
file = []
for day in os.listdir(PATH):
    domain = os.path.abspath(PATH)
    day = os.path.join(domain,day)
    file.append(day)
file.sort()
for f in file:
    data_per_day = pd.read_csv(f, header=None)
    data = data_per_day.values.tolist()
    array.extend(data)
df = pd.DataFrame(array)
# may not be ordered !!!!!
df.to_csv('../data/data.csv')



