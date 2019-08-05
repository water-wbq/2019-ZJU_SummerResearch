#coding:utf8
import os
import numpy as np
import pandas as pd

PATH = '../OriginalData'
poi_30_path = os.path.join(PATH, "poi_30.csv")

def cos_similarity(vector1, vector2):
    if ((np.linalg.norm(vector1)==0) and (np.linalg.norm(vector2)==0)):
        return 1
    elif ((np.linalg.norm(vector1)==0) or (np.linalg.norm(vector2)==0)):
        return 0
    else:
        simi = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        return simi

tmp = pd.read_csv(poi_30_path,index_col=0)

array = tmp.values
# print (array[0])
simi_array = []
for i in range(0,30):
    simi_array_row = []
    for j in range(0,30):
        simi = cos_similarity(array[i],array[j])
        simi_array_row.append(simi)
    simi_array.append(simi_array_row)
print (simi_array)
print (len(simi_array))

simi_path = os.path.join(PATH, "simi.csv")
df = pd.DataFrame(simi_array)
df.to_csv(simi_path, index=False)