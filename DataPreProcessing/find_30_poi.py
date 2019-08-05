#coding:utf-8

'''
data:
30 points id, longitude, latitude : /Users/mac/Desktop/materials/data/simpleposlist.csv
all points id, poi : /Users/mac/Desktop/materials/data/placepoi1129_r2.csv
30 points connect or not : /Users/mac/Desktop/materials/data/Lmatrix.csv
30 points distance : /Users/mac/Desktop/materials/data/L_Distance_Matrix.csv
'''

import os
import sys
import csv

reload(sys)
sys.setdefaultencoding('utf8')

PATH = '../OriginalData'

def count_poi (List1):
    vector = []
    for item in poi_set:
        vector.append(List1.count(item))
    return vector


simpleposlist_path = os.path.join(PATH, "30PointsID.csv")
id_list = []
with open (simpleposlist_path ,'rb') as simpleposlist:
    id_lines = csv.reader(simpleposlist)
    for id in id_lines:
        id_list.append(id[0])


placepoi1129_path = os.path.join(PATH, "1129PointsPOI.csv")
useful_items = []
with open (placepoi1129_path, 'rb') as placepoi1129:
    all_items = csv.reader(placepoi1129)
    for item in all_items:
        if item[0] in id_list:
            useful_items.append(item)

tmp_all_poi = []
for row in useful_items:
    tmp_all_poi.extend(row[1:])
poi_set = set(tmp_all_poi)

poi_30_path = os.path.join(PATH, "poi_30.csv")
with open (poi_30_path, 'w+') as poi_30_file:
    myWriter = csv.writer(poi_30_file)
    myWriter.writerow(['id']+list(poi_set))
    for row in useful_items:
        vector =[]
        vector.append(row[0])
        vector.extend(count_poi(row))
        myWriter.writerow(vector)
        id_list.remove(row[0])

    for left_id in id_list:
        myWriter.writerow([left_id]+[0]*(len(poi_set)))

