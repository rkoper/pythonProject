import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# print(d_15.head(300).to_string())

def FilterOne(d, x):
    d1 = d.isna().sum() / x * 100
    d2 = pd.DataFrame(data=d1)
    d3 = d2.rename(columns={0: 'RatioEmpty'})
    d4 = d3.T
    d = d.append(d4, ignore_index=True)
    d5 = d.T
    d6 = d5.rename(columns={x: 'EmptyRatio'})
    d7 = d6.drop(d6[d6.EmptyRatio > 50].index)
    d8 = d7.T
    d9 = d8[:-1]
    return d9


d_15 = pd.read_csv(r'/Users/rogerrabbit/2015.csv')
d_16 = pd.read_csv(r'/Users/rogerrabbit/2016.csv')
d_15_f1 = FilterOne(d_15, 3340)
d_16_f1 = FilterOne(d_16, 3376)

# print("before 15", d_15.columns)
# print("before 16", d_16.columns)
# print("After 15", d_15_f1.shape)
# print("After 16", d_16_f1.shape)
# print("After 16", d_16_f1.shape)


# New Col LatLng2016
d_16_1 = d_16_f1['Latitude']
d_16_2 = pd.DataFrame(d_16_1)
d_16_21 = d_16_2.astype(float)

d_16_22 = np.round(d_16_21, 2)

d_16_3 = d_16_22.astype(str)


d_16_4 = d_16_f1['Longitude']
d_16_5 = pd.DataFrame(d_16_4)
d_16_51 = d_16_5.astype(float)

d_16_52 = np.round(d_16_51, 2)

d_16_6 = d_16_52.astype(str)


d_16_L = pd.merge(d_16_3, d_16_6, left_index=True, right_index=True)
d_16_L1 = d_16_L['Latitude'] + "," + d_16_L['Longitude']
d_16_L2 = pd.DataFrame(d_16_L1)
d_16_L3 = d_16_L2.rename(columns={0: 'LatLng2016'})
d_16_n = pd.merge(d_16_L3, d_16_f1, left_index=True, right_index=True)
# print(d_16_n.head(3).to_string())


# New Col LatLng2015
d_151 = d_15["Location"]
d_152 = pd.DataFrame(d_151.str.split(",", 15).tolist())
d_153 = d_152.rename(columns={0: 'lat', 1: 'lng'})
d_15_lat = d_153['lat']
d_15_lng = d_153['lng']
d_15_lat_1 = pd.DataFrame(d_15_lat.str.split("'", 15).tolist())
d_15_lat_2 = d_15_lat_1.rename(columns={3: 'lat'})
d_15_lat_v1 = d_15_lat_2['lat']
d_15_lat_v11 = pd.DataFrame(d_15_lat_v1)

d_15_lat_v12 = d_15_lat_v11.astype(float)

d_15_lat_v13 = np.round(d_15_lat_v12, 2)

d_15_lat_12 = d_15_lat_v13.astype(str)
d_15_lat_13 = d_15_lat_12.replace("'", "")
d_15_lng_1 = pd.DataFrame(d_15_lng.str.split("'", 15).tolist())
d_15_lng_2 = d_15_lng_1.rename(columns={3: 'lng'})
d_15_lng_v1 = d_15_lng_2['lng']
d_15_lng_v11 = pd.DataFrame(d_15_lng_v1)

d_15_lng_v12 = d_15_lng_v11.astype(float)

d_15_lng_v13 = np.round(d_15_lng_v12, 2)

d_15_lng_12 = d_15_lng_v13.astype(str)
d_15_lng_13 = d_15_lng_12.replace("'", "")
d_15_L = pd.merge(d_15_lat_13, d_15_lng_13, left_index=True, right_index=True)


print(d_15_L.head(10).to_string())
# d_15_5 = pd.DataFrame(d_15_4)
# d_15_6 = d_15_5.astype(str)
# print(d_15_6.head(3).to_string())
# data = d_15_f1.append(d_16_f1, sort=True)
# print("data ", data.shape)
# print(data.head(10).to_string())
# d_01 = data.sort_values(by=['PropertyName'], ascending=False)
# print(d_01.head(10).to_string())
