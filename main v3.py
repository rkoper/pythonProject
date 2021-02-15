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
d_15_f1 = FilterOne(d_15, 3340)

df1 = d_15_f1["Location"].str.split(":", n=10, expand=True)
for x in range(8):
    d_15_f1[x] = df1[x]

print(d_15_f1.head(10).to_string())

d_15_f1.drop(0, inplace=True, axis=1)
d_15_f1.drop(3, inplace=True, axis=1)
d_15_f1.drop(6, inplace=True, axis=1)

df1 = d_15_f1[1].str.split(",", n=2, expand=True)
df1_r = df1[0].str.replace("'", '')
d_15_f1['latitude'] = df1_r

df2 = d_15_f1[2].str.split(",", n=2, expand=True)
df2_r = df2[0].str.replace("'", '')
d_15_f1['longitude'] = df2_r

df3 = d_15_f1[4].str.split(",", n=2, expand=True)
df3_r = df3[0].str.replace('"', '')
d_15_f1['address'] = df3_r

df4 = d_15_f1[5].str.split('"', n=2, expand=True)
df4_r = df4[0].str.replace('"', '')
d_15_f1['city'] = df4_r

df5 = d_15_f1[7].str.split("}", n=1, expand=True)
df5_r = df5[0].str.replace('"', '')
d_15_f1['state'] = df5_r

d_15_f1.drop(1, inplace=True, axis=1)
d_15_f1.drop(2, inplace=True, axis=1)
d_15_f1.drop(4, inplace=True, axis=1)
d_15_f1.drop(5, inplace=True, axis=1)
d_15_f1.drop(7, inplace=True, axis=1)
d_15_f1.drop('Location', inplace=True, axis=1)

# New Col LatLng2015
# d_151 = d_15["Location"]
# d_152 = pd.DataFrame(d_151.str.split(",", 30).tolist())
# d_153 = d_152.rename(columns={0: 'latitude',1: 'longitude',2: 'human_address',3: 'city',4: 'state',5: 'zip',})
# print(d_153.head(10).to_string())
