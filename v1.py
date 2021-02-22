#basic libraries
import sys
import numpy as np
import pandas as pd
import os
import warnings

#seed the project
np.random.seed(64)

#ploting libraries
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.gridspec as gridspec
plt.rcParams['figure.figsize'] = (16,8)
from scipy import stats
from scipy.stats import skew
from scipy.stats.stats import pearsonr
import seaborn as sns
sns.set(context='notebook', style='whitegrid', palette='pastel', font='sans-serif', font_scale=1, color_codes=False, rc=None)

#warning hadle
warnings.filterwarnings("ignore")

print("Set up completed")

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


def mEmptyCell(data):
    all_empty_cell = round((sum(data.isnull().sum())) * 100 / (len(data) * len(data.columns)))
    return all_empty_cell


d_15 = pd.read_csv(r'/Users/rogerrabbit/2015.csv')
d_16 = pd.read_csv(r'/Users/rogerrabbit/2016.csv')

d_15_f1 = FilterOne(d_15, 3340)
d_16_f1 = FilterOne(d_16, 3376)

df1 = d_15_f1["Location"].str.split(":", n=10, expand=True)
for x in range(8):
    d_15_f1[x] = df1[x]

df1 = d_15_f1[1].str.split(",", n=2, expand=True)
df1_r = df1[0].str.replace("'", '')
d_15_f1['Latitude'] = df1_r

df2 = d_15_f1[2].str.split(",", n=2, expand=True)
df2_r = df2[0].str.replace("'", '')
d_15_f1['Longitude'] = df2_r

df3 = d_15_f1[4].str.split(",", n=2, expand=True)
df3_r = df3[0].str.replace('"', '')
d_15_f1['Address'] = df3_r

df4 = d_15_f1[5].str.split(',', n=1, expand=True)
df4_r = df4[0].str.replace('"', '')
d_15_f1['City'] = df4_r

df5 = d_15_f1[6].str.split(",", n=1, expand=True)
df5_r = df5[0].str.replace('"', '')
d_15_f1['State'] = df5_r

df6 = d_15_f1[7].str.split("}", n=1, expand=True)
df6_r = df6[0].str.replace('"', '')
d_15_f1['ZipCode'] = df6_r

d_15_f1.drop(1, inplace=True, axis=1)
d_15_f1.drop(2, inplace=True, axis=1)
d_15_f1.drop(3, inplace=True, axis=1)
d_15_f1.drop(4, inplace=True, axis=1)
d_15_f1.drop(5, inplace=True, axis=1)
d_15_f1.drop(6, inplace=True, axis=1)
d_15_f1.drop(7, inplace=True, axis=1)
d_15_f1.drop('Location', inplace=True, axis=1)

d1 = d_15_f1["OSEBuildingID"]
d2 = d_16_f1["OSEBuildingID"]
d11 = pd.DataFrame(d1)
d21 = pd.DataFrame(d2)
d_2015 = d11.merge(d21, on='OSEBuildingID', how='outer', suffixes=['', '_'], indicator=True)
d_2015_1 = d_2015.loc[d_2015['_merge'] != 'both']
d_2015_2 = d_2015_1.loc[d_2015['_merge'] != 'right_only']

d_2015_3 = pd.merge(d_15_f1, d_2015_2, on=["OSEBuildingID", "OSEBuildingID"])

data = d_2015_3.append(d_16_f1)
data01 = data.sort_values(by=['OSEBuildingID'], ascending=True)
data02 = data01.reset_index()
data02.drop('index', inplace=True, axis=1)
data02.drop('_merge', inplace=True, axis=1)
data02.drop(0, inplace=True, axis=1)
data02.replace(0, np.nan, inplace=True)
data03 = FilterOne(data02, 3432)

data03_ = data03["ListOfAllPropertyUseTypes"].str.split(",", n=1, expand=True)
data03['ListOfAllPropertyUseTypes'] = data03_[0]
data03.drop('ListOfAllPropertyUseTypes', inplace=True, axis=1)
data03.drop('LargestPropertyUseType', inplace=True, axis=1)

data03.drop('OSEBuildingID', inplace=True, axis=1)
data03.drop('DataYear', inplace=True, axis=1)
data03.drop('BuildingType', inplace=True, axis=1)
data03.drop('PropertyName', inplace=True, axis=1)
data03.drop('TaxParcelIdentificationNumber', inplace=True, axis=1)
data03.drop('Neighborhood', inplace=True, axis=1)
data03.drop('NumberofBuildings', inplace=True, axis=1)
data03.drop('City', inplace=True, axis=1)
data03.drop('Address', inplace=True, axis=1)
data03.drop('State', inplace=True, axis=1)
data03.drop('PrimaryPropertyType', inplace=True, axis=1)
data03.drop('GHGEmissionsIntensity', inplace=True, axis=1)
data03.drop('ComplianceStatus', inplace=True, axis=1)

data_04 = data03.dropna()
data_05 = pd.DataFrame(data_04)
corr = data_05.astype('float64').corr()
'''
plt.figure(figsize=(15, 10 ))
sns.heatmap(corr, annot = True)
plt.title("XXXXXXXXXX",fontsize=15)
plt.show()
'''


def get_corelated_col(cor_dat, threshold):
    feature = []
    value = []

    for i, index in enumerate(cor_dat.index):
        if abs(cor_dat[index]) > threshold:
            feature.append(index)
            value.append(cor_dat[index])

    df = pd.DataFrame(data=value, index=feature, columns=['corr value'])
    return df


top_corelated_values = get_corelated_col(corr['TotalGHGEmissions'], 0.40)

final_num_df = data_05[top_corelated_values.index]

train_df = final_num_df.iloc[:1000, ]
test_df = final_num_df.iloc[1000:,:-1 ]

x = train_df['TotalGHGEmissions']


print("Skewness: %f" % train_df['TotalGHGEmissions'].skew())
print("Kurtosis: %f" % train_df['TotalGHGEmissions'].kurt())

train_df.head(3)

data_04 = train_df.dropna()
data_05 = pd.DataFrame(data_04)
corr = data_05.astype('float64').corr()
data_06 = corr.sort_values(by=['TotalGHGEmissions'], ascending=False)
data_07 = data_06['TotalGHGEmissions']
data_08 = data_07.drop(labels=["TotalGHGEmissions"])
data_08



figure, ax = plt.subplots(1,3, figsize = (20,8))
sns.stripplot(data=train_df, x = 'Electricity(kWh)', y='TotalGHGEmissions', ax = ax[0])
plt.show()