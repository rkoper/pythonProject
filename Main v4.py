import time
import matplotlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize': (15, 15)})
for i in tqdm(range(5)):
    time.sleep(1)

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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

df_train = final_num_df.iloc[:1000,]
df_test = final_num_df.iloc[1000:,]

X = df_train.drop(['TotalGHGEmissions'], axis=1)
y = df_train['TotalGHGEmissions']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y)


print("X_train",X_train)
print("X_test",X_test)
print("y_train",y_train)
print("y_test",y_test)

('\n'
 'from sklearn import metrics\n'
 'from sklearn.linear_model import RidgeCV\n'
 'ridge = RidgeCV(cv = 50)\n'
 'ridge.fit(X_train,y_train)\n'
 'y_pred_ = ridge.predict(X_test)\n'
 'pred_df = pd.DataFrame({\'Ridge-Actual\': y_test, \'Ridge-Predicted\': y_pred_})\n'
 'print(pred_df.head())\n'
 'print(\'Ridge - Mean Absolute Error:\', metrics.mean_absolute_error(y_test, y_pred_))\n'
 'print(\'Ridge - Mean Squared Error:\', metrics.mean_squared_error(y_test, y_pred_))\n'
 'print(\'Ridge - Root Mean Squared Error:\', np.sqrt(metrics.mean_squared_error(y_test, y_pred_)))\n'
 'print(\'Ridge - R2 Value:\', metrics.r2_score(y_test, y_pred_))\n'
 '\n'
 '\n'
 'from sklearn.linear_model import LassoCV\n'
 'from sklearn import metrics\n'
 '\n'
 '\n'
 'lasso = LassoCV(cv = 5)\n'
 'lasso.fit(X_train,y_train)\n'
 'y_pred = lasso.predict(X_test)\n'
 'pred_df = pd.DataFrame({\'Lasso - Actual\': y_test, \'Lasso - Predicted\': y_pred})\n'
 'print(pred_df.head())\n'
 '\n'
 'print(\'Lasso - Mean Absolute Error:\', metrics.mean_absolute_error(y_test, y_pred))\n'
 'print(\'Lasso - Mean Squared Error:\', metrics.mean_squared_error(y_test, y_pred))\n'
 'print(\'Lasso - Root Mean Squared Error:\', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))\n'
 'print(\'Lasso - R2 Value:\', metrics.r2_score(y_test, y_pred))\n')