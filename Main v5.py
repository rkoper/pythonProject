import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model

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

data04 = data03.fillna(0)
data05 = data04[["NaturalGas(kBtu)","Electricity(kBtu)","SiteEnergyUse(kBtu)","LargestPropertyUseTypeGFA","TotalGHGEmissions"]]
X = data05.iloc[:,0:4]
y = data05.iloc[:,4]

print("X.shape-----", X.head(3))
print("X.shape------", y.head(3))




std_scale = preprocessing.StandardScaler().fit(X)
X_scale = std_scale.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scale,y , test_size = 0.33)

lr = linear_model.LinearRegression()

lr.fit(X_train, y_train)

# On récupère l'erreur de norme 2 sur le jeu de données test comme baseline
baseline_error = np.mean((lr.predict(X_test) - y_test) ** 2)

print("baseline_error......." , baseline_error)

n_alphas = 10
alphas = np.logspace(-10, 10, n_alphas)
from sklearn.linear_model import Ridge

ridge = linear_model.Ridge()

coefs = []
errors = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)
    errors.append([baseline_error, np.mean((ridge.predict(X_test) - y_test) ** 2)])

ax = plt.gca()
ax.plot(alphas, errors, baseline_error )
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

print(np.argmin(errors))