import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
# read the csv
world = pd.read_csv("world.csv", encoding = 'ISO-8859-1')
life = pd.read_csv("life.csv", encoding = 'ISO-8859-1')

world = pd.merge(world,life,on='Country Code')
world1 = world.copy()

# preprocess datas to the form we need
world1 = world1.replace("..", np.NaN)
world1 = world1.reset_index()
world1 = world1.fillna(world1.median())
world1 = world1.drop(columns=['index','Country Name', 'Time', 'Country', 'Year', 'Country Code'])
world1 = pd.DataFrame(world1)
data = world1.copy()
data = data.drop(columns = ['Life expectancy at birth (years)'])
data1=world1[['Access to electricity (% of population) [EG.ELC.ACCS.ZS]',\
    'Adjusted net national income per capita (current US$) [NY.ADJ.NNTY.PC.CD]',\
    'Age dependency ratio (% of working-age population) [SP.POP.DPND]',\
    'Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total) [SH.DTH.COMM.ZS]',\
    'Current health expenditure per capita (current US$) [SH.XPD.CHEX.PC.CD]',\
    'Fertility rate, total (births per woman) [SP.DYN.TFRT.IN]',\
    'Fixed broadband subscriptions (per 100 people) [IT.NET.BBND.P2]',\
    'Fixed telephone subscriptions (per 100 people) [IT.MLT.MAIN.P2]',\
    'GDP per capita (constant 2010 US$) [NY.GDP.PCAP.KD]',\
    'GNI per capita, Atlas method (current US$) [NY.GNP.PCAP.CD]',\
    'Individuals using the Internet (% of population) [IT.NET.USER.ZS]',\
    'Lifetime risk of maternal death (%) [SH.MMR.RISK.ZS]',\
    'People using at least basic drinking water services (% of population) [SH.H2O.BASW.ZS]',\
    'People using at least basic drinking water services, rural (% of rural population) [SH.H2O.BASW.RU.ZS]',\
    'People using at least basic drinking water services, urban (% of urban population) [SH.H2O.BASW.UR.ZS]',\
    'People using at least basic sanitation services, urban (% of urban population) [SH.STA.BASS.UR.ZS]',\
    'Prevalence of anemia among children (% of children under 5) [SH.ANM.CHLD.ZS]',\
    'Secure Internet servers (per 1 million people) [IT.NET.SECR.P6]',\
    'Self-employed, female (% of female employment) (modeled ILO estimate) [SL.EMP.SELF.FE.ZS]',\
    'Wage and salaried workers, female (% of female employment) (modeled ILO estimate) [SL.EMP.WORK.FE.ZS]']].astype(float)
classlabel = world1['Life expectancy at birth (years)']

# create the data of f1 × f2, 210 features
newdata = data1.copy()
namelist = data1.columns.tolist()
for i in range(len(data1.columns)):
    for j in range(i, len(data1.columns)):
        if i == j:
            continue
        newdata[namelist[i] + namelist[j]] = data1.iloc[:, i] * data1.iloc[:, j]

# find best choice for kmean method
KMacc = []
k = range(2, 80)
a = 0
knn3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
for v in k:
    x_train, x_test, y_train, y_test = train_test_split(data, classlabel, train_size=0.7, test_size=0.3, random_state = 180)
    kmean = KMeans(n_clusters=v)
    kmean = kmean.fit(x_train)
    xtrain = x_train.copy()
    xtest = x_test.copy()
    xtrain['classlabe1'] = kmean.labels_
    xtest['classlabe1'] = kmean.predict(xtest)
    knn3.fit(xtrain, y_train)
    score = knn3.score(X = xtest, y = y_test)
    KMacc.append((score, a))
    a += 1
data0 = data.copy()
data0 = pd.DataFrame(data)
kvalue = sorted(KMacc)[-1][-1]
km = KMeans(n_clusters=kvalue)
km = km.fit(data)
data0['classlabe1'] = km.labels_
newdata = pd.DataFrame(newdata).astype(float)
data0 = pd.DataFrame(data0).astype(float)
longdata = pd.merge(newdata, data0)

# from mi find best 4 features
MI = mutual_info_classif(longdata, classlabel, discrete_features = True)
MId = pd.DataFrame(MI)
MId = MId.transpose()
MId.columns = longdata.columns
# find the 4 features and save the columns in dataframe
head = MId.columns.values.tolist()
v = []
for i in MId.iteritems():
    v.append(i[1][0])
h_v = []
x = 0
for x in range(0, len(v)):
    h_v.append((v[x], head[x]))
    x += 1
fourfeature = sorted(h_v)[-4:]
engineer = []
for x in range(4):
    engineer.append(fourfeature[x][1])
    x += 1
engdata = longdata[engineer]
engdata = pd.DataFrame(engdata)

# PCA method to find 4 features
pcadata = longdata.copy()
stand = StandardScaler().fit_transform(pcadata)
pca = PCA(n_components = 4)
special = pca.fit_transform(stand)
standpd = pd.DataFrame(data = special, columns=['pca1', 'pca2', 'pca3', 'pca4'])
standpd = pd.DataFrame(standpd)

# random 4 feature
randomdata = data[['Access to electricity (% of population) [EG.ELC.ACCS.ZS]',\
    'Adjusted net national income per capita (current US$) [NY.ADJ.NNTY.PC.CD]',\
    'Age dependency ratio (% of working-age population) [SP.POP.DPND]',\
    'Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total) [SH.DTH.COMM.ZS]']]
randomdata = pd.DataFrame(randomdata)

# find accuracy for 3 data
x_train1, x_test1, y_train1, y_test1 = train_test_split(engdata, classlabel, train_size = 0.7, test_size= 0.3, random_state = 180)
k1 = neighbors.KNeighborsClassifier(n_neighbors = 3)
k1.fit(x_train1, y_train1)
k1pred = k1.predict(x_test1)
k1accur = accuracy_score(y_test1, k1pred)
print('Accuracy of feature engineering: ', round(k1accur, 3))

x_train2, x_test2, y_train2, y_test2 = train_test_split(standpd, classlabel, train_size = 0.7, test_size= 0.3, random_state = 180)
k2 = neighbors.KNeighborsClassifier(n_neighbors = 3)
k2.fit(x_train2, y_train2)
k2pred = k2.predict(x_test2)
k2accur = accuracy_score(y_test2, k2pred)
print('Accuracy of PCA:', round(k2accur, 3))

x_train3, x_test3, y_train3, y_test3 = train_test_split(randomdata, classlabel, train_size = 0.7, test_size= 0.3, random_state = 180)
k3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
k3.fit(x_train3, y_train3)
k3pred = k3.predict(x_test3)
k3accur = accuracy_score(y_test3, k3pred)
print('Accuracy of first four features:', round(k3accur, 3))