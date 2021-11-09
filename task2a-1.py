import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# read the csv
world = pd.read_csv("world.csv", encoding = 'ISO-8859-1')
life = pd.read_csv("life.csv", encoding = 'ISO-8859-1')

world = pd.merge(world,life,on='Country Code')
world1 = world.copy()

# preprocess to the form we need
world1 = world1.replace("..", np.NaN)
world1 = world1.reset_index()
world1 = world1.fillna(world1.median())
world1 = world1.drop(columns=['index','Country Name', 'Time', 'Country Code', 'Country', 'Year'])
world1 = pd.DataFrame(world1)
print(world1)
world1.to_csv("useless.csv")
# convert my datas and
data=world1[['Access to electricity (% of population) [EG.ELC.ACCS.ZS]',\
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
classlabel=world1['Life expectancy at birth (years)']
data = data.apply(lambda x: x - x.mean())
data = pd.DataFrame(data)
data = StandardScaler().fit_transform(data)
data = pd.DataFrame(data)
x_train, x_test, y_train, y_test = train_test_split(data, classlabel, train_size = 0.7, test_size= 0.3, random_state = 200)

k3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
k7 = neighbors.KNeighborsClassifier(n_neighbors = 7)
k3.fit(x_train, y_train)
k7.fit(x_train, y_train)
k3pred = k3.predict(x_test)
k7pred = k7.predict(x_test)
k3accur = accuracy_score(y_test, k3pred)
k7accur = accuracy_score(y_test, k7pred)

# decision tree
dt = DecisionTreeClassifier(criterion='entropy', random_state = 200)
dt.fit(x_train, y_train)
dtpred = dt.predict(x_test)
dtaccure = accuracy_score(y_test, dtpred)


# find the median, mean and variance
world2 = data
world2 = pd.DataFrame([round(world2.median(), 3), round(world2.mean(), 3), round(world2.var(), 3)]).transpose()
world2.columns = ['median', 'mean', 'variance']
world2['feature'] = ['Access to electricity (% of population) [EG.ELC.ACCS.ZS]',\
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
    'Wage and salaried workers, female (% of female employment) (modeled ILO estimate) [SL.EMP.WORK.FE.ZS]']
world2.set_index(['feature'], inplace = True)
world2 = pd.DataFrame(world2)
world2.to_csv('task2a.csv', index=False)
# print the accuracy of decision tree and k-nn(k=3 and 7)
print('Accuracy of decision tree:', round(dtaccure, 3))
print('Accuracy of k-nn (k=3):', round(k3accur, 3))
print('Accuracy of k-nn (k=7):', round(k7accur, 3))