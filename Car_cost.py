# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:03:50 2020

@author: LENOVO
"""

#HACKATHON BY IMARTICUS LEARNING

#libraries
import pandas as pd
import numpy as np

#read train and test sets
#combining datasets
train=pd.read_csv('Train_used_cars.csv')
test=pd.read_csv('Test_used_cars.csv')
train['source']='train'
test['source']='test'
data=pd.concat([train,test],ignore_index=True,sort=True)

#divide Name into Model and Brand 
data['Brand']=data['Name'].apply(lambda x:x.split()[0])
Name=list(data.Name)
Model=[]
for i in range(len(Name)):
    Model.append(" ".join(Name[i].split(" ")[1:]).strip())
data['Model']=Model

data.dtypes

categorical_columns=[x for x in data.dtypes.index if data.dtypes[x]=='object']
#exclude mixdataypes columns and source coumn
categorical_columns=[x for x in categorical_columns if x not in['Name','source']]
#print frequencies of categorical_columns
for col in categorical_columns:
    print("Frequencies of categories for variable %s" %col )
    print(data[col].value_counts())

#remove units from engine, mileage and power
data.Engine=data.Engine.str.split(expand=True)
data['Engine']=data['Engine'].astype(float)
data.Mileage=data.Mileage.str.split(expand=True)
data['Mileage']=data['Mileage'].astype(float)
data.New_Price=data.New_Price.str.split(expand=True)
data['New_Price']=data['New_Price'].astype(float)
data.Power=data.Power.str.split(expand=True)
data['Power']=pd.to_numeric(data['Power'],errors='coerce')

data.apply(lambda x:sum(x.isnull()))

#make new column for how old is the used car
data['no of years']=2019-data['Year']

#drop unecessary columns
data.drop(['Name','Year'],axis=1,inplace=True)

data['Fuel_Type'].value_counts()

dl=data.loc[data['Fuel_Type']=='Diesel']
pl=data.loc[data['Fuel_Type']=='Petrol']
cng=data.loc[data['Fuel_Type']=='CNG']
lpg=data.loc[data['Fuel_Type']=='LPG']
El=data.loc[data['Fuel_Type']=='Electric']

dl.apply(lambda x:sum(x.isnull()))
dl['Engine'].fillna((data['Engine'].mean()),inplace=True)
dl['Power'].fillna((dl['Power'].mean()),inplace=True)
dl['Seats'].fillna(int(dl['Seats'].mean()),inplace=True)
dl['New_Price'].fillna((data['New_Price'].mean()),inplace=True)


pl.apply(lambda x:sum(x.isnull()))
pl['Engine'].fillna((pl['Engine'].mean()),inplace=True)
pl['Power'].fillna((pl['Power'].mean()),inplace=True)
pl['Seats'].fillna(int(pl['Seats'].mean()),inplace=True)
pl['New_Price'].fillna((pl['New_Price'].mean()),inplace=True)

El.apply(lambda x:sum(x.isnull()))
El['Mileage'].value_counts() 
#here mileage column is empty so fill it after joining everything to data 
El['New_Price'].fillna((El['New_Price'].mean()),inplace=True)

cng.apply(lambda x:sum(x.isnull()))
cng['New_Price'].value_counts()
cng['New_Price'].fillna((cng['New_Price'].mean()),inplace=True)
cng['Power'].fillna((cng['Power'].mean()),inplace=True)


lpg.apply(lambda x:sum(x.isnull()))
#new price column is empty so fill it after joining everything to data 


data.loc[data.Fuel_Type.isin(dl.Fuel_Type),['Engine','Power','Seats','New_Price']]=dl[['Engine','Power','Seats','New_Price']]
data.loc[data.Fuel_Type.isin(pl.Fuel_Type),['Engine','Power','Seats','New_Price']]=pl[['Engine','Power','Seats','New_Price']]
data.loc[data.Fuel_Type.isin(El.Fuel_Type),['New_Price']]=El[['New_Price']]
data.loc[data.Fuel_Type.isin(cng.Fuel_Type),['New_Price','Power']]=cng[['New_Price','Power']]

#fill missing values of mileage column
data['Mileage'].fillna((data['Mileage'].mean()),inplace=True)
#fill missing values of new price column
data['New_Price'].fillna((data['New_Price'].mean()),inplace=True)


describe=data.describe()
data['Mileage'].replace(min(data['Mileage']),(data['Mileage'].mean()),inplace=True)
data['Seats'].replace(min(data['Seats']),(data['Seats'].mean()),inplace=True)
data['no of years'].replace(min(data['no of years']),(data['no of years'].mean()),inplace=True)
data['New_Price'].replace(min(data['New_Price']),(data['New_Price'].mean()),inplace=True)


#data transformation
data['Engine']=np.log(data['Engine'])
data['Kilometers_Driven']=np.log(data['Kilometers_Driven'])
data['Mileage']=np.log(data['Mileage'])
data['New_Price']=np.log(data['New_Price'])
data['Power']=np.log(data['Power'])

data['no of years'].value_counts()
data.dtypes

#label encoder
from sklearn.preprocessing import LabelEncoder
data.dtypes
le=LabelEncoder()
categorical_columns=['Brand','Location','Fuel_Type','Transmission','Owner_Type']
for i in categorical_columns:
    data[i] = le.fit_transform(data[i])
data.dtypes

data=pd.get_dummies(data,columns=['Brand','Location','Fuel_Type','Transmission','Owner_Type'])

#remove dummy variable trap
del data['Brand_0']
del data['Location_0']
del data['Fuel_Type_0']
del data['Transmission_0']
del data['Owner_Type_0']



#split dataset into train and test set
train=data.loc[data['source']=='train']
test=data.loc[data['source']=='test']

x=train.copy()
y=train.iloc[:,5:6]

x.drop(['Price','source','Model'],axis=1,inplace=True)
test.drop(['Price','source','Model'],axis=1,inplace=True)


#split data into train and valid set
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(x,y,test_size=0.2,random_state=0)


#Initializing MULTIPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#Fitting the regressor with training data
lr.fit(X_train,y_train)
y_pred=lr.predict(X_valid)

#Calculating Score
def rmsle(y, y0):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p((y0** 2) ** 0.5), 2)))

error = rmsle(y_valid, y_pred)
score = 1- error
score

y_test=lr.predict(test)
y_test = pd.DataFrame(y_test)
y_test.rename(columns={ 0: 'Price'}, inplace=True)
y_test.to_excel("oldcars2.xlsx", index = False)
#score=0.5743
#Public score = 0.7984

#Initializing SUPPORT VECTOR REGRESSION
from sklearn.svm import SVR
regressor=SVR(kernel='poly',gamma=0.03)
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_valid)
y_pred=pd.DataFrame(y_pred)

def rmsle(y, y0):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p((y0** 2) ** 0.5), 2)))

error = rmsle(y_valid.values, y_pred.values)
score = 1- error
score

y_test=regressor.predict(test)
y_test = pd.DataFrame(y_test)
y_test.rename(columns={ 0: 'Price'}, inplace=True)
y_test.to_excel("oldcars_svr_poly.xlsx", index = False)
#score 0.7466 (with rbf and gamma=0.07)
#score 0.7469 (with rbf and gamma=0.08)
#score 0.7812  (with poly and gamma=0.02)
#score 0.79718 (with poly and gamma=0.03)

#fitting DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_valid)
y_pred=pd.DataFrame(y_pred)

def rmsle(y, y0):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p((y0** 2) ** 0.5), 2)))

error = rmsle(y_valid.values, y_pred.values)
score = 1- error
score
# 0.7166
y_test=regressor.predict(test)
y_test = pd.DataFrame(y_test)
y_test.rename(columns={ 0: 'Price'}, inplace=True)
y_test.to_excel("oldcars_dtr.xlsx", index = False)


#fitting RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 300)
rf.fit(X_train,y_train) 
y_pred = rf.predict(X_valid)
y_pred=pd.DataFrame(y_pred)

#Calculating Score
def rmsle(y, y0):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p((y0** 2) ** 0.5), 2)))

error = rmsle(y_valid.values, y_pred.values)
score = 1- error
score

y_test=rf.predict(test)
y_test = pd.DataFrame(y_test)
y_test.rename(columns={ 0: 'Price'}, inplace=True)
y_test.to_excel("oldcars_rf1.xlsx", index = False)
#Score 0.8238 (200 estimators)
#Score 0.8243 (300 estimators)  #public Score 0.9236