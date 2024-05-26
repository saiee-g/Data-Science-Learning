# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:23:04 2024

@author: snowfox
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

data=pd.DataFrame(housing.data)
data.columns=housing.feature_names
print(data.head())

data['PRICE']=housing.target
print(data.isnull().sum())

x=data.drop(['PRICE'], axis=1)
y=data['PRICE']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
model=lm.fit(xtrain, ytrain)

ytrain_pred=lm.predict(xtrain)
ytest_pred=lm.predict(xtest)

df=pd.DataFrame(ytrain_pred, ytrain)
df=pd.DataFrame(ytest_pred,ytest)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(ytest, ytest_pred)
print(mse)
mse=mean_squared_error(ytrain_pred, ytrain)
print(mse)

plt.scatter(ytrain, ytrain_pred, c='blue', marker='o', label='Training data')
plt.scatter(ytest, ytest_pred, c='lightgreen', marker='s', label='Test data')
plt.xlabel('True values')
plt.ylabel('Predicted')
plt.title('True value vs Predicted value')
plt.legend(loc='upper left')
plt.plot()
plt.show()


