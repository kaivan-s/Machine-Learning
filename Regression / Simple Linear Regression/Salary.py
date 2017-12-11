#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:04:20 2017

@author: kaivanshah
"""

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

#importing datasets
dataset=pd.read_csv('Salary.csv')
X= dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#Fitting regressor into the model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#To show the predicted values from our regression model and then compare to original values
Y_pred=regressor.predict(X_test)


#Not necessary only for graphs 
#Now visualising by plotting 
#1) FOR TRAINING SET
mp.scatter(X_train,Y_train,color='red')
mp.plot(X_train,regressor.predict(X_train))
mp.title('Salary vs Experience (TRAINING SET)')
mp.xlabel('Experience')
mp.ylabel('Salary')
mp.show()

#2) FOR TEST SET
mp.scatter(X_test,Y_test,color='red')
mp.plot(X_train,regressor.predict(X_train))
mp.title('Salary vs Experience (TEST SET)')
mp.xlabel('Experience')
mp.ylabel('Salary')
mp.show()
















