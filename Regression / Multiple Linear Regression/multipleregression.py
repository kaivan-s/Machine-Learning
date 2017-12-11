#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:30:09 2017

@author: kaivanshah
"""

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

dataset=pd.read_csv('Startups.csv')
X= dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#Because cities need to be converted into Numbers
#See results individually for LabelEncoder and OneHotEncoder to see the difference 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)

#Only to increase the efficiency
#Building optimal model by using backward elimination and removing not necessary variables
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,2,3,4]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()







