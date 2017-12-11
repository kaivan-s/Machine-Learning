#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:22:27 2017

@author: kaivanshah
"""
#Before the program a employee at a new company said that his salary was 160k in the last 
#company so the, HR team of the new company to check whether the employee saying truth or not 
#They took records as in dataset of the previous company of the employee and checked 
#consider for level 6.5 the employee has salary 160k 
#This model will check the truthness of the employee !! 

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

dataset=pd.read_csv('position_Salaries.csv')


X= dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values


#no training and test because our dataset has less content

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,Y)

#for polynomial import preprocessing rather than linear_model
#With the next 3 lines we transformed the matrix X in to more degrees because to increase the accuracy
#But overall the model is linear so another object will fit X_poly and Y
#First linear was just to compare with the polynomial 

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)

regressor2=LinearRegression()
regressor2.fit(X_poly,Y)


#visualising linear regression results 
mp.scatter(X,Y,color="red")
mp.plot(X,regressor.predict(X),color='blue')
mp.title('Truth or Bluff (Linear Regression)')
mp.xlabel('position level')
mp.ylabel('salary')
mp.show()


#visualising poly regression results

mp.scatter(X,Y,color="red")
mp.plot(X,regressor2.predict(poly_reg.fit_transform(X)),color='blue')
mp.title('Truth or Bluff (Polynomial Regression)')
mp.xlabel('position level')
mp.ylabel('salary')
mp.show()

#predicting new result with linear model 
regressor.predict(6.5)

#predicting new result with poly model
regressor2.predict(poly_reg.fit_transform(6.5))

#Hence by the result we see that the employee is saying the truth with a minor gap of 2k dollars
 

