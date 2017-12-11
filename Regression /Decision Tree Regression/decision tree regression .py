# Decision tree will divide the observations or the data into the intervals and all the points in that interval will have the
# same value !!!



import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

dataset=pd.read_csv('position_Salaries.csv')


X= dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()
regressor.fit(X,Y)

y_pred=regressor.predict(6.5) 

mp.scatter(X,Y,color='red')
mp.plot(X,regressor.predict(X),color='blue')
mp.title("Truth or bluff (Decision Tree Regression)")
mp.xlabel('Position')
mp.ylabel('Salary')
mp.show()

"""In decision tree we take average for each interval 
   So the value of each and every point in that interval must be equal 
   but as from the above graph it varies so for each and every point to be equal 
   we need to take measurement for every point given by:-  """
   
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))   

""" arange from numpy takes min and max value and difference we want 
between each values so here min value of X and max value of X and an interval of 0.01 
reshape function will reform the X_grid into points of interval 0.01
 """

mp.scatter(X,Y,color='red')
mp.plot(X_grid,regressor.predict(X_grid),color='blue')
mp.title("Truth or bluff (Decision Tree Regression)")
mp.xlabel('Position')
mp.ylabel('Salary')
mp.show()
