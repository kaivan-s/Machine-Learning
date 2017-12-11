"""Consider the problem from polynomial regression only Employee says that when level is 
6.5 his salary is 160k, we have to check if he is bluffing or not !!!"""

""" Random Forest Regression :-  It is a model for almost the perfect accuracy 
 Random forest divides the trees and take predictions !!! 

For eg:- 
         consider a box of candies randomly kept in the box now around 100 people are asked
to guess the number of candies. Now this is random forest regression here 100 people 
will guess the number of candies and average of that will be taken !! 

Here we are using it for dividing the decision trees but random forest can be used to 
merge differenct models also !!"""


#importing libraries 
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

#providing dataset
dataset=pd.read_csv('position_Salaries.csv')

#Dependent and independent variables 
X= dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Note here the RandomForestRegressor is in sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state=0)
#n_estimators will take the number of guesses like here we are using 300 trees for 
#our model !! Remember more the trees better the guessing !! 
regressor.fit(X,Y)

y_pred=regressor.predict(6.5)

#Grid is for every measurement for eg not only integers but 1.1 1.2 etc on graph for better accuracy on graph !! 
X_grid = np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
mp.scatter(X,Y,color='red')
mp.plot(X_grid,regressor.predict(X_grid),color='blue')
mp.title('Truth or Bluff (Random Forest Regression) ')
mp.xlabel('position level')
mp.ylabel('salaries')
mp.show()

