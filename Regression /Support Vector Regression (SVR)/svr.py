#SVR IS ALMOST SAME AS THE POLYNOMIAL REGRESSION (NON-LINEAR) 



import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

#importing datasets
dataset=pd.read_csv('position_Salaries.csv')

#Dividing independent and dependent variables
X= dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values


"""Here we need feature scaling because svr does not do it automatically 
as it is not used much """
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
Y= sc_y.fit_transform(Y)

"""Creating object"""
from sklearn.svm import SVR
regressor=SVR()
regressor.fit(X,Y)



"""Here for prediction as sc_X is in scaled we need to inverse transform to get our 
amount in dollars and also the transform method takes an array as an argument so we need to 
write 6.5 as shown below """
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
             

"""Plotting the graph """                                           
mp.scatter(X,Y,color="red")
mp.plot(X,regressor.predict(X),color='blue')
mp.title('Truth or Bluff (SVR Regression)')
mp.xlabel('position level')
mp.ylabel('salary')
mp.show()
