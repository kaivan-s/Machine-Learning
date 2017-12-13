""" Logistic Regression """

"""
PROBLEM :- Here the dataset given contains userid, age, sal, gender and the last 
coloumn contains the true or false value of the employee whether he/she bought a new car 
released in the market or not.
1=true
0=false

Description : What we have to do is to create a model that will predict the number of employees
which will buy the car by dividing our dataset into training and test sets !!

Feature scaling is required here because age and salary cannot be compared so they need
to be scaled 

Here sklearn.linear_model will be used because Logistic regression is a linear model 
classifier.

Confusion Matrix :- 
                    It is used to compare two observations one the actual one and 
other the predicted one, to check the accuracy of our model 

Remember the classes which need to be imported !!!      """

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,4].values


from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)



from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

#To check how many predictions are correct 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)

"""Graph will be added soon !! """
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
mp.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('white', 'black')))
mp.xlim(X1.min(), X1.max())
mp.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    mp.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
               c = ListedColormap(('red', 'green'))(i), label = j)
mp.title('Logistic Regression (Test set)')
mp.xlabel('Age')
mp.ylabel('Estimated Salary')
mp.legend()
mp.show()



