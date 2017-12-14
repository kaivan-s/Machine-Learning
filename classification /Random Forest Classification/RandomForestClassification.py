#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random Forest Classification : 
                                Predictions of many trees are taken and average of that
is taken  !!! 

Works as a group of Decision Trees!!

No of trees we have to provide in the argument !!  """



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

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0,n_estimators=300)
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

#White=red 
#Black=green 

#As for proper visualisation use the following colours
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
mp.title('Decision Tree(Test set)')
mp.xlabel('Age')
mp.ylabel('Estimated Salary')
mp.legend()
mp.show()


