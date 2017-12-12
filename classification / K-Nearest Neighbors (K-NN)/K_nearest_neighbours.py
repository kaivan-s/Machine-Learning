"""K_Nearest_Neighbors 

Consider the same problem from the Logistic regression !!

Here by the model we will predict whether the employee will buy the SUV or not 
by considering the distance of the neighbors 

Lets say there are 2 grps called red and blue 

red has 7 observations so does blue !!

Now consider any point anywhere called gray, now the model sees where the gray is and
by default takes 5 nearest points from gray to nearest points in red and blue.

Further, consider Most nearest = Any point in blue 
second nearest = Any point in red 
third nearest = Any point in blue 
fourth nearest = Any point in blue
fifth nearest = Any point in red 

From the above data we see that total 3 points from blue and 2 points from red are nearer 
gray, Remember dont consider the order but consider total number of points in each group !

So here blue =3 and red =2 so the gray will be classified as blue !!!!

Here more accuracy is obtained as it is not linear so there will be rough curves in graph
rather than single straight line as in Logistic Regression 

In confusion matrix as I have written if u write the same then primary diagonal 
will have correct predictions !!!! 

Here you will get approx 64+29 = 93

Dataset same as Logistic Regression !!

Remember the classes and libraries used !!!                 """

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


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)

