"""Decision Tree classification :- 
                                    Same algorithm as decision tree regression !
Main difference between Regression and classification is that in regression we predict
a number salary or weather, While in classification we just classify between male female 
yes no etc !!

Like here we need to predict whether the person will buy the SUV or not From the dataset
same as logistic regression !!! 

For more knowledge in Decision tree visit regression folder and decision tree as here 
the algorithm is same !!

Dataset is same as Logistic Regression so problem statement You can get it in 
Classification folder logistic regression !! 
 
"""




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

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0,criterion='entropy')
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


