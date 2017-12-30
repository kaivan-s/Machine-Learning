""" Deep Learning :- 
                 Artificial Neural Network !!!  
        "One of the most powerful tools in machine learning"    """
                 
#Tensorflow and theano is for numerical computations which is used for research purposes for devloping new 
#new neural networks !! 
#Use keras which is combination of theano and tensorflow
                 


import numpy as np 
import matplotlib.pyplot as mp 
import pandas as pd 


dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

#Because there are Strings in the dataset so to convert into binary 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_1 = LabelEncoder()
X[:,1] = le_1.fit_transform(X[:,1])
le_2 = LabelEncoder()
X[:,2] = le_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#To avoid dummy variable trap 
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train , X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.2, random_state = 0 )

#Feature Scaling is necessary because for large data it will be easy to compute
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#implementing ArtificialNeuralNetwork
#importing keras library for efficient and fast network 
import keras

#sequential = Initialise neural network
from keras.models import Sequential

#dense = To create the layers like input hidden and output
from keras.layers import Dense

#Initialising the ANN
seq = Sequential()

#Adding different layers 
# 1) Input Layer and First hidden layer ... 
seq.add(Dense(output_dim = 6 , init='uniform' , activation = 'relu', input_dim = 11))

#Adding another hidden layer....
seq.add(Dense(output_dim = 6 ,init = 'uniform' , activation = 'relu' ))

#Output Layer Sigmoid to get rank wise probability..... of who will leave the bank..
seq.add(Dense(output_dim = 1 , activation = 'sigmoid' , init='uniform'))

#Compiling the ANN
#Adam is type stochastic gradient
#loss will be same as logistic Regression.....
seq.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Selecting number of epochs and seeing how the model is improving the efficiency...
#batch_size =10 and nb_epoch = 100
seq.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)

y_pred = seq.predict(X_test)
#if y_pred > 0.5 then 1 and if less then 1 then 0
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)