"""KMeans Clustering :- To find a group of customers who can be called as a target of the 
shop!!!"""

#Importing data set
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')

X=dataset.iloc[:,[3,4]].values


#Plotting the elbow curve to find the number of clusters 
from sklearn.cluster import KMeans
a=[]

#Consider 10 clusters to be max 
for i in range(1,11):
    #creating objects init=K-means++ to avoid random initialisation trap 
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    #Storing every value obtained from kmeans to a and joining them 
    a.append(kmeans.inertia_)
#Plotting the curve 
mp.plot(range(1,11),a)
mp.title('The Elbow View')
mp.xlabel('Clusters')
mp.ylabel('wcss')
mp.show()

#As from the graph we can see that elbow starts to form at 5 Clusters
#Now forming the graph with 5 clusters 

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_means=kmeans.fit_predict(X)

#Here we need 5 clusters as follows :-
mp.scatter(X[y_means==0,0],X[y_means==0,1],s=100,color='red',label='Careful')
mp.scatter(X[y_means==1,0],X[y_means==1,1],s=100,color='blue',label='Standard')
mp.scatter(X[y_means==2,0],X[y_means==2,1],s=100,color='cyan',label='Target 1')
mp.scatter(X[y_means==3,0],X[y_means==3,1],s=100,color='magenta',label='Target 2')
mp.scatter(X[y_means==4,0],X[y_means==4,1],s=100,color='green',label='Ignore')
mp.title("KMeans Clustering")
mp.xlabel('Annual Income K$')
mp.ylabel('Spending Score')
mp.legend()
mp.show()

"""From the graph for 
1) red:- spends low but income high so ignorable
2) blue:- Earns average spends average so average 
3) cyan:- earns high spends high so they are the target customers 
4) magenta:- earns less but spends more they can also be called as target 
5) green:- spends less and earns less so ignorable 

So here from the data we can predict that profit of the shop will be high if they 
sell more what the target customers buy or are interested in it because they are spending 
more in the shop !!!    


"""