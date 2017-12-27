"""Hierarchial Clustering : 
Same problem as in K_means : 
    Difference is that in k_means we used elbow method to determine 
number of clusters here we are using dendrograms method to determine 
number of clusters !!!

Dendrograms :- 
                It is from the scipy library It will plot a graph from 
the bottom up approach as the clusters combine according to the Euclidean
distance !! 

Now in the Graph the longest vertical which is not cut by any horizontal line
lets say that is X. 
Now consider X and draw a horizontal line from beginning to the end and in
between the path count how many horizontal lines are being cut that is the 
number of clusters we need for the model 

Remember Dendrograms work on the principle of Euclidean Distance !!!!!  """

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')

X=dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
mp.title("Dendrograms")
mp.xlabel("customers")
mp.ylabel("Euclidean Distance")
mp.show()

"Agglomerative Clustering uses bottom-up approach"
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)


mp.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,color='red',label='Careful')
mp.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,color='blue',label='Standard')
mp.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,color='cyan',label='Target 1')
mp.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,color='magenta',label='Target 2')
mp.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,color='green',label='Ignore')
mp.title("Hierarchial Clustering")
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
more in the shop !!!  """
