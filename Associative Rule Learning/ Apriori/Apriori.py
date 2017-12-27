"""Associative Rule Learning :- Apriori model based on 

    A person who bought X also Bought Y 
    
    
    Here in the dataset we have 7500 records of different people who has bought 
    different things !! 
    
    Now our model helps us to find best combination of food items so accordingly the 
    manager can arrange the items in the shop and earn more !! 
    

"""

"""Refer Image for understanding the comment section"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
print(results)

"""
min_support = percent of people buying particular product divide by total customers 

min_confidence = If a person buys x then he has a chance of min_confidence * 100 percentage 
of buying y 

min_lift = Ratio of people who already have y and recommending x to only those people !

min_length = 2 minimum 2 products in the list in dataset 


FOR EG = Consider the case 1 : 
        support is 0.004 so here .4 percent people buy chicken and light cream together out 
        of 7500
        
        confidence is 29% so here there are 29% chances that if people buy lightcream then 
        there are 29% percent chances that they will buy chicken
        
        Here we are recommending chicken only to the people who already has bought cream !!
        so it is considered as improvement in confidence !
        
        lift is 4.84 which is good enough !! 
        
    
Also the results displayed are according to the importance !!


"""