"""Reinforcement Learning :- 
                    From the details of t predicting the results of t+1 
                    
UpperConfidenceBound = UCB, here we have datasets of 10k customers who viewed the ad
    where it is 1 now our algortihm will predict that which ad is most suitable for 
    earning more profit Algorithm is as follows !!! 
    
    From the result of the histogram we came to know that 5th ad was to be published 
    as index starts from 0 """

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
import math as m 

dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

N=10000
d=10
number_of_selections = [0] * d
sum_of_rewards = [0] * d
ad_selected = []
total_reward = 0

for j in range(0,10000):
    max_upperbound = 0
    ad =0 
    
    for i in range(0,d):
        if(number_of_selections[i] > 0):
            avg_rew = sum_of_rewards[i]/number_of_selections[i]
            delta_i = m.sqrt(3/2 * m.log(j+1)/number_of_selections[i])
            upper_bound = avg_rew + delta_i
        else:
            upper_bound = 1e400 #random high value 
        
        if upper_bound > max_upperbound:
            max_upperbound = upper_bound
            ad = i 
    ad_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[j,ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
mp.hist(ad_selected)    
mp.title("For most profit")
mp.xlabel("Ads No")
mp.ylabel("No of customers")
mp.show()