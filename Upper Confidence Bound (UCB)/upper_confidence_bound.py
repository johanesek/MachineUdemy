import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math

# Importing datasets
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implement UCB
d = 10
N = 10000
ads_selected = []
numbers_of_selections = [0] * d #creates vector of 0 long d
sums_of_rewards = [0] * d
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if(numbers_of_selections[i]>0):
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(n + 1)/numbers_of_selections)
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    sums_of_rewards[ad] = sums_of_rewards[ad] + dataset[n, ad] 