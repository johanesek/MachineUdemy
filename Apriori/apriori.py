# Apriory

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing datasets
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)]) # Converting dataset dataframe into list of lists

# Training Apriori on the dataset
from apyori import apriori

# min_support = 3*7/7500 (sold at leat 3 times a day)
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Visualising the results
results = list(rules)
print(results)