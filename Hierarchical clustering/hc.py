import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('mall_customers.csv')
X = dataset.iloc[:,[3,4]].values

#Using the dendogram to find the optimum number of clusters
'''
import scipy.cluster.hierarchy as sch 
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()
'''
# Fitting hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualisation of resulting clusters
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s=10, c='red', label='Cluster 1' )
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s=10, c='blue', label='Cluster 2' )
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s=10, c='green', label='Cluster 3' )
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s=10, c='cyan', label='Cluster 4' )
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s=10, c='magenta', label='Cluster 5' )

plt.legend()
plt.show()