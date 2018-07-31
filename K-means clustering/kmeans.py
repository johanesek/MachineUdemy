import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('mall_customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using the elbow method to find the optimum number of clusters
from sklearn.cluster import KMeans
'''
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
'''

# applying kmeans to the mall dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
ykmeans = kmeans.fit_predict(X)

#Visualising clusters
plt.scatter(X[ykmeans == 0,0], X[ykmeans == 0,1], s=10, c='red', label='Cluster 1' )
plt.scatter(X[ykmeans == 1,0], X[ykmeans == 1,1], s=10, c='blue', label='Cluster 2' )
plt.scatter(X[ykmeans == 2,0], X[ykmeans == 2,1], s=10, c='green', label='Cluster 3' )
plt.scatter(X[ykmeans == 3,0], X[ykmeans == 3,1], s=10, c='cyan', label='Cluster 4' )
plt.scatter(X[ykmeans == 4,0], X[ykmeans == 4,1], s=10, c='magenta', label='Cluster 5' )
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 20, c = 'yellow', label = 'centroids')
plt.legend()
plt.show()