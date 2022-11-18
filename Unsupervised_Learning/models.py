import numpy as np
import pandas as pd
from sklearn import cluster

"""Compute euclidean distance between two data points"""
def dist(i1,i2):
    return np.sqrt((i1[0]-i2[0])**2+(i1[1]-i2[1])**2)

"""Search for point i in a list"""
def search(i, to_class_0):
    for j in to_class_0:
        if j[0] == i[0] and j[1] == i[1]:
            return False
    return True

"""
Proposed Alogrithm:
-Runs DBSCAN on full dataset
-Saves two most popular classes and reclassifies rest of the datapoints, say C1 and C2
-Each remaining datapoint is classified to the class corresponding to the closest classified point
-To avoid chaining effect, only original C1 and C2 are considered while re-classifying
"""
class proposed():
    def __init__(self,n_clusters):
        self.n_clusters = n_clusters
    
    def fit_predict(self,data):
        y_pred = cluster.DBSCAN(eps=0.06).fit_predict(data)  #Running DBSCAN
        for n,i in enumerate(y_pred):       #putting remaining points to a third class (C3)
            if i!=1 and i!=0:
                y_pred[n] = -1
        y_pred = pd.DataFrame(y_pred)
        y_pred.columns = ['Target']
        full = pd.concat([data,y_pred],axis=1)

        class_0 = full.loc[full['Target'] == 0]
        class_1 = full.loc[full['Target'] == 1]
        class_2 = full.loc[full['Target'] == -1]

        arr0 = class_0.to_numpy()
        arr1 = class_1.to_numpy()
        arr2 = class_2.to_numpy()

        to_class_0 = []
        to_class_1 = []

        #Computing distance and reclassifying remaining datapoints (from C3)
        for i in arr2:
            dist_0 = 10000
            for j in arr0:
                d = dist(i,j)
                if d<dist_0:
                    dist_0 = d
            dist_1 = 10000
            for j in arr1:
                d = dist(i,j)
                if d<dist_0:
                    dist_1 = d
            if dist_0<=dist_1:
                to_class_0.append(i)
            else:
                to_class_1.append(i)

        full = full.to_numpy()

        y_pred2 = []

        #Compiling final prediction list
        for n,i in enumerate(full):
            if i[2] != -1:
                y_pred2.append(int(i[2]))
            else:
                if search(i, to_class_0):
                    y_pred2.append(1)
                else:
                    y_pred2.append(0)
        return y_pred2

"""
Buckshot Algorithm:
-perfrom single linkage agglomerative clustering on sqrt(k*n) points
-compute centroids of the clusters obtained
-use these centroids as initial seeds for K-means on full dataset
"""
class buckshot():
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, data):
        n = data.count()
        num_pts = int(np.sqrt(self.n_clusters*n[0]))
        sampled = data.sample(n=num_pts)
        sampled = sampled.to_numpy()
        #Performing Single Linkage Agglomerative Clustering
        agglo_pred = cluster.AgglomerativeClustering(n_clusters = self.n_clusters, linkage = 'single').fit_predict(sampled)

        class_0 = []
        class_1 = []
        for n,i in enumerate(sampled):
            if agglo_pred[n] == 0:
                class_0.append(i)
            else:
                class_1.append(i)
        #Finding centroids of obtained classes        
        centroids = np.array([[*np.mean(class_0,axis=0)],[*np.mean(class_1,axis=0)]])
        #Performing K-Means with newly obtained centroids as seeds
        return cluster.KMeans(n_clusters=self.n_clusters, init=centroids).fit_predict(data)

