from utils import read_data, plot, plot_clustered, save_clusters
from Clustering import data_clustering

"""
Use kmeans for K means Clustering
Use single for Single Linkage Hierarchical Agglomerative Clustering
Use complete for Complete Linkage Hierarchical Agglomerative Clustering
Use average for Average Linkage Hierarchical Agglomerative Clustering
Use optics for OPTICS clustering
Use dbscan for DBSCAN clustering
Use spectral for Spectral clustering
Use kmedoid for K Mediod Clustering
Use buckshot for Buckshot Clustering algorithm
Use proposed to run the Proposed clustering algorithm
"""
cls_opt='proposed'

"""Define number of clusters to be generated"""
n_clusters = 2    

"""Choose whether to save cluster predictions in txt file"""
save_output = False

data = read_data()   #read data.csv file
plot(data)    #plot unclustered data
cls = data_clustering(clf_opt=cls_opt, n_clusters=n_clusters)   #define a clustering algorithm
y_pred = cls.clustering(data)    #Run clustering
plot_clustered(data, y_pred)    #plot the clusters
if save_output: save_clusters(y_pred, 'clustered_output.txt')     #Write outputs to txt file