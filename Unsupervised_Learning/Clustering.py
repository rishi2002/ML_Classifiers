from sklearn import cluster
from sklearn_extra.cluster import KMedoids
from models import proposed, buckshot

class data_clustering():
    def __init__(self,clf_opt='kmeans',n_clusters=2):
        self.clf_opt=clf_opt
        self.n_clusters=n_clusters

    """Choose a clustering algorithm"""
    def clustering_pipeline(self):
    
        if self.clf_opt=='kmeans':
            print('\n\t Performing K Means clustering \n')
            return cluster.KMeans(n_clusters=self.n_clusters)
            
        elif self.clf_opt == 'single':
            print('\n\t Performing Single Linkage Hierarchical Agglomerative Clustering \n')
            return cluster.AgglomerativeClustering(n_clusters = self.n_clusters, affinity = 'euclidean', linkage = 'single')

        elif self.clf_opt == 'complete':
            print('\n\t Performing Complete Linkage Hierarchical Agglomerative Clustering \n')
            return cluster.AgglomerativeClustering(n_clusters = self.n_clusters, affinity = 'euclidean', linkage = 'complete')
             
        elif self.clf_opt == 'average':
            print('\n\t Performing Global Average Linkage Hierarchical Agglomerative Clustering \n')
            return cluster.AgglomerativeClustering(n_clusters = self.n_clusters, affinity = 'euclidean', linkage = 'average')
            
        elif self.clf_opt == 'dbscan':
            print('\n\t Performing DBSCAN \n')
            return cluster.DBSCAN(eps=0.06)

        elif self.clf_opt == 'spectral':
            print('\n\t Performing Spectral Clustering \n')
            return cluster.SpectralClustering(n_clusters=self.n_clusters)

        elif self.clf_opt == 'optics':
            print('\n\t Performing OPTICS Clustering \n')
            return cluster.OPTICS(min_samples=7, xi=0.1, min_cluster_size=0.2)

        elif self.clf_opt == 'kmedoid':
            print('\n\t Performing K Medoid Clustering \n')
            return KMedoids(n_clusters=2)

        elif self.clf_opt == 'proposed':
            print('\n\t Performing proposed clustering method\n')
            return proposed(n_clusters = 2)

        elif self.clf_opt == 'buckshot':
            print('\n\t Running Buckshot Clustering Algorithm\n')
            return buckshot(n_clusters = 2)

        else:
            print("Enter valid clf_opt \n default: kmeans")
            return None

    """Fit chosen algortihm to given data"""
    def clustering(self,data):
        clf = self.clustering_pipeline()
        y_pred = clf.fit_predict(data)
        return y_pred