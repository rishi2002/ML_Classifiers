# ML_Classifiers
I have built some simple ML classifiers using the SKLearn library.

## Supervised Learning
I have trained a Logistic Regression model, K Nearest Neighbours classifier, Gaussian Naive Bayes classifier, and a Support Vector Machine.

Change the value in the clf_opt variable to run different classifiers. <br>
clf_opt = 'svm' for Support Vector Machine<br>
clf_opt = 'knn' for K Neareat Neighbours<br>
clf_opt = 'nb' for Naive Bayes classifier<br>
clf_opt = 'lr' for Logistic Regression

## UnSupervised Learning
Use n_clusters to change number of clusters (Use proposed model with n_clusters = 2)
Set save_output = True to save the cluster predictions list to a .txt file
Choose cls_opt to change the clustering algorithm. Accepted values are written against clustering algorithms below

### I have implemented 10 clustering algorithms:
- K-Means Clustering Algorithm - 'kmeans'
- Single Linkage Hierarchical Agglomerative Clustering - 'single'
- Complete Linkage Hierarchical Agglomerative Clustering - 'complete'
- Average Linkage Hierarchical Agglomerative Clustering - 'average'
- OPTICS clustering - 'optics'
- DBSCAN clustering - 'dbscan'
- Spectral clustering - 'spectral'
- K Mediod Clustering - 'kmedoid'
- Buckshot Clustering algorithm - 'buckshot'
- Proposed Algorithm - 'proposed'

### Proposed Algorithm
- Runs DBSCAN on full dataset
- Saves two most popular classes and reclassifies rest of the datapoints, say C1 and C2
- Each remaining datapoint is classified to the class corresponding to the closest classified point
- To avoid chaining effect, only original C1 and C2 are considered while re-classifying
