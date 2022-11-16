from classifier_20227 import read_data
from classifier_20227 import plot_data
from classifier_20227 import data_classification

#Use svm for Support Vecotr Machine
#Use knn for K Neareat Neighbours
#Use nb for Naive Bayes classifier
#Use lr for Logistic Regression
clf_opt='svm'

train_data, y, test_data = read_data()                  #Reading the csv files
plot_data(train_data, y)                                #plotting the training data
train_data, y, test_data = read_data()                  #Reading the csv files
clf=data_classification(clf_opt)                        #Defining the classifier/model
clf.classification_score(train_data,y, clf_opt)         #Print classification score of model using train-test split
clf.classification(train_data,test_data,y,clf_opt)      #Classify and write csv file of test data predictions