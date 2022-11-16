from cProfile import label
from traceback import clear_frames
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def read_data():
    train_data = pd.read_csv('training_data.csv',header=None)
    train_data.columns=['X1','X2']
    y = pd.read_csv('training_data_class_labels.csv', header = None)
    test = pd.read_csv('test_data.csv',header=None)

    return train_data, y, test

def plot_data(train_data,y):
    train_data_plus_y = train_data
    train_data_plus_y['target'] = y
    class_0 = train_data.loc[train_data['target'] == 0]
    class_1 = train_data.loc[train_data['target'] == 1]
    plt.plot(class_0['X1'],class_0['X2'], '.', label="0")
    plt.plot(class_1['X1'],class_1['X2'], '.', label ='1')
    plt.title("Original Data")
    plt.legend()
    plt.show()


class data_classification():
     def __init__(self,clf_opt='lr'):
        self.clf_opt=clf_opt

     def classification_pipepline(self):
        if self.clf_opt=='lr':
            print('\n\t Training Logistic Regression Classifier \n')
            clf = LogisticRegression(solver='newton-cg')        #separately tested multiple solvers, found best results using newton-cg

        elif self.clf_opt=='svm':
            print('\n\tTraining Support Vector Machine Classifier \n')
            clf = SVC()

        elif self.clf_opt=='knn':
            print('\n\tTraining K Nearest Neighbours Classifier \n')
            clf = KNeighborsClassifier(n_neighbors=15)          #separately tested values of k to find best f1 score for k=15

        elif self.clf_opt=='nb':
            print('\n\tTraining Naive Bayes Classifier \n')
            clf = GaussianNB()                                  #seperately tested gaussian and bernoulli naive bayes, GausssianNB gave best results

        else:
            print("Enter valid clf_opt")
            sys.exit(0)
        return clf

     def write_to_csv(self,y_pred,clf_opt):
        #writes the predictions of the test data to a csv file
        with open(clf_opt+"_output_20227.csv", "w") as f:
            for i in y_pred:
                f.write(str(i)+'\n')

     def classification_score(self,train_data,y,clf_opt):

        #Splits training dataset, trains the model on train split and evaluates on validation split
        X_train, X_test, y_train, y_test = train_test_split(train_data,y,train_size=0.8)
        clf = self.classification_pipepline()
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_test)
        print(classification_report(y_test, y_predicted))

        #Run classifier on full training dataset and plot the classified points
        y_pred=clf.predict(train_data)
        class_0 = train_data.loc[y_pred == 0]
        class_1 = train_data.loc[y_pred == 1]
        plt.plot(class_0['X1'],class_0['X2'], '.',label='0')
        plt.plot(class_1['X1'],class_1['X2'], '.',label='1')
        plt.title("Split done by "+ clf_opt)
        plt.legend()
        plt.show()

     def classification(self,train_data,test_data,y,clf_opt):
        #Predicts the classes of the testing data and writes the csv file
        clf = self.classification_pipepline()
        clf.fit(train_data,y)
        y_pred = clf.predict(test_data)
        self.write_to_csv(y_pred,clf_opt)
