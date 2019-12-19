"""
COSC 522
Final Project - Milestone 3
Adrian Cross, Xuesong Fan, and Aaron Wilson
"""

import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC

def pandas_conv(file):
    new_file=pd.DataFrame(file)
    return new_file

def pandas_conv_class(file,class_file):
    new_file=pd.DataFrame(file)
    new_file['type']=class_file
    return new_file


def load_training(file):
    """load training data from file"""
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


def load_testing(file):
    """load testing data from file"""
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    return data


def euc2(a, b):
    """euclidean distance square"""
    return np.dot(np.transpose(a - b), (a - b))


def mah2(a, b, sigma):
    """mahalanobis distance square"""
    return np.dot(np.transpose(a - b), np.dot(np.linalg.inv(sigma), (a - b)))


def norm(Tr, Te):
    """normalize the data"""
    m_ = np.mean(Tr, axis=0)
    sigma_ = np.std(Tr, axis=0)
    nTr = (Tr - m_) / sigma_
    nTe = (Te - m_) / sigma_
    return nTr, nTe


def pca(Tr, Te, err):
    """PCA"""
    Tr_cov = np.cov(np.transpose(Tr))
    eigval, eigvec = np.linalg.eig(Tr_cov)
    sort_eigval = eigval[np.argsort(-eigval)]
    sort_eigvec = eigvec[np.argsort(-eigval)]
    tot_ = np.sum(sort_eigval)
    sum_ = 0.0
    for i in range(len(sort_eigval)):
        sum_ += sort_eigval[i]
        err_ = 1 - sum_ / tot_
        if err_ <= err:
            break
    print(i + 1, 'features were kept with the error rate of', "%.2f" %(err_ * 100), '%')
    P_ = sort_eigvec[:i + 1]
    pTr = Tr.dot(np.transpose(P_))
    pTe = Te.dot(np.transpose(P_))
    return pTr, pTe


def fld(Tr, y, Te):
    """FLD"""
    covs_, means_, n_, S_ = {}, {}, {}, {}
    Sw_ = None
    classes_ = np.unique(y)
    for c in classes_:
        arr = Tr[y == c]
        covs_[c] = np.cov(np.transpose(arr))
        means_[c] = np.mean(arr, axis=0)  # mean along rows
        n_[c] = len(arr)
        if Sw_ is None:
            Sw_ = (n_[c] - 1) * covs_[c]
        else:
            Sw_ += (n_[c] - 1) * covs_[c]
    w_ = np.dot(np.linalg.inv(Sw_), means_[0]-means_[1])
    fTr = Tr.dot(np.transpose(w_))
    fTe = Te.dot(np.transpose(w_))
    return fTr, fTe


def eva(y, y_model):
    """ return accuracy score """
    assert len(y) == len(y_model)
    accu = np.count_nonzero(y == y_model) / len(y)
    TP = TN = FP = FN = 0
    for i in range(len(y)):
        if y_model[i] == y[i] == 1:
            TP += 1
        if y_model[i] == y[i] == 0:
            TN += 1
        if y_model[i] == 1 and y_model[i] != y[i]:
            FP += 1
        if y_model[i] == 0 and y_model[i] != y[i]:
            FN += 1
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    print('accuracy = ', "%.2f" %(accu * 100), '%')
    print('TP = ', TP)
    print('TN = ', TN)
    print('FP = ', FP)
    print('FN = ', FN)
    print('sensitivity = ', "%.2f" %(sens * 100), '%')
    print('specificity = ', "%.2f" %(spec * 100), '%')
    return accu

def cross_val(Xtrain, Xtest, ytrain, ytest,model,k=1,kern='linear'):
    n1 = np.count_nonzero(ytrain)
    n0 = len(ytrain) - n1
    num_subsets = 10
    kf = KFold(n_splits=num_subsets)
    conf_mat = np.zeros([2, 2])
    acc = []
    
    for train_index, test_index in kf.split(Xtrain):

        Xtrain_batch, Xtest_batch = Xtrain[train_index], Xtrain[test_index]
        ytrain_batch, ytest_batch = ytrain[train_index], ytrain[test_index]

        # Train your model here using the 'Xtrain_batch' and 'ytrain_batch' data sets
        # Test the model using the 'Xtest_batch' and 'ytest_batch' data sets
        if model=='kNN':
            y_predicted = kNN(Xtrain_batch.copy(),ytrain_batch.copy(),Xtest_batch.copy(),k)
            
        if model=='k_means':
            #uses the training data to set the cluster centers
            #use the test data to find the closest cluster center
            kmeans = KMeans(n_clusters=2,max_iter=k).fit(Xtrain_batch)
            y_predicted = kmeans.predict(Xtest_batch)
        
        if model=='SVM':
            svclassifier = SVC(kernel=kern, degree=k,gamma='scale')
            svclassifier.fit(Xtrain_batch, ytrain_batch)
            y_predicted = svclassifier.predict(Xtest_batch)
        
        conf_mat += confusion_matrix(ytest_batch, y_predicted)
        acc.append(100*(np.count_nonzero(ytest_batch == y_predicted) / len(ytest_batch)))

    # Confusion matrix here is of the form
    #   [TN  FP]
    #   [FN  TP]

    # Get confusion matrix in percentages
    conf_mat = 100 * conf_mat / np.array([n0, n1])
    accuracy = np.mean(acc)
    return conf_mat, accuracy
  
def kNN(X_train,y_train,X_test,k):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred
       
    
    
class mpp:
    def __init__(self, case=1):
        self.case_ = case

    def fit(self, Tr, y):
        # derive the model
        self.covs_, self.means_, self.pw_ = {}, {}, {}
        self.covsum_ = None

        self.classes_ = np.unique(y)  # get unique labels as dictionary items
        self.classn_ = len(self.classes_)

        for c in self.classes_:
            arr = Tr[y == c]
            self.covs_[c] = np.cov(np.transpose(arr))
            self.means_[c] = np.mean(arr, axis=0)  # mean along rows
            if self.covsum_ is None:
                self.covsum_ = self.covs_[c].copy()
            else:
                self.covsum_ += self.covs_[c]
            self.pw_[c] = len(arr) / len(y)

        # used by case II
        self.covavg_ = self.covsum_ / self.classn_

        # used by case I
        if type(self.covavg_) != np.ndarray:
            self.varavg_ = self.covavg_.copy()
        else:
            self.varavg_ = np.sum(np.diagonal(self.covavg_)) / len(self.covavg_)

        return None

    def disc(self, Te):
        # eval all data
        y = []
        disc = np.zeros(self.classn_)
        ne = len(Te)

        if type(self.covavg_) != np.ndarray:
            for i in range(ne):
                for c in self.classes_:
                    if self.case_ == 1:
                        edist2 = (Te[i] - self.means_[c]) ** 2
                        disc[c] = -edist2 / (2 * self.varavg_) + np.log(self.pw_[c])
                    elif self.case_ == 2:
                        mdist2 = ((Te[i] - self.means_[c]) ** 2) / self.covavg_
                        disc[c] = -mdist2 / 2 + np.log(self.pw_[c])
                    elif self.case_ == 3:
                        mdist2 = ((Te[i] - self.means_[c]) ** 2) / self.covs_[c]
                        disc[c] = -mdist2 / 2 - np.log(self.covs_[c]) / 2 + np.log(self.pw_[c])
                    else:
                        print("Can only handle case numbers 1, 2, 3.")
                        sys.exit(1)
                y.append(disc.argmax())
        else:
            for i in range(ne):
                for c in self.classes_:
                    if self.case_ == 1:
                        edist2 = euc2(self.means_[c], Te[i])
                        disc[c] = -edist2 / (2 * self.varavg_) + np.log(self.pw_[c])
                    elif self.case_ == 2:
                        mdist2 = mah2(self.means_[c], Te[i], self.covavg_)
                        disc[c] = -mdist2 / 2 + np.log(self.pw_[c])
                    elif self.case_ == 3:
                        mdist2 = mah2(self.means_[c], Te[i], self.covs_[c])
                        disc[c] = -mdist2 / 2 - np.log(np.linalg.det(self.covs_[c])) / 2 \
                                  + np.log(self.pw_[c])
                    else:
                        print("Can only handle case numbers 1, 2, 3.")
                        sys.exit(1)
                y.append(disc.argmax())

        return y


