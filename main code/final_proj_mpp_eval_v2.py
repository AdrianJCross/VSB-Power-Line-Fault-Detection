"""
COSC 522
Final Project - Milestone 3
Adrian Cross, Xuesong Fan, and Aaron Wilson
"""

import numpy as np
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



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

# Use the following function if it is desired to reduce the size of the training set (0's are over-represented while 1's are under-represented)
def reduce_training_size(Tr, ytrain, pw0):

    num_pos = np.count_nonzero(ytrain)

    Tr_comb = np.column_stack((Tr, ytrain))
    # P(w0) = 0.9, P(w1) = 0.1
    num_desired_neg = np.int(np.round(num_pos * (pw0/(1-pw0))))

    Tr_neg_shuf = shuffle(Tr_comb[ytrain == 0])
    Tr_pos_shuf = shuffle(Tr_comb[ytrain == 1])

    Tr_new_neg = Tr_neg_shuf[0:num_desired_neg]

    Tr_new = shuffle(np.row_stack((Tr_new_neg, Tr_pos_shuf)))

    return Tr_new[:, :-1], Tr_new[:, -1].astype(int)

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
    # print('accuracy = ', "%.2f" %(accu * 100), '%')
    # print('TP = ', TP)
    # print('TN = ', TN)
    # print('FP = ', FP)
    # print('FN = ', FN)
    # print('sensitivity = ', "%.2f" %(sens * 100), '%')
    # print('specificity = ', "%.2f" %(spec * 100), '%')
    return accu


class mpp:
    def __init__(self, case=1):
        self.case_ = case

    def fit(self, Tr, y, pw):
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
            # self.pw_[c] = len(arr) / len(y)
            self.pw_[c] = pw[c]

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

def plot_results(ntpr, nfpr, nacc, ftpr, ffpr, facc, ptpr, pfpr, pacc, p):

    plt.subplot(121)
    plt.plot(p, nacc, 'bo-', label='nX accuracy')
    plt.plot(p, facc, 'r*:', label='fX accuracy')
    plt.plot(p, pacc, 'gd-.', label='pX accuracy')
    plt.grid()
    plt.xlabel('P(w0)')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')

    # Create median line
    median = np.linspace(0, 1, len(p))

    plt.subplot(122)
    plt.plot(nfpr, ntpr, 'bo-', label='nX ROC')
    plt.plot(ffpr, ftpr, 'r*:', label='fX ROC')
    plt.plot(pfpr, ptpr, 'gd-.', label='pX ROC')
    plt.plot(median, median, 'bs-', label='median')
    plt.grid()
    plt.xlabel('False Positive Rate (%)')
    plt.ylabel('True Positive Rate (%)')
    plt.legend(loc='lower right')

    plt.show()



def main():

    case = 3
    prior_range_class0 = np.linspace(0.05, 0.95, 19)
    prior_range_class1 = 1 - prior_range_class0

    nX_tp = np.zeros(len(prior_range_class0))
    nX_fp = np.zeros(len(nX_tp))
    nX_acc = np.zeros(len(nX_tp))

    pX_tp = np.zeros(len(prior_range_class0))
    pX_fp = np.zeros(len(pX_tp))
    pX_acc = np.zeros(len(pX_tp))

    fX_tp = np.zeros(len(prior_range_class0))
    fX_fp = np.zeros(len(fX_tp))
    fX_acc = np.zeros(len(fX_tp))

    Xtrain, ytrain = load_training('training_data_new.csv')
    Xtest = load_testing('test_data_new.csv')

    Xtrain, ytrain = reduce_training_size(Xtrain, ytrain, pw0=0.80)

    n0 = len(np.where(ytrain == 0)[0])
    n1 = len(np.where(ytrain == 1)[0])

    nXtrain, nXtest = norm(Xtrain, Xtest)
    pXtrain, pXtest = pca(nXtrain, nXtest, 0.1)
    fXtrain, fXtest = fld(nXtrain, ytrain, nXtest)

    k = 10

    kf = KFold(n_splits=k)

    evals_nX = [None] * k
    conf_mats_nX = [None] * k

    # Iterate over each prior probability

    for p in range(0, len(prior_range_class0)):

        pw = [prior_range_class0[p], prior_range_class1[p]]

        count_nX = 0
        ## Normalized Data First
        y_models_n = [None] * k
        y_tests_n = [None] * k

        conf_mat_nX = np.zeros([2, 2])
        for train_index, test_index in kf.split(nXtrain):

            nXtrain_batch, nXtest_batch = nXtrain[train_index], nXtrain[test_index]
            ytrain_batch, y_tests_n[count_nX] = ytrain[train_index], ytrain[test_index]

            model = mpp(case=case)
            model.fit(nXtrain_batch, ytrain_batch, pw=pw)
            y_models_n[count_nX] = model.disc(nXtest_batch)

            evals_nX[count_nX] = eva(y_tests_n[count_nX], y_models_n[count_nX])
            conf_mat_nX += confusion_matrix(y_tests_n[count_nX], y_models_n[count_nX])

            count_nX += 1

        accu_overall_nX = np.mean(evals_nX)
        nX_acc[p] = accu_overall_nX
        nX_tp[p] = conf_mat_nX[0, 0]
        nX_fp[p] = conf_mat_nX[1, 0]

        # print('Case {}: nX accuracy = {}, k-fold = {}'.format(case, accu_overall_nX, k))
        # print('Case {}: nX confusion matrix = {}, k-fold = {}'.format(case, conf_mat_overall_nX, k))

        evals_fX = [None] * k
        conf_mats_fX = [None] * k

        count_fX = 0

        y_models_f = [None] * k
        y_test_f = [None] * k

        conf_mat_fX = np.zeros([2, 2])
        ## FLD Data
        for train_index, test_index in kf.split(fXtrain):
            fXtrain_batch, fXtest_batch = fXtrain[train_index], fXtrain[test_index]
            ytrain_batch, y_test_f[count_fX] = ytrain[train_index], ytrain[test_index]

            model = mpp(case=case)
            model.fit(fXtrain_batch, ytrain_batch, pw=pw)
            y_models_f[count_fX] = model.disc(fXtest_batch)

            evals_fX[count_fX] = eva(y_test_f[count_fX], y_models_f[count_fX])
            conf_mat_fX += confusion_matrix(y_test_f[count_fX], y_models_f[count_fX])

            count_fX += 1


        accu_overall_fX = np.mean(evals_fX)

        fX_acc[p] = accu_overall_fX
        fX_tp[p] = conf_mat_fX[0, 0]
        fX_fp[p] = conf_mat_fX[1, 0]

        # print('Case {}: fX accuracy = {}, k-fold = {}'.format(case, accu_overall_fX, k))
        # print('Case {}: fX confusion matrix = {}, k-fold = {}'.format(case, conf_mat_overall_fX, k))

        evals_pX = [None] * k
        conf_mats_pX = [None] * k

        count_pX = 0
        y_models_p = [None] * k
        y_tests_p = [None] * k
        conf_mat_pX = np.zeros([2, 2])

        ## PCA Data
        for train_index, test_index in kf.split(pXtrain):
            pXtrain_batch, pXtest_batch = pXtrain[train_index], pXtrain[test_index]
            ytrain_batch, y_tests_p[count_pX] = ytrain[train_index], ytrain[test_index]

            model = mpp(case=case)
            model.fit(pXtrain_batch, ytrain_batch, pw=pw)
            y_models_p[count_pX] = model.disc(pXtest_batch)

            evals_pX[count_pX] = eva(y_tests_p[count_pX], y_models_p[count_pX])
            conf_mat_pX += confusion_matrix(y_tests_p[count_pX], y_models_p[count_pX])

            count_pX += 1


        accu_overall_pX = np.mean(evals_pX)
        pX_acc[p] = accu_overall_pX
        pX_tp[p] = conf_mat_pX[0, 0]
        pX_fp[p] = conf_mat_pX[1, 0]

        # print('Case {}: pX accuracy = {}, k-fold = {}'.format(case, accu_overall_pX, k))
        # print('Case {}: pX confusion matrix = {}, k-fold = {}'.format(case, conf_mat_overall_pX, k))

    nX_tp = nX_tp / n0
    nX_fp = nX_fp / n1

    fX_tp = fX_tp / n0
    fX_fp = fX_fp / n1

    pX_tp = pX_tp / n0
    pX_fp = pX_fp / n1

    plot_results(nX_tp, nX_fp, nX_acc, fX_tp, fX_fp, fX_acc, pX_tp, pX_fp, pX_acc, prior_range_class0)

    cross_val_tests = [y_tests_n, y_test_f, y_tests_p]
    cross_val_models = [y_models_n, y_models_f, y_models_p]
    conf_mats = [conf_mat_nX, conf_mat_fX, conf_mat_pX]

    np.save('mpp_reduced_tests', cross_val_tests)
    np.save('mpp_reduced_models', cross_val_models)
    np.save('mpp_reduced_cm', conf_mats)


if __name__ == "__main__":
    main()
