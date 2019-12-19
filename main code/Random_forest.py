"""
COSC 522
Final Project - Random forest
Adrian Cross, Xuesong Fan, and Aaron Wilson
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


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
    # sens = TP / (TP + FN)
    # spec = TN / (TN + FP)
    # print('accuracy = ', "%.2f" %(accu * 100), '%')
    # print('TP = ', TP)
    # print('TN = ', TN)
    # print('FP = ', FP)
    # print('FN = ', FN)
    # print('sensitivity = ', "%.2f" %(sens * 100), '%')
    # print('specificity = ', "%.2f" %(spec * 100), '%')
    return accu


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

    n1 = np.count_nonzero(ytrain)
    n0 = len(ytrain) - n1

    nXtrain, nXtest = norm(Xtrain, Xtest)
    fXtrain, fXtest = fld(nXtrain, ytrain, nXtest)
    pXtrain, pXtest = pca(nXtrain, nXtest, 0.1)

    rf = RandomForestRegressor(n_estimators=100)

    num_subsets = 10

    kf = KFold(n_splits=num_subsets)

    for p in range(0, len(prior_range_class0)):

        pw = [prior_range_class0[p], prior_range_class1[p]]

        evals_nX = [None] * num_subsets
        conf_mats_nX = [None] * num_subsets

        count_nX = 0
        ## Normalized Data First
        for train_index, test_index in kf.split(nXtrain):

            nXtrain_batch, nXtest_batch = nXtrain[train_index], nXtrain[test_index]
            ytrain_batch, ytest_batch = ytrain[train_index], ytrain[test_index]

            rf.fit(nXtrain_batch, ytrain_batch)
            y_predicted = np.round(rf.predict(nXtest_batch))

            evals_nX[count_nX] = eva(ytest_batch, y_predicted)
            conf_mats_nX[count_nX] = confusion_matrix(ytest_batch, y_predicted).ravel()

            count_nX += 1

        performance_overall_nX = np.sum(np.asarray(conf_mats_nX), axis=0)
        conf_mat_overall_nX = np.array([[performance_overall_nX[3], performance_overall_nX[1]], [performance_overall_nX[2], performance_overall_nX[0]]])
        accu_overall_nX = np.mean(evals_nX)
        nX_acc[p] = accu_overall_nX
        nX_tp[p] = performance_overall_nX[3]
        nX_fp[p] = performance_overall_nX[1]

        evals_fX = [None] * num_subsets
        conf_mats_fX = [None] * num_subsets

        count_fX = 0
        ## FLD
        for train_index, test_index in kf.split(fXtrain):

            fXtrain_batch, fXtest_batch = fXtrain[train_index].reshape(-1, 1), fXtrain[test_index].reshape(-1, 1)
            ytrain_batch, ytest_batch = ytrain[train_index], ytrain[test_index]

            rf.fit(fXtrain_batch, ytrain_batch)
            y_predicted = np.round(rf.predict(fXtest_batch))

            evals_fX[count_fX] = eva(ytest_batch, y_predicted)
            conf_mats_fX[count_fX] = confusion_matrix(ytest_batch, y_predicted).ravel()

            count_fX += 1

        performance_overall_fX = np.sum(np.asarray(conf_mats_fX), axis=0)
        conf_mat_overall_fX = np.array([[performance_overall_fX[3], performance_overall_fX[1]], [performance_overall_fX[2], performance_overall_fX[0]]])
        accu_overall_fX = np.mean(evals_fX)
        fX_acc[p] = accu_overall_fX
        fX_tp[p] = performance_overall_fX[3]
        fX_fp[p] = performance_overall_fX[1]

        evals_pX = [None] * num_subsets
        conf_mats_pX = [None] * num_subsets

        count_pX = 0
        ## PCA
        for train_index, test_index in kf.split(pXtrain):

            pXtrain_batch, pXtest_batch = pXtrain[train_index], pXtrain[test_index]
            ytrain_batch, ytest_batch = ytrain[train_index], ytrain[test_index]

            rf.fit(pXtrain_batch, ytrain_batch)
            y_predicted = np.round(rf.predict(pXtest_batch))

            evals_pX[count_pX] = eva(ytest_batch, y_predicted)
            conf_mats_pX[count_pX] = confusion_matrix(ytest_batch, y_predicted).ravel()

            count_pX += 1

        performance_overall_pX = np.sum(np.asarray(conf_mats_pX), axis=0)
        conf_mat_overall_pX = np.array([[performance_overall_pX[3], performance_overall_pX[1]], [performance_overall_pX[2], performance_overall_pX[0]]])
        accu_overall_pX = np.mean(evals_pX)
        pX_acc[p] = accu_overall_pX
        pX_tp[p] = performance_overall_pX[3]
        pX_fp[p] = performance_overall_pX[1]

    nX_tp = nX_tp / n1
    nX_fp = nX_fp / n0

    fX_tp = fX_tp / n1
    fX_fp = fX_fp / n0

    pX_tp = pX_tp / n1
    pX_fp = pX_fp / n0

    plot_results(nX_tp, nX_fp, nX_acc, fX_tp, fX_fp, fX_acc, pX_tp, pX_fp, pX_acc, prior_range_class0)


if __name__ == "__main__":
    main()
