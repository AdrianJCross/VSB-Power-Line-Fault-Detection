"""
COSC 522
Final Project - Main Code
Adrian Cross, Xuesong Fan, and Aaron Wilson
"""

import numpy as np, time
from sklearn.model_selection import KFold
import MPP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans


def load_data(file):
    """load training data from file"""
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    X = data[:, 4:-1]
    y = data[:, -1].astype(int)
    return X, y


def norm(Tr):
    """normalize the data"""
    m_ = np.mean(Tr, axis=0)
    sigma_ = np.std(Tr, axis=0)
    nTr = (Tr - m_) / sigma_
    return nTr


def fld(Tr, y):
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
    return fTr


def pca(Tr, err):
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
    return pTr


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
    '''
    spec = TN / (TN + FP)
    print('accuracy = ', "%.2f" %(accu * 100), '%')
    print('TP = ', TP)
    print('TN = ', TN)
    print('FP = ', FP)
    print('FN = ', FN)
    print('sensitivity = ', "%.2f" %(sens * 100), '%')
    print('specificity = ', "%.2f" %(spec * 100), '%')
    '''
    return accu, sens


def main():
    # Load data
    Xtrain, ytrain = load_data('training_data_new.csv')

    # Normalization and Dimensionality Reduction
    nXtrain = norm(Xtrain)
    fXtrain = fld(nXtrain, ytrain)
    pXtrain = pca(nXtrain, 0.1)

    ksplit = 10
    kf = KFold(n_splits=ksplit)

    '''
    ###############
    ##### MPP #####
    ###############
    for n in range(1, 4):
        print('MPP-case', n)

        # Normalized
        nX_conf_mat = np.zeros([2, 2])
        count_nX = 0
        accs_nX = [None] * ksplit
        sens_nX = [None] * ksplit
        y_models_n = [None] * ksplit
        y_tests_n = [None] * ksplit
        nX_time_start = time.time()
        for train_index, test_index in kf.split(nXtrain):
            nXtrain_batch, nXtest_batch = nXtrain[train_index], nXtrain[test_index]
            ytrain_batch, y_tests_n[count_nX] = ytrain[train_index], ytrain[test_index]
            model = MPP.mpp(case=n)
            model.fit(nXtrain_batch, ytrain_batch)
            y_models_n[count_nX] = model.disc(nXtest_batch)
            nX_conf_mat += confusion_matrix(y_tests_n[count_nX], y_models_n[count_nX])
            accs_nX[count_nX], sens_nX[count_nX] = eva(y_tests_n[count_nX], y_models_n[count_nX])
            count_nX += 1
        nX_time_end = time.time()
        acc_nX = np.mean(accs_nX)
        sen_nX = np.mean(sens_nX)
        TN = nX_conf_mat[0, 0]
        FP = nX_conf_mat[0, 1]
        FN = nX_conf_mat[1, 0]
        TP = nX_conf_mat[1, 1]
        MCC_nX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        print('nX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_nX, sen_nX, MCC_nX, (nX_time_end - nX_time_start)))

        # FLD
        fX_conf_mat = np.zeros([2, 2])
        count_fX = 0
        accs_fX = [None] * ksplit
        sens_fX = [None] * ksplit
        y_models_f = [None] * ksplit
        y_tests_f = [None] * ksplit
        fX_time_start = time.time()
        for train_index, test_index in kf.split(fXtrain):
            fXtrain_batch, fXtest_batch = fXtrain[train_index], fXtrain[test_index]
            ytrain_batch, y_tests_f[count_fX] = ytrain[train_index], ytrain[test_index]
            model = MPP.mpp(case=n)
            model.fit(fXtrain_batch, ytrain_batch)
            y_models_f[count_fX] = model.disc(fXtest_batch)
            fX_conf_mat += confusion_matrix(y_tests_f[count_fX], y_models_f[count_fX])
            accs_fX[count_fX], sens_fX[count_fX] = eva(y_tests_f[count_fX], y_models_f[count_fX])
            count_fX += 1
        fX_time_end = time.time()
        acc_fX = np.mean(accs_fX)
        sen_fX = np.mean(sens_fX)
        TN = fX_conf_mat[0, 0]
        FP = fX_conf_mat[0, 1]
        FN = fX_conf_mat[1, 0]
        TP = fX_conf_mat[1, 1]
        MCC_fX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        print('fX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_fX, sen_fX, MCC_fX, (fX_time_end - fX_time_start)))

        # PCA
        pX_conf_mat = np.zeros([2, 2])
        count_pX = 0
        accs_pX = [None] * ksplit
        sens_pX = [None] * ksplit
        y_models_p = [None] * ksplit
        y_tests_p = [None] * ksplit
        pX_time_start = time.time()
        for train_index, test_index in kf.split(pXtrain):
            pXtrain_batch, pXtest_batch = pXtrain[train_index], pXtrain[test_index]
            ytrain_batch, y_tests_p[count_pX] = ytrain[train_index], ytrain[test_index]
            model = MPP.mpp(case=n)
            model.fit(pXtrain_batch, ytrain_batch)
            y_models_p[count_pX] = model.disc(pXtest_batch)
            pX_conf_mat += confusion_matrix(y_tests_p[count_pX], y_models_p[count_pX])
            accs_pX[count_pX], sens_pX[count_pX] = eva(y_tests_p[count_pX], y_models_p[count_pX])
            count_pX += 1
        pX_time_end = time.time()
        acc_pX = np.mean(accs_pX)
        sen_pX = np.mean(sens_pX)
        TN = pX_conf_mat[0, 0]
        FP = pX_conf_mat[0, 1]
        FN = pX_conf_mat[1, 0]
        TP = pX_conf_mat[1, 1]
        MCC_pX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        print('pX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_pX, sen_pX, MCC_pX, (pX_time_end - pX_time_start)))

    ###############
    ##### kNN #####
    ###############
    k = 10
    print('kNN, k =', k)

    # Normalized
    nX_conf_mat = np.zeros([2, 2])
    count_nX = 0
    accs_nX = [None] * ksplit
    sens_nX = [None] * ksplit
    y_models_n = [None] * ksplit
    y_tests_n = [None] * ksplit
    nX_time_start = time.time()
    for train_index, test_index in kf.split(nXtrain):
        nXtrain_batch, nXtest_batch = nXtrain[train_index], nXtrain[test_index]
        ytrain_batch, y_tests_n[count_nX] = ytrain[train_index], ytrain[test_index]
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(nXtrain_batch, ytrain_batch)
        y_models_n[count_nX] = model.predict(nXtest_batch)
        nX_conf_mat += confusion_matrix(y_tests_n[count_nX], y_models_n[count_nX])
        accs_nX[count_nX], sens_nX[count_nX] = eva(y_tests_n[count_nX], y_models_n[count_nX])
        count_nX += 1
    nX_time_end = time.time()
    acc_nX = np.mean(accs_nX)
    sen_nX = np.mean(sens_nX)
    TN = nX_conf_mat[0, 0]
    FP = nX_conf_mat[0, 1]
    FN = nX_conf_mat[1, 0]
    TP = nX_conf_mat[1, 1]
    MCC_nX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('nX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_nX, sen_nX, MCC_nX, (
            nX_time_end - nX_time_start)))

    # FLD
    fX_conf_mat = np.zeros([2, 2])
    count_fX = 0
    accs_fX = [None] * ksplit
    sens_fX = [None] * ksplit
    y_models_f = [None] * ksplit
    y_tests_f = [None] * ksplit
    fX_time_start = time.time()
    for train_index, test_index in kf.split(fXtrain):
        fXtrain_batch, fXtest_batch = fXtrain[train_index].reshape(-1, 1), fXtrain[test_index].reshape(-1, 1)
        ytrain_batch, y_tests_f[count_fX] = ytrain[train_index], ytrain[test_index]
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(fXtrain_batch, ytrain_batch)
        y_models_f[count_fX] = model.predict(fXtest_batch)
        fX_conf_mat += confusion_matrix(y_tests_f[count_fX], y_models_f[count_fX])
        accs_fX[count_fX], sens_fX[count_fX] = eva(y_tests_f[count_fX], y_models_f[count_fX])
        count_fX += 1
    fX_time_end = time.time()
    acc_fX = np.mean(accs_fX)
    sen_fX = np.mean(sens_fX)
    TN = fX_conf_mat[0, 0]
    FP = fX_conf_mat[0, 1]
    FN = fX_conf_mat[1, 0]
    TP = fX_conf_mat[1, 1]
    MCC_fX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('fX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_fX, sen_fX, MCC_fX, (fX_time_end - fX_time_start)))

    # PCA
    pX_conf_mat = np.zeros([2, 2])
    count_pX = 0
    accs_pX = [None] * ksplit
    sens_pX = [None] * ksplit
    y_models_p = [None] * ksplit
    y_tests_p = [None] * ksplit
    pX_time_start = time.time()
    for train_index, test_index in kf.split(pXtrain):
        pXtrain_batch, pXtest_batch = pXtrain[train_index], pXtrain[test_index]
        ytrain_batch, y_tests_p[count_pX] = ytrain[train_index], ytrain[test_index]
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(pXtrain_batch, ytrain_batch)
        y_models_p[count_pX] = model.predict(pXtest_batch)
        pX_conf_mat += confusion_matrix(y_tests_p[count_pX], y_models_p[count_pX])
        accs_pX[count_pX], sens_pX[count_pX] = eva(y_tests_p[count_pX], y_models_p[count_pX])
        count_pX += 1
    pX_time_end = time.time()
    acc_pX = np.mean(accs_pX)
    sen_pX = np.mean(sens_pX)
    TN = pX_conf_mat[0, 0]
    FP = pX_conf_mat[0, 1]
    FN = pX_conf_mat[1, 0]
    TP = pX_conf_mat[1, 1]
    MCC_pX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('pX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_pX, sen_pX, MCC_pX, (pX_time_end - pX_time_start)))

    '''
    ################
    ##### BPNN #####
    ################
    print('BPNN')
    nSize = len(nXtrain.T)
    fSize = 1
    pSize = len(pXtrain.T)
    num_neurons = 3
    num_hidden_layers = 1
    learning_rate = 3.0
    epochs = 30

    # Normalized
    nX_conf_mat = np.zeros([2, 2])
    nX_time_start = time.time()
    count_nX = 0
    accs_nX = [None] * ksplit
    y_models_n = [None] * ksplit
    y_tests_n = [None] * ksplit
    for train_index, test_index in kf.split(nXtrain):
        nXtrain_batch, nXtest_batch = nXtrain[train_index], nXtrain[test_index]
        ytrain_batch, y_tests_n[count_nX] = ytrain[train_index], ytrain[test_index]
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(num_neurons, input_shape=((nSize,)), activation='sigmoid'))
        if num_hidden_layers > 1:
            for h in range(0, num_hidden_layers):
                model.add(keras.layers.Dense(num_neurons, activation='sigmoid'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        # Comment out the below if confident in neural network size
        # The following two lines simply print out the size of each layer in the neural net
        for layer in model.layers:
            print("Input shape: ", layer.input_shape[1], ", Output shape: ", layer.output_shape[1])
        simple_sgd = keras.optimizers.SGD(lr=learning_rate)
        model.compile(optimizer=simple_sgd, loss='mean_squared_error', metrics=['accuracy'])
        hist = model.fit(nXtrain_batch, ytrain_batch, epochs=epochs)
        # model.evaluate(nXtest_batch, ytest_batch)
        y_models_n[count_nX] = model.predict(nXtest_batch) > 0.5
        nX_conf_mat += confusion_matrix(y_tests_n[count_nX], y_models_n[count_nX])
        accs_nX[count_nX] = np.mean(hist.history.get('accuracy'))
        count_nX += 1
    nX_time_end = time.time()
    acc_nX = np.mean(accs_nX)
    TN = nX_conf_mat[0, 0]
    FP = nX_conf_mat[0, 1]
    FN = nX_conf_mat[1, 0]
    TP = nX_conf_mat[1, 1]
    sen_nX = TP/(TP + FN)
    MCC_nX = (TP * TN - FP * FN)/np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    # FLD
    fX_conf_mat = np.zeros([2, 2])
    fX_time_start = time.time()
    count_fX = 0
    accs_fX = [None] * ksplit
    y_models_f = [None] * ksplit
    y_tests_f = [None] * ksplit
    for train_index, test_index in kf.split(fXtrain):
        fXtrain_batch, fXtest_batch = fXtrain[train_index].reshape(-1, 1), fXtrain[test_index].reshape(-1, 1)
        ytrain_batch, y_tests_f[count_fX] = ytrain[train_index], ytrain[test_index]
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(num_neurons, input_shape=((fSize,)), activation='sigmoid'))
        if num_hidden_layers > 1:
            for h in range(0, num_hidden_layers):
                model.add(keras.layers.Dense(num_neurons, activation='sigmoid'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        # Comment out the below if confident in neural network size
        # The following two lines simply print out the size of each layer in the neural net
        for layer in model.layers:
            print("Input shape: ", layer.input_shape[1], ", Output shape: ", layer.output_shape[1])
        simple_sgd = keras.optimizers.SGD(lr=learning_rate)
        model.compile(optimizer=simple_sgd, loss='mean_squared_error', metrics=['accuracy'])
        hist = model.fit(fXtrain_batch, ytrain_batch, epochs=epochs)
        # model.evaluate(fXtest_batch, ytest_batch)
        y_models_f[count_fX] = model.predict(fXtest_batch) > 0.5
        fX_conf_mat += confusion_matrix(y_tests_f[count_fX], y_models_f[count_fX])
        accs_fX[count_fX] = np.mean(hist.history.get('accuracy'))
        count_fX += 1
    fX_time_end = time.time()
    acc_fX = np.mean(accs_fX)
    TN = fX_conf_mat[0, 0]
    FP = fX_conf_mat[0, 1]
    FN = fX_conf_mat[1, 0]
    TP = fX_conf_mat[1, 1]
    sen_fX = TP/(TP + FN)
    MCC_fX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    # PCA
    pX_conf_mat = np.zeros([2, 2])
    pX_time_start = time.time()
    count_pX = 0
    accs_pX = [None] * ksplit
    y_models_p = [None] * ksplit
    y_tests_p = [None] * ksplit
    for train_index, test_index in kf.split(pXtrain):
        pXtrain_batch, pXtest_batch = pXtrain[train_index], pXtrain[test_index]
        ytrain_batch, y_tests_p[count_pX] = ytrain[train_index], ytrain[test_index]
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(num_neurons, input_shape=((pSize,)), activation='sigmoid'))
        if num_hidden_layers > 1:
            for h in range(0, num_hidden_layers):
                model.add(keras.layers.Dense(num_neurons, activation='sigmoid'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        # Comment out the below if confident in neural network size
        # The following two lines simply print out the size of each layer in the neural net
        for layer in model.layers:
            print("Input shape: ", layer.input_shape[1], ", Output shape: ", layer.output_shape[1])
        simple_sgd = keras.optimizers.SGD(lr=learning_rate)
        model.compile(optimizer=simple_sgd, loss='mean_squared_error', metrics=['accuracy'])
        hist = model.fit(pXtrain_batch, ytrain_batch, epochs=epochs)
        # model.evaluate(pXtest_batch, ytest_batch)
        y_models_p[count_pX] = model.predict(pXtest_batch) > 0.5
        pX_conf_mat += confusion_matrix(y_tests_p[count_pX], y_models_p[count_pX])
        accs_pX[count_pX] = np.mean(hist.history.get('accuracy'))
        count_pX += 1
    pX_time_end = time.time()
    acc_pX = np.mean(accs_pX)
    TN = pX_conf_mat[0, 0]
    FP = pX_conf_mat[0, 1]
    FN = pX_conf_mat[1, 0]
    TP = pX_conf_mat[1, 1]
    sen_pX = TP/(TP + FN)
    MCC_pX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    print('nX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_nX, sen_nX, MCC_nX, (
            nX_time_end - nX_time_start)))
    print('fX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_fX, sen_fX, MCC_fX, (
                fX_time_end - fX_time_start)))
    print('pX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_pX, sen_pX, MCC_pX, (pX_time_end - pX_time_start)))

    '''
    #########################
    ##### Random Forest #####
    #########################
    no_trees = 100
    print('Random Forest with number of trees =', no_trees)

    # Normalized
    nX_conf_mat = np.zeros([2, 2])
    count_nX = 0
    accs_nX = [None] * ksplit
    sens_nX = [None] * ksplit
    y_models_n = [None] * ksplit
    y_tests_n = [None] * ksplit
    nX_time_start = time.time()
    for train_index, test_index in kf.split(nXtrain):
        nXtrain_batch, nXtest_batch = nXtrain[train_index], nXtrain[test_index]
        ytrain_batch, y_tests_n[count_nX] = ytrain[train_index], ytrain[test_index]
        model = RandomForestRegressor(n_estimators=no_trees)
        model.fit(nXtrain_batch, ytrain_batch)
        y_models_n[count_nX] = np.round(model.predict(nXtest_batch))
        nX_conf_mat += confusion_matrix(y_tests_n[count_nX], y_models_n[count_nX])
        accs_nX[count_nX], sens_nX[count_nX] = eva(y_tests_n[count_nX], y_models_n[count_nX])
        count_nX += 1
    nX_time_end = time.time()
    acc_nX = np.mean(accs_nX)
    sen_nX = np.mean(sens_nX)
    TN = nX_conf_mat[0, 0]
    FP = nX_conf_mat[0, 1]
    FN = nX_conf_mat[1, 0]
    TP = nX_conf_mat[1, 1]
    MCC_nX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('nX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_nX, sen_nX, MCC_nX, (
            nX_time_end - nX_time_start)))

    # FLD
    fX_conf_mat = np.zeros([2, 2])
    count_fX = 0
    accs_fX = [None] * ksplit
    sens_fX = [None] * ksplit
    y_models_f = [None] * ksplit
    y_tests_f = [None] * ksplit
    fX_time_start = time.time()
    for train_index, test_index in kf.split(fXtrain):
        fXtrain_batch, fXtest_batch = fXtrain[train_index].reshape(-1, 1), fXtrain[test_index].reshape(-1, 1)
        ytrain_batch, y_tests_f[count_fX] = ytrain[train_index], ytrain[test_index]
        model = RandomForestRegressor(n_estimators=no_trees)
        model.fit(fXtrain_batch, ytrain_batch)
        y_models_f[count_fX] = np.round(model.predict(fXtest_batch))
        fX_conf_mat += confusion_matrix(y_tests_f[count_fX], y_models_f[count_fX])
        accs_fX[count_fX], sens_fX[count_fX] = eva(y_tests_f[count_fX], y_models_f[count_fX])
        count_fX += 1
    fX_time_end = time.time()
    acc_fX = np.mean(accs_fX)
    sen_fX = np.mean(sens_fX)
    TN = fX_conf_mat[0, 0]
    FP = fX_conf_mat[0, 1]
    FN = fX_conf_mat[1, 0]
    TP = fX_conf_mat[1, 1]
    MCC_fX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('fX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_fX, sen_fX, MCC_fX, (fX_time_end - fX_time_start)))

    # PCA
    pX_conf_mat = np.zeros([2, 2])
    count_pX = 0
    accs_pX = [None] * ksplit
    sens_pX = [None] * ksplit
    y_models_p = [None] * ksplit
    y_tests_p = [None] * ksplit
    pX_time_start = time.time()
    for train_index, test_index in kf.split(pXtrain):
        pXtrain_batch, pXtest_batch = pXtrain[train_index], pXtrain[test_index]
        ytrain_batch, y_tests_p[count_pX] = ytrain[train_index], ytrain[test_index]
        model = RandomForestRegressor(n_estimators=no_trees)
        model.fit(pXtrain_batch, ytrain_batch)
        y_models_p[count_pX] = np.round(model.predict(pXtest_batch))
        pX_conf_mat += confusion_matrix(y_tests_p[count_pX], y_models_p[count_pX])
        accs_pX[count_pX], sens_pX[count_pX] = eva(y_tests_p[count_pX], y_models_p[count_pX])
        count_pX += 1
    pX_time_end = time.time()
    acc_pX = np.mean(accs_pX)
    sen_pX = np.mean(sens_pX)
    TN = pX_conf_mat[0, 0]
    FP = pX_conf_mat[0, 1]
    FN = pX_conf_mat[1, 0]
    TP = pX_conf_mat[1, 1]
    MCC_pX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('pX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_pX, sen_pX, MCC_pX, (pX_time_end - pX_time_start)))

    ###############
    ##### SVM #####
    ###############
    kernel_ = 'rbf'
    degree_ = 1
    print('SVM')

    # Normalized
    nX_conf_mat = np.zeros([2, 2])
    count_nX = 0
    accs_nX = [None] * ksplit
    sens_nX = [None] * ksplit
    y_models_n = [None] * ksplit
    y_tests_n = [None] * ksplit
    nX_time_start = time.time()
    for train_index, test_index in kf.split(nXtrain):
        nXtrain_batch, nXtest_batch = nXtrain[train_index], nXtrain[test_index]
        ytrain_batch, y_tests_n[count_nX] = ytrain[train_index], ytrain[test_index]
        model = SVC(kernel=kernel_, degree=degree_, gamma='scale')
        model.fit(nXtrain_batch, ytrain_batch)
        y_models_n[count_nX] = model.predict(nXtest_batch)
        nX_conf_mat += confusion_matrix(y_tests_n[count_nX], y_models_n[count_nX])
        accs_nX[count_nX], sens_nX[count_nX] = eva(y_tests_n[count_nX], y_models_n[count_nX])
        count_nX += 1
    nX_time_end = time.time()
    acc_nX = np.mean(accs_nX)
    sen_nX = np.mean(sens_nX)
    TN = nX_conf_mat[0, 0]
    FP = nX_conf_mat[0, 1]
    FN = nX_conf_mat[1, 0]
    TP = nX_conf_mat[1, 1]
    MCC_nX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('nX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_nX, sen_nX, MCC_nX, (
            nX_time_end - nX_time_start)))

    # FLD
    fX_conf_mat = np.zeros([2, 2])
    count_fX = 0
    accs_fX = [None] * ksplit
    sens_fX = [None] * ksplit
    y_models_f = [None] * ksplit
    y_tests_f = [None] * ksplit
    fX_time_start = time.time()
    for train_index, test_index in kf.split(fXtrain):
        fXtrain_batch, fXtest_batch = fXtrain[train_index].reshape(-1, 1), fXtrain[test_index].reshape(-1, 1)
        ytrain_batch, y_tests_f[count_fX] = ytrain[train_index], ytrain[test_index]
        model = SVC(kernel=kernel_, degree=degree_, gamma='scale')
        model.fit(fXtrain_batch, ytrain_batch)
        y_models_f[count_fX] = np.round(model.predict(fXtest_batch))
        fX_conf_mat += confusion_matrix(y_tests_f[count_fX], y_models_f[count_fX])
        accs_fX[count_fX], sens_fX[count_fX] = eva(y_tests_f[count_fX], y_models_f[count_fX])
        count_fX += 1
    fX_time_end = time.time()
    acc_fX = np.mean(accs_fX)
    sen_fX = np.mean(sens_fX)
    TN = fX_conf_mat[0, 0]
    FP = fX_conf_mat[0, 1]
    FN = fX_conf_mat[1, 0]
    TP = fX_conf_mat[1, 1]
    MCC_fX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('fX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_fX, sen_fX, MCC_fX, (fX_time_end - fX_time_start)))

    # PCA
    pX_conf_mat = np.zeros([2, 2])
    count_pX = 0
    accs_pX = [None] * ksplit
    sens_pX = [None] * ksplit
    y_models_p = [None] * ksplit
    y_tests_p = [None] * ksplit
    pX_time_start = time.time()
    for train_index, test_index in kf.split(pXtrain):
        pXtrain_batch, pXtest_batch = pXtrain[train_index], pXtrain[test_index]
        ytrain_batch, y_tests_p[count_pX] = ytrain[train_index], ytrain[test_index]
        model = SVC(kernel=kernel_, degree=degree_, gamma='scale')
        model.fit(pXtrain_batch, ytrain_batch)
        y_models_p[count_pX] = np.round(model.predict(pXtest_batch))
        pX_conf_mat += confusion_matrix(y_tests_p[count_pX], y_models_p[count_pX])
        accs_pX[count_pX], sens_pX[count_pX] = eva(y_tests_p[count_pX], y_models_p[count_pX])
        count_pX += 1
    pX_time_end = time.time()
    acc_pX = np.mean(accs_pX)
    sen_pX = np.mean(sens_pX)
    TN = pX_conf_mat[0, 0]
    FP = pX_conf_mat[0, 1]
    FN = pX_conf_mat[1, 0]
    TP = pX_conf_mat[1, 1]
    MCC_pX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('pX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_pX, sen_pX, MCC_pX, (pX_time_end - pX_time_start)))

    ##################
    ##### kmeans #####
    ##################
    n_clusters_ = 2
    max_iter_ = 10
    print('kmeans')

    # Normalized
    nX_conf_mat = np.zeros([2, 2])
    count_nX = 0
    accs_nX = [None] * ksplit
    sens_nX = [None] * ksplit
    y_models_n = [None] * ksplit
    y_tests_n = [None] * ksplit
    nX_time_start = time.time()
    for train_index, test_index in kf.split(nXtrain):
        nXtrain_batch, nXtest_batch = nXtrain[train_index], nXtrain[test_index]
        ytrain_batch, y_tests_n[count_nX] = ytrain[train_index], ytrain[test_index]
        model = KMeans(n_clusters=n_clusters_, max_iter=max_iter_).fit(nXtrain_batch)
        y_models_n[count_nX] = model.predict(nXtest_batch)
        nX_conf_mat += confusion_matrix(y_tests_n[count_nX], y_models_n[count_nX])
        accs_nX[count_nX], sens_nX[count_nX] = eva(y_tests_n[count_nX], y_models_n[count_nX])
        count_nX += 1
    nX_time_end = time.time()
    acc_nX = np.mean(accs_nX)
    sen_nX = np.mean(sens_nX)
    TN = nX_conf_mat[0, 0]
    FP = nX_conf_mat[0, 1]
    FN = nX_conf_mat[1, 0]
    TP = nX_conf_mat[1, 1]
    MCC_nX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('nX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_nX, sen_nX, MCC_nX, (
            nX_time_end - nX_time_start)))

    # FLD
    fX_conf_mat = np.zeros([2, 2])
    count_fX = 0
    accs_fX = [None] * ksplit
    sens_fX = [None] * ksplit
    y_models_f = [None] * ksplit
    y_tests_f = [None] * ksplit
    fX_time_start = time.time()
    for train_index, test_index in kf.split(fXtrain):
        fXtrain_batch, fXtest_batch = fXtrain[train_index].reshape(-1, 1), fXtrain[test_index].reshape(-1, 1)
        ytrain_batch, y_tests_f[count_fX] = ytrain[train_index], ytrain[test_index]
        model = KMeans(n_clusters=n_clusters_, max_iter=max_iter_).fit(fXtrain_batch)
        y_models_f[count_fX] = model.predict(fXtest_batch)
        fX_conf_mat += confusion_matrix(y_tests_f[count_fX], y_models_f[count_fX])
        accs_fX[count_fX], sens_fX[count_fX] = eva(y_tests_f[count_fX], y_models_f[count_fX])
        count_fX += 1
    fX_time_end = time.time()
    acc_fX = np.mean(accs_fX)
    sen_fX = np.mean(sens_fX)
    TN = fX_conf_mat[0, 0]
    FP = fX_conf_mat[0, 1]
    FN = fX_conf_mat[1, 0]
    TP = fX_conf_mat[1, 1]
    MCC_fX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('fX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_fX, sen_fX, MCC_fX, (fX_time_end - fX_time_start)))

    # PCA
    pX_conf_mat = np.zeros([2, 2])
    count_pX = 0
    accs_pX = [None] * ksplit
    sens_pX = [None] * ksplit
    y_models_p = [None] * ksplit
    y_tests_p = [None] * ksplit
    pX_time_start = time.time()
    for train_index, test_index in kf.split(pXtrain):
        pXtrain_batch, pXtest_batch = pXtrain[train_index], pXtrain[test_index]
        ytrain_batch, y_tests_p[count_pX] = ytrain[train_index], ytrain[test_index]
        model = KMeans(n_clusters=n_clusters_, max_iter=max_iter_).fit(pXtrain_batch)
        y_models_p[count_pX] = model.predict(pXtest_batch)
        pX_conf_mat += confusion_matrix(y_tests_p[count_pX], y_models_p[count_pX])
        accs_pX[count_pX], sens_pX[count_pX] = eva(y_tests_p[count_pX], y_models_p[count_pX])
        count_pX += 1
    pX_time_end = time.time()
    acc_pX = np.mean(accs_pX)
    sen_pX = np.mean(sens_pX)
    TN = pX_conf_mat[0, 0]
    FP = pX_conf_mat[0, 1]
    FN = pX_conf_mat[1, 0]
    TP = pX_conf_mat[1, 1]
    MCC_pX = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('pX: accuracy: {:.4f}, sensitivity: {:.4f}, MCC: {:.4f}, time to complete: {:.4f} seconds'.format(acc_pX, sen_pX, MCC_pX, (pX_time_end - pX_time_start)))
    '''


if __name__ == "__main__":
    main()
