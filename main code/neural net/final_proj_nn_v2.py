import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
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

def norm(Tr):
    """normalize the data"""
    m_ = np.mean(Tr, axis=0)
    sigma_ = np.std(Tr, axis=0)
    nTr = (Tr - m_) / sigma_
    # nTe = (Te - m_) / sigma_
    return nTr#, nTe

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
    # pTe = Te.dot(np.transpose(P_))
    return pTr#, pTe

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
    # fTe = Te.dot(np.transpose(w_))
    return fTr#, fTe

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

    Xtrain, ytrain = load_training('training_data_new.csv')
    Xtest = load_testing('test_data_new.csv')

    # # Uncomment the following line if a reduced training set size is needed
    # Xtrain, ytrain = reduce_training_size(Xtrain, ytrain, pw0=0.80)

    nXtrain = norm(Xtrain)
    nSize = len(nXtrain.T)
    pXtrain = pca(nXtrain, 0.1)
    pSize = len(pXtrain.T)
    fXtrain = fld(nXtrain, ytrain)
    fSize = 1

    num_neurons = 3

    num_hidden_layers = 1

    learning_rate = 3.0

    epochs = 30

    ksplit = 10

    kf = KFold(n_splits=ksplit)

    nX_time_start = time.time()
    nX_conf_mat = np.zeros([2, 2])

    count_nX = 0

    accs_nX = np.zeros(ksplit)
    y_models_n = [None] * ksplit
    y_tests_n = [None] * ksplit

    for train_index, test_index in kf.split(nXtrain):

        nXtrain_batch, nXtest_batch = nXtrain[train_index], nXtrain[test_index]
        ytrain_batch, y_tests_n[count_nX] = ytrain[train_index], ytrain[test_index]

        model_nX = keras.models.Sequential()
        model_nX.add(keras.layers.Dense(num_neurons, input_shape=((nSize,)), activation='sigmoid'))

        if num_hidden_layers > 1:
            for h in range(0, num_hidden_layers):
                model_nX.add(keras.layers.Dense(num_neurons, activation='sigmoid'))

        model_nX.add(keras.layers.Dense(1, activation='sigmoid'))

        # Comment out the below if confident in neural network size
        # The following two lines simply print out the size of each layer in the neural net
        for layer in model_nX.layers:
            print("Input shape: ", layer.input_shape[1], ", Output shape: ", layer.output_shape[1])

        simple_sgd = keras.optimizers.SGD(lr=learning_rate)
        model_nX.compile(optimizer=simple_sgd, loss='mean_squared_error', metrics=['accuracy'])

        hist = model_nX.fit(nXtrain_batch, ytrain_batch, epochs=epochs)

        # model_nX.evaluate(nXtest_batch, ytest_batch)

        y_models_n[count_nX] = model_nX.predict(nXtest_batch) > 0.5

        nX_conf_mat += confusion_matrix(y_tests_n[count_nX], y_models_n[count_nX])

        accs_nX[count_nX] = np.mean(hist.history.get('accuracy'))

        count_nX += 1

    nX_acc = np.mean(accs_nX)
    nX_sens = nX_conf_mat[1, 1]/(nX_conf_mat[1, 1] + nX_conf_mat[1, 0])
    nX_spec = nX_conf_mat[0, 0]/(nX_conf_mat[0, 0] + nX_conf_mat[0, 1])

    nX_time_end = time.time()

    pX_time_start = time.time()

    pX_conf_mat = np.zeros([2, 2])

    count_pX = 0

    accs_pX = np.zeros(ksplit)
    y_tests_p = [None] * ksplit
    y_models_p = [None] * ksplit

    for train_index, test_index in kf.split(pXtrain):

        pXtrain_batch, pXtest_batch = pXtrain[train_index], pXtrain[test_index]
        ytrain_batch, y_tests_p[count_pX] = ytrain[train_index], ytrain[test_index]

        model_pX = keras.models.Sequential()
        model_pX.add(keras.layers.Dense(num_neurons, input_shape=((pSize,)), activation='sigmoid'))

        if num_hidden_layers > 1:
            for h in range(0, num_hidden_layers):
                model_pX.add(keras.layers.Dense(num_neurons, activation='sigmoid'))

        model_pX.add(keras.layers.Dense(1, activation='sigmoid'))

        # Comment out the below if confident in neural network size
        # The following two lines simply print out the size of each layer in the neural net
        # for layer in model_nX.layers:
        #     print("Input shape: ", layer.input_shape[1], ", Output shape: ", layer.output_shape[1])

        simple_sgd = keras.optimizers.SGD(lr=learning_rate)
        model_pX.compile(optimizer=simple_sgd, loss='mean_squared_error', metrics=['accuracy'])

        hist = model_pX.fit(pXtrain_batch, ytrain_batch, epochs=epochs)

        # model_nX.evaluate(nXtest_batch, ytest_batch)

        y_models_p[count_pX] = model_pX.predict(pXtest_batch) > 0.5

        pX_conf_mat += confusion_matrix(y_tests_p[count_pX], y_models_p[count_pX])

        accs_pX[count_pX] = np.mean(hist.history.get('accuracy'))

        count_pX += 1

    pX_acc = np.mean(accs_pX)
    pX_sens = pX_conf_mat[1, 1]/(pX_conf_mat[1, 1] + pX_conf_mat[1, 0])
    pX_spec = pX_conf_mat[0, 0]/(pX_conf_mat[0, 0] + pX_conf_mat[0, 1])

    pX_time_end = time.time()

    fX_time_start = time.time()

    fX_conf_mat = np.zeros([2, 2])

    count_fX = 0

    accs_fX = np.zeros(ksplit)
    y_tests_f = [None] * ksplit
    y_models_f = [None] * ksplit

    for train_index, test_index in kf.split(fXtrain):

        fXtrain_batch, fXtest_batch = fXtrain[train_index], fXtrain[test_index]
        ytrain_batch, y_tests_f[count_fX] = ytrain[train_index], ytrain[test_index]

        model_fX = keras.models.Sequential()
        model_fX.add(keras.layers.Dense(num_neurons, input_shape=((fSize,)), activation='sigmoid'))

        if num_hidden_layers > 1:
            for h in range(0, num_hidden_layers):
                model_fX.add(keras.layers.Dense(num_neurons, activation='sigmoid'))

        model_fX.add(keras.layers.Dense(1, activation='sigmoid'))

        # Comment out the below if confident in neural network size
        # The following two lines simply print out the size of each layer in the neural net
        # for layer in model_fX.layers:
        #     print("Input shape: ", layer.input_shape[1], ", Output shape: ", layer.output_shape[1])

        simple_sgd = keras.optimizers.SGD(lr=learning_rate)
        model_fX.compile(optimizer=simple_sgd, loss='mean_squared_error', metrics=['accuracy'])

        hist = model_fX.fit(fXtrain_batch, ytrain_batch, epochs=epochs)

        # model_nX.evaluate(nXtest_batch, ytest_batch)

        y_models_f[count_fX] = model_fX.predict(fXtest_batch) > 0.5

        fX_conf_mat += confusion_matrix(y_tests_f[count_fX], y_models_f[count_fX])

        accs_fX[count_fX] = np.mean(hist.history.get('accuracy'))

        count_fX += 1

    fX_acc = np.mean(accs_fX)
    fX_sens = fX_conf_mat[1, 1]/(fX_conf_mat[1, 1] + fX_conf_mat[1, 0])
    fX_sens = fX_conf_mat[0, 0]/(fX_conf_mat[0, 0] + fX_conf_mat[0, 1])


    fX_time_end = time.time()

    print('nX accuracy: {}, time to complete: {} seconds, sensitivity {}%'.format(nX_acc, (nX_time_end-nX_time_start), nX_sens))
    print('pX accuracy: {}, time to complete: {} seconds, sensitivity {}%'.format(pX_acc, (pX_time_end-pX_time_start), pX_sens))
    print('fX accuracy: {}, time to complete: {} seconds, sensitivity {}%'.format(fX_acc, (fX_time_end-fX_time_start), fX_sens))


    # cross_val_tests = [y_tests_n, y_tests_f, y_tests_p]
    # cross_val_models = [y_models_n, y_models_f, y_models_p]
    # conf_mats = [nX_conf_mat, fX_conf_mat, pX_conf_mat]
    #
    # np.save('bpnn_reduced_tests', cross_val_tests)
    # np.save('bpnn_reduced_models', cross_val_models)
    # np.save('bpnn_reduced_cm', conf_mats)

    a = 1





if __name__ == "__main__":

    main()