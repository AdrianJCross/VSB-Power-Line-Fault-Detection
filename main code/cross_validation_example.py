import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

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

def main():

    Xtrain, ytrain = load_training('training_data_new.csv')

    n1 = np.count_nonzero(ytrain)
    n0 = len(ytrain) - n1

    Xtest = load_testing('test_data_new.csv')

    num_subsets = 10

    kf = KFold(n_splits=num_subsets)

    conf_mat = np.zeros([2, 2])
    acc = []

    for train_index, test_index in kf.split(Xtrain):

        Xtrain_batch, Xtest_batch = Xtrain[train_index], Xtrain[test_index]
        ytrain_batch, ytest_batch = ytrain[train_index], ytrain[test_index]

        # Train your model here using the 'Xtrain_batch' and 'ytrain_batch' data sets
        # Test the model using the 'Xtest_batch' and 'ytest_batch' data sets

        y_predicted = your_model(blah blah blah)

        conf_mat += confusion_matrix(ytest_batch, y_predicted)
        acc.append(100*(np.count_nonzero(ytest_batch == y_predicted) / len(ytest_batch)))

    # Confusion matrix here is of the form
    #   [TN  FP]
    #   [FN  TP]

    # Get confusion matrix in percentages
    conf_mat = 100 * conf_mat / np.array([n0, n1])
    accuracy = 100 * np.mean(acc)

if __name__ == "__main__":
    main()