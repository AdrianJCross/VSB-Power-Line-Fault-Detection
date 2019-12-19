"""
COSC 522
Final Project - MPP
Adrian Cross, Xuesong Fan, and Aaron Wilson
"""

import numpy as np
import sys


def euc2(a, b):
    """euclidean distance square"""
    return np.dot(np.transpose(a - b), (a - b))


def mah2(a, b, sigma):
    """mahalanobis distance square"""
    return np.dot(np.transpose(a - b), np.dot(np.linalg.inv(sigma), (a - b)))


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


def main():
    Xtrain, ytrain = load_training('training_data_new.csv')
    Xtest = load_testing('test_data_new.csv')

    nXtrain, nXtest = norm(Xtrain, Xtest)
    pXtrain, pXtest = pca(nXtrain, nXtest, 0.1)

    model = mpp(2)
    model.fit(pXtrain, ytrain)
    ytrain_model = model.disc(pXtrain)
    eva(ytrain, ytrain_model)
    y_pseudo = model.disc(pXtest)

    pX = np.concatenate((pXtrain, pXtest))
    y = np.concatenate((ytrain, y_pseudo))

    model_whole = mpp(2)
    model_whole.fit(pX, y)
    ytrain_model2 =  model_whole.disc(pXtrain)
    eva(ytrain, ytrain_model2)
    y_model = model_whole.disc(pX)
    eva(y, y_model)


if __name__ == "__main__":
    main()
