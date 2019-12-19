import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def fused_confusion_matrix(conf_mats, num_classifiers, num_classes):

    num_fused_cols = 2 ** num_classifiers
    classes = np.arange(num_classes)

    permutations = [p for p in product(classes, repeat=2)]

    fused_conf_mat = np.zeros([num_classes, num_fused_cols])

    n = np.sum(conf_mats[0], axis=0)

    priors = n / np.sum(conf_mats[0])

    conf_mats_pct = [p/n for p in conf_mats]

    for col in range(0, num_fused_cols):

        order = permutations[col]
        fused_conf_mat[:, col] = conf_mats_pct[0][:, order[0]]*conf_mats_pct[1][:, order[1]]

    # Create lookup table

    lookup = [permutations, np.argmax(fused_conf_mat, axis=0)]

    return lookup

def main():

    mpp_cross_val_tests = np.load('mpp_reduced_tests.npy', allow_pickle=True)
    mpp_cross_val_models = np.load('mpp_reduced_models.npy', allow_pickle=True)
    mpp_conf_mats = np.load('mpp_reduced_cm.npy', allow_pickle=True)

    bpnn_cross_val_tests = np.load('bpnn_reduced_tests.npy', allow_pickle=True)
    bpnn_cross_val_models = np.load('bpnn_reduced_tests.npy', allow_pickle=True)
    bpnn_conf_mats = np.load('bpnn_reduced_cm.npy', allow_pickle=True)

    # Choose the nX results (nX = 0, fX = 1, pX = 2)
    mpp_conf_mat_best = mpp_conf_mats[0]
    mpp_cross_val_models_best = mpp_cross_val_models[0]
    mpp_cross_val_tests_best = mpp_cross_val_tests[0]

    bpnn_conf_mat_best = bpnn_conf_mats[0]
    bpnn_cross_val_models_best = bpnn_cross_val_models[0]
    bpnn_cross_val_tests_best = bpnn_cross_val_tests[0]

    num_classifiers = 2

    num_classes = 2

    fused_conf_mat = fused_confusion_matrix([mpp_conf_mat_best, bpnn_conf_mat_best],
                                            num_classifiers=num_classifiers, num_classes=num_classes)

    fused_cross_val = [None] * len(mpp_cross_val_models_best)

    fused_accs = [None] * len(mpp_cross_val_models_best)

    total_size = int(np.sum(mpp_conf_mats[0]))

    for k in range(0, len(mpp_cross_val_models_best)):

        mpp_k = np.asarray(mpp_cross_val_models_best[k])
        bpnn_k = np.asarray(bpnn_cross_val_models_best[k])

        fused_labels = np.zeros(len(mpp_k))

        for j in range(0, len(mpp_k)):

            ordered_labels = (mpp_k[j], bpnn_k[j])
            fused_labels[j] = fused_conf_mat[1][fused_conf_mat[0].index(ordered_labels)]

        fused_cross_val[k] = fused_labels
        fused_accs[k] = 100 * ((np.count_nonzero(fused_labels[0] == mpp_cross_val_tests_best[0])) / len(mpp_k))

    acc = np.mean(fused_accs)

    print('Classifier fusion between MPP case 1 and BPNN produces an accuracy of {}%.'.format(acc))
    a = 1


if __name__ == "__main__":
    main()
