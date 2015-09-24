from ionmf.factorization.model import iONMF
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip
import os
import sys
np.set_printoptions(precision=5)


def load_data(path, kmer=True, rg=True, clip=True, rna=True, go=True):
    """
        Load data matrices from the specified folder.
    """

    data = dict()
    if go:   data["X_GO"]   = pickle.load(gzip.open(os.path.join(path,
                                            "matrix_GeneOntology.pkl.gz")))
    if kmer: data["X_KMER"] = pickle.load(gzip.open(os.path.join(path,
                                            "matrix_RNAkmers.pkl.gz")))
    if rg:   data["X_RG"]   = pickle.load(gzip.open(os.path.join(path,
                                            "matrix_RegionType.pkl.gz")))
    if clip: data["X_CLIP"] = pickle.load(gzip.open(os.path.join(path,
                                            "matrix_Cobinding.pkl.gz")))
    if rna:  data["X_RNA"]  = pickle.load(gzip.open(os.path.join(path,
                                            "matrix_RNAfold.pkl.gz")))
    data["Y"] = pickle.load(gzip.open(os.path.join(path,
                                            "matrix_Response.pkl.gz")))
    return data


def load_labels(path, kmer=True, rg=True, clip=True, rna=True, go=True):
    """
        Load column labels for data matrices.
    """

    labels = dict()
    labels["X_GO"]   = pickle.load(gzip.open(os.path.join(path,
                                            "labels_GeneOntology.pkl.gz")))
    labels["X_KMER"] = pickle.load(gzip.open(os.path.join(path,
                                            "labels_RNAkmers.pkl.gz")))
    labels["X_RG"]   = pickle.load(gzip.open(os.path.join(path,
                                            "labels_RegionType.pkl.gz")))
    labels["X_CLIP"] = pickle.load(gzip.open(os.path.join(path,
                                            "labels_Cobinding.pkl.gz")))
    labels["X_RNA"]  = pickle.load(gzip.open(os.path.join(path,
                                            "labels_RNAfold.pkl.gz")))
    return labels


def run():

    # Select example protein folder from the dataset
    protein = sys.argv[1]

    # Load training data and column labels
    training_data = load_data("../datasets/clip/%s/2000/training_sample_0"
                              % protein,
                              go=False, kmer=False)
    training_labels = load_labels("../datasets/clip/%s/2000/training_sample_0"
                                  % protein,)
    model = iONMF(rank=5, max_iter=100, alpha=10.0)

    # Fit all training data
    model.fit(training_data)

    # Make predictions about class on all training data
    # delete class from dictionary
    test_data = load_data("../datasets/clip/%s/2000/test_sample_0" % protein,
                          go=False, kmer=False)
    true_y = test_data["Y"].copy()
    del test_data["Y"]
    results = model.predict(test_data)

    # Evaluate prediction on holdout test set
    predictions = results["Y"]
    auc = roc_auc_score(true_y, predictions)
    print "Test AUC: ", auc

    # Draw low-dimensional components for Region types (H_RG)
    # and RNA structure (H_RNA)
    # with mean values in coefficient matrix W for positive (+) and negative (-)
    # positions
    f, axes = plt.subplots(model.rank, 3, sharex='col',
                           figsize=(15, 8))
    H_RNA   = model.basis_["X_RNA"]
    H_RG    = model.basis_["X_RG"]
    labelset = sorted(set(training_labels["X_RG"]))

    positives = training_data["Y"].nonzero()[0]
    negatives = (training_data["Y"] == 0).nonzero()[0]
    for k in xrange(model.rank):

        # Values in the coefficient (W) matrix
        w_positives = model.coef_[positives, :][:, k].mean()
        w_negatives = model.coef_[negatives, :][:, k].mean()
        e_positives = model.coef_[positives, :][:, k].std() / np.sqrt(len(positives))
        e_negatives = model.coef_[negatives, :][:, k].std() / np.sqrt(len(negatives))
        axes[k, 2].bar([0], [w_negatives], yerr =[(0,), (e_positives, )],
                       color="blue", align="center")
        axes[k, 2].bar([1], [w_positives], yerr  =[(0,), (e_negatives, )],
                       color="green", align="center")

        # Plot RNA structure
        axes[k, 1].plot(H_RNA[k, :].ravel(),)

        # Plot region types
        for label in labelset:
            indices = np.where(map(lambda e: e == label, training_labels["X_RG"]))[0]
            axes[k, 0].plot(H_RG[k, indices].ravel(), label=label)
        axes[k, 0].set_ylabel("Module %d" % k)


    j = model.rank - 1
    axes[0, 0].legend(bbox_to_anchor=(0., 1.04, 1., .102), loc=3,
             ncol=3, mode="expand", borderaxespad=0.)
    axes[0, 1].set_title("Double-stranded RNA")
    axes[0, 2].set_title("Mean values in the coefficient matrix (W)")
    axes[j, 0].set_xticks(np.linspace(0, H_RNA.shape[1], 5))
    axes[j, 0].set_xticklabels([-50, -25, 0, 25, 50])
    axes[j, 0].set_xlabel("Position relative to cross-link site")
    axes[j, 1].set_xticks(np.linspace(0, H_RNA.shape[1], 5))
    axes[j, 1].set_xticklabels([-50, -25, 0, 25, 50])
    axes[j, 1].set_xlabel("Position relative to cross-link site")
    axes[j, 2].set_xticks([0, 1])
    axes[j, 2].set_xticklabels(["-", "+"])


    plt.show()
    

if __name__ == "__main__":
    run()

