from ionmf.factorization.model import iONMF
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=5)


def load_data():

    # Load data
    Xp = np.loadtxt(fname="../datasets/yeast_rpr_Xp.txt")   # Features; Positive differential expression
    Xn = np.loadtxt(fname="../datasets/yeast_rpr_Xn.txt")   # Features; Negative differential expression
    y0 = np.loadtxt(fname="../datasets/yeast_rpr_y0.txt")   # Class 0 binary vector
    y1 = np.loadtxt(fname="../datasets/yeast_rpr_y1.txt")   # Class 1 binary vector
    y2 = np.loadtxt(fname="../datasets/yeast_rpr_y2.txt")   # Class 2 binary vector

    return {
        "Pos_diff_expr": Xp,
        "Neg_diff_expr": Xn,
        "Class_0": y0.reshape((len(y0), 1)),
        "Class_1": y1.reshape((len(y1), 1)),
        "Class_2": y2.reshape((len(y2), 1)),
    }



def run():

    datadict = load_data()
    model = iONMF(rank=5, max_iter=100, alpha=0.0)

    # Fit all training data
    model.fit(datadict)

    # Make predictions about class on all training data
    # using only expression data ...
    testdict = dict()
    testdict["Pos_diff_expr"] = datadict["Pos_diff_expr"]
    testdict["Neg_diff_expr"]  = datadict["Neg_diff_expr"]
    rdict = model.predict(testdict)


    # ... and calculate training error
    true_y = np.zeros((len(datadict["Class_0"]), 1))
    true_y[np.where(datadict["Class_0"])] = 0
    true_y[np.where(datadict["Class_1"])] = 1
    true_y[np.where(datadict["Class_2"])] = 2

    predictions = np.array([np.argmax([rdict["Class_0"][i], rdict["Class_1"][i],
                                       rdict["Class_2"][i]])
                            for i in xrange(len(true_y))])

    acc = np.sum(predictions == true_y.ravel()) / float(len(true_y))
    print "Training accuracy: ", acc

    # Plot matrices
    plt.figure(figsize=(12, 12))
    for ki, ky in enumerate(testdict.keys()):
        plt.subplot(len(testdict), 2, 2*ki+1)
        plt.title(ky)
        plt.imshow(datadict[ky])
        plt.subplot(len(testdict), 2, 2*ki+2)
        plt.title(ky + " (approx.)")
        plt.imshow(model.coef_.dot(model.basis_[ky]))
    plt.show()

if __name__ == "__main__":
    run()

