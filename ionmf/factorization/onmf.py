from numpy.random import rand
from numpy import nan_to_num

def onmf(X, rank, alpha=1.0, max_iter=100, H_init=None, W_init=None):
        """
        Orthogonal non-negative matrix factorization.

        Parameters
        ----------
        X: array [m x n]
            Data matrix.
        rank: int
            Maximum rank of the factor model.
        alpha: int
            Orthogonality regularization parameter.
        max_iter: int
            Maximum number of iterations.
        H_init: array [rank x n]
            Fixed initial basis matrix.
        W_init: array [m x rank]
            Fixed initial coefficient matrix.

        Returns
        W: array [m x rank]
            Coefficient matrix (row clustering).
        H: array [rank x n]
            Basis matrix (column clustering / patterns).
        """

        m, n = X.shape
        W = rand(m, rank) if isinstance(W_init, type(None)) else W_init
        H = rand(rank, n) if isinstance(H_init, type(None)) else H_init

        for itr in xrange(max_iter):
            if isinstance(W_init, type(None)):
                enum = X.dot(H.T)
                denom = W.dot(H.dot(H.T))
                W = nan_to_num(W * enum/denom)

            if isinstance(H_init, type(None)):
                HHTH = H.dot(H.T).dot(H)
                enum = W.T.dot(X) + alpha * H
                denom = W.T.dot(W).dot(H) + 2.0 * alpha * HHTH
                H = nan_to_num(H * enum / denom)


        return W, H
