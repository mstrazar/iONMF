from onmf import onmf
from numpy import zeros, hstack
from warnings import filterwarnings

class iONMF:
    """ Integrative orthogonal non-negative matrix factorization.

    Parameters
    ----------

    data: dictionary of array describing samples with multiple data sources.
        data = {
            "data_source_1": X_1 array [n_samples, n_features_1],
            "data_source_2": X_2 array [n_samples, n_features_2],
            ...
            "data_source_N": X_N array [n_samples, n_features_N],
        }
        Data sources must match in the number of rows.

    alpha: float, optional [default=1.0]
        Orthofonality regularization parameter.

    rank: int, optional [default=10]
        Maximum rank of basis (H) and coefficient matrices (W).

    random_state: int, optional [default=None]
        The seed used to initialize the model prior to learning.
        Defaults to numpy.random.

    Returns
    -------
    coef_: array [n_core_samples]
        Coefficient matrix (row clustering).
        W array [n_samples, rank]

    basis_ : array [n_samples]
        Basis matrices, one per data source (column clustering / patterns).
        data = {
            "data_source_1": H_1 array [rank, n_features_1],
            "data_source_2": H_2 array [rank, n_features_2],
            ...
            "data_source_N": H_N array [rank, n_features_N],
        }

    Notes
    -----
    See examples/ for an example.

    """

    def __init__(self, rank=10, max_iter=100, alpha=1.0):
        filterwarnings("ignore")
        self.coef_        = None
        self.basis_       = None
        self.keys_        = None
        self.n_           = None
        self.rank         = rank
        self.max_iter     = max_iter
        self.alpha        = alpha
        self.instantiated = False


    def fit(self, data):
        """ Fit the iONMF model.

        Parameters
        ----------

        data: dictionary of array describing samples with multiple data sources.
        data = {
            "data_source_1": X_1 array [n_samples, n_features_1],
            "data_source_2": X_2 array [n_samples, n_features_2],
            ...
            "data_source_N": X_N array [n_samples, n_features_N],
        }
        Data sources must match in the number of rows.
        """

        self.keys_ = sorted(data.keys())
        self.n_    = [data[ky].shape[1] for ky in self.keys_]
        self.m     = data[self.keys_[0]].shape[0]
        if not all([data[ky].shape[0] == self.m for ky in self.keys_]):
            raise ValueError("The number of rows must match for all matrices!")

        # Fill training data matrix
        X          = zeros((self.m, sum(self.n_)))
        t          = 0
        for ny, ky in zip(self.n_, self.keys_,):
            X[:, t:t+ny] = data[ky]
            t += ny

        # Run factorization
        W, H  = onmf(X, rank=self.rank,
                     max_iter=self.max_iter,
                     alpha=self.alpha)

        # Set model variables
        self.coef_  = W
        self.basis_ = dict()
        t           = 0
        for ny, ky in zip(self.n_, self.keys_):
            self.basis_[ky] = H[:, t:t+ny]
            t += ny

        self.instantiated = True


    def predict(self, data_test):
        """ Predict the values for test samples based on a (non-empty)
        subset of avalible data sources .

        Parameters
        ----------

        data_test: dictionary of array describing samples with multiple data sources.
        data_test = {
            "data_source_1": X_1 array [n_samples, n_features_1],
            "data_source_2": X_2 array [n_samples, n_features_2],
            ...
            "data_source_N": X_N array [n_samples, n_features_N],
        }
        Test data sources must match in the number of rows.
        Dictionary keys must match keys in the traning data.
        """

        if not self.instantiated:
            raise ValueError("Run the method iONMF.fit first!")

        keys_test = sorted(data_test.keys())
        if set(keys_test) - set(self.keys_):
            raise ValueError("Test data dictionary contains unknown keys!")

        m_test = len(data_test[keys_test[0]])
        if not all([data_test[ky].shape[0] == m_test for ky in keys_test]):
            raise ValueError("The number of rows must match for all matrices!")

        X_test = hstack([data_test[ky] for ky in keys_test])
        H_test = hstack([self.basis_[ky] for ky in keys_test])

        # Infer values of W for test samples based on a non-empty subset of
        #   observed data sources (matrices)
        W_test, _ = onmf(X_test, rank=self.rank, alpha=self.alpha,
                      H_init=H_test)

        # Predict all remaining matrices
        remaining_keys = set(self.keys_) - set(keys_test)
        predictions = dict([(ky, W_test.dot(self.basis_[ky]))
                            for ky in remaining_keys])

        return predictions
