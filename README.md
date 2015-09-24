# iONMF
Integrative orthogonal non-negative matrix factorization

An integrative approach to model and predict multiple data sources based on orthogonal matrix factorization. 
For details of the model, please refer to 
Stražar M., Žitnik M., Zupan B., Ule. J, Curk. T: Orthogonal matrix factorization enables integrative analysis of multiple RNA binding proteins
(to appear).


## Basic usage
The framework is simple to use within Python scripts.  Assume the data is represented as matrices X1, X2, ... XN - data source, where rows represent samples and columns represent features. There can be multiple feature matrices as long as they
share the same number of rows. 

The iONMF models approximates each matrix such that
X1 = WH1
X2 = WH2
...
X2 = WH3

where W, and H1, H2, ... HN are non-negative matrices of lower ranks and their product approximates the data sources X1, X2, ... XN.
The common coefficient matrix W is common to all data sources and represents clustering of rows, while each basis matrix H1, H2, ... HN represents clustering of columns. Due to non-negativty constraints, the rows of H can be interpreted as commonly occuring combinations of features. 

Such model can be used to provide and interpetation of the dataset (unsupervised learning; clustering of rows and most commonly occuring features) or prediction (supervised learning), where one or more datasource is missing for a set of test samples, provided at least one X is available. 


### Dataset preparation
The data is assumed to be stored in one or more Numpy arrays (numpy.ndarray) and contain only non-negative values.
The dataset is then stored as a dictionary

dataset = {
  "data_source_name_1": X1,
  "data_source_name_2": X2,
  "data_source_name_3": X3,
}

where the keys are data source names and X1, X2 represent matrices (Numpy arrays having the same number of rows.)

### Running the matrix factorization model

The model is initizalized as a class as follows
from ionmf.factorization.model import iONMF
model = iONMF(rank=5, max_iter=100, alpha=1.0)

where rank is the maximum rank of the low-rank approximation, max_iter the number of iterations during optimization and alpha the orthogonality regularization. Higher alpha yields more distinct (orthogonal) basis vectors.

Having prepared the dataset, the model is trained as follows:
model.fit(dataset)

THe low-rank approximations can be accessed e.g.
model.basis_["data_source_name_1"]  # Get the basis (H) matrix for data_source_name_1
model.coef_                         # Get the coefficient (W) matrix


Now suppose we have a another set of samples, where one or more data sources are missing.
testset = {
  "data_source_name_1": Y1,
  "data_source_name_3": Y3,
}
In this example, the data_source_name_2 is missing. Again, Y1...Y3 must share the same number of rows. Having trained a model on the previous (training) dataset, 
the missing data sources can be filled in.

results = model.predict(testset)

The result is a dictionary results that contains approximations to all data sources that were missing from testset.
Y2 = results["data_source_name_2"]




## Pre-prepared examples



### RNA-bining proteins dataset (CLIP)


### Yeast RPR dataset
