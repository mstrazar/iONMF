# iONMF
Integrative orthogonal non-negative matrix factorization

An integrative approach to model and predict multiple data sources based on orthogonal matrix factorization. 
For details of the model, please refer to 
Stražar M., Žitnik M., Zupan B., Ule. J, Curk. T: Orthogonal matrix factorization enables integrative analysis of multiple RNA binding proteins
(to appear).


## Basic usage
The framework is simple to use within Python scripts.  Assume the data is represented as matrices X1, X2, ... XN - data source, where rows represent samples and columns represent features. There can be multiple feature matrices as long as they
share the same number of rows. 

An iONMF model approximates each matrix X_i with a matrix product W H_i, such that
X_1 ~ W H_1
X_2 ~ W H_2
...
X_N ~ W H3
where W, and H1, H2, ... HN are non-negative matrices of lower ranks and their matrix product approximates the data sources X1, X2, ... XN.

The common coefficient matrix W is common to all data sources and represents clustering of rows, while each basis matrix H1, H2, ... HN represents clustering of columns. Due to non-negativity constraints, the rows of H can be interpreted as commonly occuring combinations of features. 

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

Prepared practical examples for usage of the model are available.

### Yeast RPR dataset
See ionmf.example.yeast_rpr.py 
A simple example of running iONMF on a differential gene expression dataset. The dataset contains 186 samples and 79 genes, divided into 3 classes. 

Some values in the matrix X contain negative values. The following trick is used. The number of columns is doubled to form the matrices Xp, Xn each of size (186 rows, 79 columns). All positive values are stored in Xp, while absolute values of negative cells are stored in Xn.

Additional three 186 x 1 matrices are stored to represent the three classes (value 1 if the sample i belongs to the class).

In this demonstrative example, the model is trained on the whole dataset. In the test phase, the datasources representing classes are removed from the model. After prediction, each sample is assigne to class 0, 1, or 2 depending on maximum value in approximated columns class_0, class_1, class_2. The training accuracy is measured as the fraction of correctly classified examples.

Each matrix X is plotted along with the approximation W H. Increasing the maximum model rank would yield more accurate approximations.

### RNA-bining proteins dataset (CLIP)



