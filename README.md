# iONMF
Integrative orthogonal non-negative matrix factorization

An integrative approach to model and predict multiple data sources based on orthogonal matrix factorization. 
For details of the model, please refer to 
Stražar M., Žitnik M., Zupan B., Ule. J, Curk. T: Orthogonal matrix factorization enables integrative analysis of multiple RNA binding proteins
(to appear).


## Basic usage
The framework is simple to use within Python scripts. 

### Dataset preparation
The data is assumed to be stored in one or more Numpy arrays (numpy.ndarray), 
where rows represent samples and columns represent features. There can be multiple feature matrices as long as they
share the same number of rows.

The dataset is then stored as a dictionary

dataset = {
  "data_source_name_1": X1,
  "data_source_name_2": X2,
  # etc. ...
}

where the keys are data source names and X1, X2 represent matrices (Numpy arrays having the same number of rows.)




## Pre-prepared examples



### RNA-bining proteins dataset (CLIP)


### Yeast RPR dataset
