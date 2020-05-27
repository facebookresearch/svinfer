# LinearRegression
```
class LinearRegression(
    x_columns, y_column, x_s2,
    fit_intercept=True, df_corrected=True, n_replications=500,
    random_state=None,
)
```
Statistically valid inference for linear regression. The class LinearRegression fits a linear regression where some variables in the training data are subject to measurement errors. It uses a computationally efficient algorithm for large datasets.

## Parameters
- **x_columns**: *list of strings*.
Specify the column names for the features in the data.
- **y_column**: *string*.
Specify the column name for the response in the data.
- **x_s2**: *list of doubles*.
Specify the variance of noise that was added to each of the features. It’s length should be equal to that of x_columns. If no noise is added to a feature, use 0.0 in the corresponding place.
- **fit_intercept**: *bool, optional, default to True*.
Whether to include the intercept into the model. If set to False, the design matrix will only consist of the features specified. If set to True, a column of 1 will be added to the design matrix besides features specified.
- **df_corrected**: *bool, optional, default to True*.
Whether to adjust the degree of freedom when estimating the error variance. If set to False, the degree of freedom is n, where n is the sample size. If set to True, the degree of freedom is n - p, where p is the number of columns in the design matrix.
- **n_replications**: *int, optional, default to 500*.
The number of simulation replicates. It should be a positive integer.
- **random_state**: *int or None, optional, default to None*.
Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls.

## Attributes
- **beta**: *array of shape (n_features, )*
The vector of regression coefficients. If fit_intercept is set to True, the first element of beta is the intercept, followed by other regression coefficients.
- **sigma_sq**: *double*
Estimated error variance for the error term.
- **beta_vcov**: *array of shape (n_features, n_features)*
The estimated variance-covariance matrix of the regression coefficients.
- **beta_standarderror**: *array of the shape (n_features,)*
The standard errors of regression coefficients.

## Method
- **fit(self, data)**: Fit the linear regression.
    - Parameter:
        - **data**: an instance of `AbstractProcessor`. Usually created by `DataFrameProcessor` or `DatabaseProcessor`.
    - Return:
        - **self**

## Example
See [this example](../examples/linear_regression_with_dataframe.py) assuming the training data is a `pandas.DataFrame` in memory.
See [this example](../examples/linear_regression_with_database.py) assuming the training data is a table in a database.
