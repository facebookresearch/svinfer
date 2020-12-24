# LogisticRegression
```
class LogisticRegression(
    x_columns, y_column, x_s2, fit_intercept=True
)
```
Statistically valid inference for logistic regression. The class LogisticRegression fits a logistic regression where some predictors in the training data are subject to measurement errors.
The response variable is coded as 0/1.

## Parameters
- **x_columns**: *list of strings*.
Specify the column names for the features in the data.
- **y_column**: *string*.
Specify the column name for the response in the data.
- **x_s2**: *list of doubles*.
Specify the variance of noise that was added to each of the features. It’s length should be equal to that of x_columns. If no noise is added to a feature, use 0.0 in the corresponding place.
- **fit_intercept**: *bool, optional, default to True*.
Whether to include the intercept into the model. If set to False, the design matrix will only consist of the features specified. If set to True, a column of 1 will be added to the design matrix besides features specified.

## Attributes
- **beta**: *array of shape (n_features, )*
The vector of regression coefficients. If fit_intercept is set to True, the first element of beta is the intercept, followed by other regression coefficients.
- **beta_vcov**: *array of shape (n_features, n_features)*
The estimated variance-covariance matrix of the regression coefficients.
- **beta_standarderror**: *array of the shape (n_features,)*
The standard errors of regression coefficients.
- **success**: *bool*.
Whether the model fitting is successful. Do not use the model results when False.

## Method
- **fit(self, data)**: Fit the linear regression.
    - Parameter:
        - **data**: an instance of `AbstractProcessor`. Usually created by `DataFrameProcessor` or `DatabaseProcessor`.
    - Return:
        - **self**

## Example
See [this example](../examples/logistic_regression_with_dataframe.py) assuming the training data is a `pandas.DataFrame` in memory.
See [this example](../examples/logistic_regression_with_database.py) assuming the training data is a table in a database.
