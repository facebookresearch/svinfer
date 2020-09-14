# SummaryStatistics
```
class SummaryStatistics(columns, x_s2, bias=True)
```
The class SummaryStatistics provides summary statistics for each of the feature columns where the data may have measurement errors.

## Parameters
- **x_columns**: *list of strings*.
Specify the column names for the features in the data.
- **x_s2**: *list of doubles*.
Specify the variance of noise that was added to each of the features. It’s length should be equal to that of x_columns. If no noise is added to a feature, use 0.0 in the corresponding place.
- **bias**: *bool, optional, default to True*.
If False, the calculations of skewness and kurtosis are corrected for bias. That is, the skewness is computed sd the adjusted Fisher-Pearson standardized moment coefficientthe, while the kurtosis is calculated using k statistics to eliminate bias coming from biased moment estimators.

## Attributes
- **summary_statistics**: *pandas.DataFrame of shape (n_features, 4)*
Summary statistics for each feature, including average, standard deviation, skewness and kurtosis.

## Method
- **estimate_summary_statistics(self, data, bias)**: Fit the linear regression.
    - Parameter:
        - **data**: an instance of `AbstractProcessor`. Usually created by `DataFrameProcessor` or `DatabaseProcessor`.
    - Return:
        - **self**

## Example
See [this example](../examples/summary_statistics_with_dataframe.py) assuming the training data is a `pandas.DataFrame` in memory.
See [this example](../examples/summary_statistics_with_database.py) assuming the training data is a table in a database.
