# svinfer
The Statistical Valid Inference (svinfer) is a Python package which provides statistical models with valid inference for analyzing privacy protected data where carefully calibrated noises are injected to protect individuals' information.

Information regarding added noise is usually released together with a privacy protected data set. The structure of the noise usually remains the same if one selects a subset of the columns and/or a subset of the rows of privacy protected data. In contrast, the structure of the noise is likely to change if one aggregates privacy protected data (e.g. via the GROUP BY statement) after the noise is ingested. In the latter scenario, one needs to carefully derive the new distribution and variance of the noise and make a case-by-case decision on whether/how to apply the svinfer package. In cases where the noise variance is not constant across rows, the svinfer package should not be applied.

## Requirements
svinfer requires
* Python3 (>=3.7)
* numpy (>=1.18)
* pandas (>=1.0)
* sqlalchemy (>=1.2)

## Building svinfer
```
python3 setup.py sdist bdist_wheel
```

## Installing svinfer
```
pip install --upgrade svinfer-0.1.3-py3-none-any.whl
```

## How sinfer works
The LinearRegression method follows the methodology in this [paper](https://gking.harvard.edu/dpd).

## Examples
See the [examples](examples/).

## License
svifer is Apache-2.0 licensed, as found in the LICENSE file.
