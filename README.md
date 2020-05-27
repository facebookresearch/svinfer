# svinfer
The Statistical Valid Inference (svinfer) is a Python package which provides statistical models with valid inference for analyzing privacy protected data where carefully calibrated noises are injected to protect individuals' information.

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
