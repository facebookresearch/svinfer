#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Illustrate how to run linear model (y ~ x1 + x2) with statistically
valid inference when x1, x2 contains designed noise, when training data
is a pandas DataFrame in memory.
"""

import numpy as np
import pandas as pd
from svinfer.linear_model import LinearRegression
from svinfer.processor import DataFrameProcessor


def simulate_training_data(x_s2):
    # generate independent variables
    # where the random noise is added to the
    n = 10000
    np.random.seed(0)
    z1 = np.random.poisson(lam=7, size=n)
    z2 = np.random.poisson(lam=9, size=n) + 2 * z1

    # generate y based on z1, z2
    # add noise ~ N(0, 2^2) to independent variable z1
    # add noise ~ N(0, 1^2) to independent variable z2
    # generate training data
    data = pd.DataFrame(
        {
            "y": 10 + 12 * z1 - 3 * z2 + 2 * np.random.standard_normal(size=n),
            "x1": z1 + np.random.standard_normal(size=n) * np.sqrt(x_s2[0]),
            "x2": z2 + np.random.standard_normal(size=n) * np.sqrt(x_s2[1]),
        }
    )
    return data


if __name__ == "__main__":
    # get training data
    # assume the variance of the added noise are 4 and 1 for each predictor
    x_s2 = [4, 1]
    data = simulate_training_data(x_s2)

    # fit y ~ x1 + x2, where x1 and x2 have added noise
    df_data = DataFrameProcessor(data)
    model = LinearRegression(
        ["x1", "x2"],  # column names for predictors
        "y",  # column name for the response
        x_s2,  # variances of the added noises to each predictor
    ).fit(df_data)

    # check result
    print(f"beta_tilde is: \n{model.beta}")
    # expected results should be close to
    # beta_tilde is:
    # [10.53475783 12.26662045 -3.11457588]
    print(f"beta_tilde's standard error is: \n{model.beta_standarderror}")
    # expected results should be close to
    # beta_tilde's standard error is:
    # [1.29629034 0.49465959 0.19243032]
    print(f"beta_tile's variance-covariance matrix: \n{model.beta_vcov}")
    # expected results should be close to
    # beta_tile's variance-covariance matrix:
    # [[ 1.68036864  0.42095858 -0.19852479]
    # [ 0.42095858  0.24468811 -0.0930343 ]
    # [-0.19852479 -0.0930343   0.03702943]]
    print(f"estimated residual variance is {model.sigma_sq}")
    # expected results should be close to
    # estimated residual variance is 0.5136891806650965
