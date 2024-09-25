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
Illustrate how to run logistic model (y_binary ~ x1 + x2) with statistically
valid inference when x1, x2 contains designed noise, when training data
is a pandas DataFrame in memory.
"""

import numpy as np
import pandas as pd
from scipy import special, stats
from svinfer.linear_model import LogisticRegression
from svinfer.processor import DataFrameProcessor


def simulate_training_data(n=100000, seed=0):
    # generate independent variables
    # where the random noise will be added later
    np.random.seed(seed)
    z1 = np.random.poisson(lam=7, size=n)
    z2 = np.random.poisson(lam=9, size=n) + (2 * z1)
    # generate y_binary based on z1, z2
    # add noise ~ N(0, 2^2) to independent variable z1
    # add noise ~ N(0, 1^2) to independent variable z2
    # generate training data
    data = pd.DataFrame(
        {
            "x1": z1 + np.random.standard_normal(size=n) * 2.0,
            "x2": z2 + np.random.standard_normal(size=n) * 1.0,
            "y_binary": stats.bernoulli.rvs(special.expit(1 + 2 * z1 - 0.5 * z2)),
        }
    )
    return data


if __name__ == "__main__":
    # get training data
    data = simulate_training_data()
    # specify that variance of the added noise to each predictor is 4 and 1 respectively
    x_s2 = [4, 1]

    # fit y_binary ~ x1 + x2, where x1 and x2 have added noise
    df_data = DataFrameProcessor(data)
    model = LogisticRegression(
        ["x1", "x2"],  # column names for predictors
        "y_binary",  # column name for the response
        x_s2,  # variances of the added noises to each predictor
    ).fit(df_data)

    # check result
    print(f"The regression coefficients are: \n{model.beta}")
    # expected results should be close to
    # The regression coefficients are:
    # [ 0.95180511  1.99898351 -0.49728914]
    print(
        "The standard error of the regression coefficients are: \n{}".format(
            model.beta_standarderror
        )
    )
    # expected results should be close to
    # The standard error of the regression coefficients are is:
    # [0.2237052  0.18448407 0.05588327]
    print(
        "The estimated variance-covariance matrix of the regression coefficients are: \n{}".format(
            model.beta_vcov
        )
    )
    # expected results should be close to
    # The estimated variance-covariance matrix of the regression coefficients are:
    # [[ 0.05004401  0.02866712 -0.00977895]
    #  [ 0.02866712  0.03403437 -0.01021528]
    #  [-0.00977895 -0.01021528  0.00312294]]
