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
is stored as a table in SQLite database.
"""

from svinfer.processor import DatabaseProcessor
from svinfer.linear_model import LinearRegression

import sqlite3
from linear_regression_with_dataframe import simulate_training_data


if __name__ == "__main__":
    # get training data
    # assume the variance of the added noise are 4 and 1 for each predictor
    # assume the training data is stored as a table called my_data in SQLite database
    x_s2 = [4, 1]
    data = simulate_training_data(x_s2)
    connection = sqlite3.connect(":memory:")
    data.to_sql("my_data", con=connection)

    # fit y ~ x1 + x2, where x1 and x2 have added noise
    db_data = DatabaseProcessor(connection, "my_data")
    model = LinearRegression(
        ["x1", "x2"],  # column names for predictors
        "y",  # column name for the response
        x_s2,  # variances of the added noises to each predictor
        random_state=123,  # optional, to ensure reproducibility
    ).fit(db_data)

    # check result
    print("beta_tilde is: \n{}".format(model.beta))
    # expect results to be close to
    # beta_tilde is:
    # [10.53475783 12.26662045 -3.11457588]
    print("beta_tilde's standard error is: \n{}".format(model.beta_standarderror))
    # expect results to be close to
    # beta_tilde's standard error is:
    # [1.28940235 0.45779356 0.17814397]
    print("beta_tile's variance-covariance matrix: \n{}".format(model.beta_vcov))
    # expect results to be close to
    # beta_tile's variance-covariance matrix:
    # [[1.66255843  0.35312458 -0.17656444]
    #  [0.35312458  0.20957495 -0.07915853]
    #  [-0.17656444 -0.07915853 0.03173527]]
    print("estimated residual variance is {}".format(model.sigma_sq))
    # expect results to be close to
    # estimated residual variance is 0.5136891806650965
