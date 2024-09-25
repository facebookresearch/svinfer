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
Illustrate how to get summary statistics for x1 and x2 where x1, x2 contains
designed noise, when training data is a pandas DataFrame in memory.
"""

import numpy as np
import pandas as pd
from svinfer.processor import DataFrameProcessor
from svinfer.summary_statistics import SummaryStatistics


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

    # get summary statistics for x1 and x2, where x1 and x2 have added noise
    df_data = DataFrameProcessor(data)
    result = SummaryStatistics(
        ["x1", "x2"],  # column names for features of interest
        x_s2,  # variances of the added noises to each feature
    ).estimate_summary_statistics(df_data)

    # check result
    print(f"summary Statistics for x1 and x2 are: \n{result.summary_statistics}")
    # expect results to be
    #     summary Statistics for x1 and x2 are:
    #         average  standard_deviation  skewness  kurtosis
    #     x1   7.005687            2.611832  0.481705  3.449543
    #     x2  23.042510            6.071953  0.303291  3.174952
