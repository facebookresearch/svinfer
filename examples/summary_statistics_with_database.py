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
designed noise, when training data is stored as a table in SQLite database.
"""

import sqlite3

from summary_statistics_with_dataframe import simulate_training_data
from svinfer.processor import DatabaseProcessor
from svinfer.summary_statistics import SummaryStatistics


if __name__ == "__main__":
    # get training data
    # assume the variance of the added noise are 4 and 1 for each predictor
    # assume the training data is stored as a table called my_data in SQLite database
    x_s2 = [4, 1]
    data = simulate_training_data(x_s2)
    connection = sqlite3.connect(":memory:")
    data.to_sql("my_data", con=connection)

    # get summary statistics for x1 and x2, where x1 and x2 have added noise
    db_data = DatabaseProcessor(connection, "my_data")
    result = SummaryStatistics(
        ["x1", "x2"],  # column names for features of interest
        x_s2,  # variances of the added noises to each feature
    ).estimate_summary_statistics(db_data)

    # check result
    print(
        "summary Statistics for x1 and x2 are: \n{}".format(result.summary_statistics)
    )
    # expect results to be:
    # summary Statistics for x1 and x2 are:
    #     average  standard_deviation  skewness  kurtosis
    # x1   7.005687            2.611832  0.481705  3.449543
    # x2  23.042510            6.071953  0.303291  3.174952
