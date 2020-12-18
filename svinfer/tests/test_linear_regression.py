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

import unittest

import numpy as np
import sqlite3

from ..linear_model.linear_regression import LinearRegression
from ..processor.commons import DataFrameProcessor, DatabaseProcessor
from .utilities import check_if_almost_equal, simulate_test_data

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.data = simulate_test_data()

    def test_on_clear_data(self):
        """
        SVILinearRegegression should return the same results
        as a classic linear regression when features have no noise.
        The benchmark results are from
        sklearn.linear_model.LinearRegression.
        """
        df_data = DataFrameProcessor(self.data)
        result = (
            LinearRegression(["z1", "z2"], "y", [0.0, 0.0], fit_intercept=True)
            .fit(df_data)
            .beta
        )
        self.assertTrue(
            check_if_almost_equal(
                result,
                np.array([10.018132113216382, 11.994724266890941, -2.9988864645685593]),
                absolute_tolerance=1e-12,
                relative_tolerance=1e-12,
            )
        )

    def test_compare_with_Rcode(self):
        """
        LinearRegression should return the same results as the
        R package (https://github.com/georgieevans/PrivacyUnbiased).
        The benchmarks are from the R package when applying to the
        same data set. The estimate beta_vcov includes randomness.
        Thus, its comparison threshold is higher than the beta and
        sigma_sq, which are determined.
        """
        df_data = DataFrameProcessor(self.data)
        model = LinearRegression(
            ["x1", "x2"],
            "y",
            [2.0 ** 2, 1.0 ** 2],
            fit_intercept=True,
            df_corrected=True,
            n_replications=50000,
            random_state=1,
        ).fit(df_data)
        self.assertTrue(
            check_if_almost_equal(
                model.beta,
                np.array([10.077380244481004, 11.988784655257064, -3.002691057441349]),
                absolute_tolerance=1e-12,
                relative_tolerance=1e-8,
            )
        )
        self.assertTrue(
            check_if_almost_equal(model.sigma_sq, 2.892436195392406, 1e-8)
        )
        self.assertTrue(
            check_if_almost_equal(
                model.beta_vcov,
                np.array(
                    [
                        [
                            0.15603614740819050,
                            0.035350694019662628,
                            -0.017294841513339671,
                        ],
                        [
                            0.03535069401966263,
                            0.019901074328507087,
                            -0.007602090516864799,
                        ],
                        [
                            -0.01729484151333967,
                            -0.007602090516864799,
                            0.003068457648484046,
                        ],
                    ]
                ),
                absolute_tolerance=1e-12,
                relative_tolerance=1e-2,
            )
        )

    def test_compare_database_and_dataframe_version(self):
        """
        DatabaseProcessor and DataFrameProcessor should return
        the same intermediate results for linear regression,
        when giving the same training data,
        which will further ensure the same linear model results.
        """
        df_data = DataFrameProcessor(self.data)
        df_result = LinearRegression(
            ["x1", "x2"],
            "y",
            [2.0 ** 2, 1.0 ** 2],
            fit_intercept=True,
            df_corrected=True,
            n_replications=50000,
            random_state=1,
        )._preprocess_data(df_data)

        connection = sqlite3.connect(":memory:")
        self.data.to_sql("db_data", con=connection)
        db_data = DatabaseProcessor(connection, "db_data")
        db_result = LinearRegression(
            ["x1", "x2"],
            "y",
            [2.0 ** 2, 1.0 ** 2],
            fit_intercept=True,
            df_corrected=True,
            n_replications=50000,
            random_state=1,
        )._preprocess_data(db_data)

        for i in range(4):
            self.assertTrue(
                check_if_almost_equal(
                    df_result[i], db_result[i],
                    absolute_tolerance=1e-12,
                    relative_tolerance=1e-12,
                )
            )

    def test_compare_database_and_dataframe_version_with_filter(self):
        """
        DatabaseProcessor and DataFrameProcessor should return
        the same intermediate results for linear regression,
        when giving the same training data,
        which will further ensure the same linear model results.
        """
        filtered_data = self.data[(self.data["filter1"] == 1) & (self.data["filter2"] == 1)]
        df_data = DataFrameProcessor(filtered_data)
        df_result = LinearRegression(
            ["x1", "x2"],
            "y",
            [2.0 ** 2, 1.0 ** 2],
            fit_intercept=True,
            df_corrected=True,
            n_replications=50000,
            random_state=1,
        )._preprocess_data(df_data)

        connection = sqlite3.connect(":memory:")
        self.data.to_sql("db_data", con=connection)
        db_data = DatabaseProcessor(
            connection,
            "db_data",
            filters={"filter1": [1], "filter2": [1]}
        )
        db_result = LinearRegression(
            ["x1", "x2"],
            "y",
            [2.0 ** 2, 1.0 ** 2],
            fit_intercept=True,
            df_corrected=True,
            n_replications=50000,
            random_state=1,
        )._preprocess_data(db_data)

        for i in range(4):
            self.assertTrue(
                check_if_almost_equal(
                    df_result[i], db_result[i],
                    absolute_tolerance=1e-12,
                    relative_tolerance=1e-12,
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
