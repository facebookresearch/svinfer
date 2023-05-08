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

import sqlite3
import unittest

import pandas as pd

# import numpy as np
from scipy import stats

from ..processor.commons import DatabaseProcessor, DataFrameProcessor
from ..summary_statistics.summary_statistics import SummaryStatistics
from .utilities import check_if_almost_equal, simulate_test_data


class TestSummaryStatistics(unittest.TestCase):
    def setUp(self):
        self.data = simulate_test_data()

    def test_compare_with_python_buildin(self):
        """
        The SummaryStatistics should return identical results to what the off-the-shelf functions
        return, when apply to the non-noisy data.
        """
        df_data = DataFrameProcessor(self.data)
        # version 1: does not correct bias in skewness and kurtosis
        estimator1 = (
            SummaryStatistics(["z1", "z2"], [0, 0])
            .estimate_summary_statistics(df_data)
            .summary_statistics
        )
        truth1 = pd.DataFrame(
            {
                "average": self.data[["z1", "z2"]].mean(),
                "standard_deviation": [
                    stats.tstd(self.data["z1"]),
                    stats.tstd(self.data["z2"]),
                ],
                "skewness": [stats.skew(self.data["z1"]), stats.skew(self.data["z2"])],
                "kurtosis": [
                    stats.kurtosis(self.data["z1"], fisher=False),
                    stats.kurtosis(self.data["z2"], fisher=False),
                ],
            }
        )

        for s in ["average", "standard_deviation", "skewness", "kurtosis"]:
            self.assertTrue(
                check_if_almost_equal(
                    estimator1[s],
                    truth1[s],
                    absolute_tolerance=1e-12,
                    relative_tolerance=1e-12,
                )
            )

        # version 2: correct bias in skewness and kurtosis
        estimator2 = (
            SummaryStatistics(["z1", "z2"], [0, 0], bias=False)
            .estimate_summary_statistics(df_data)
            .summary_statistics
        )
        truth2 = pd.DataFrame(
            {
                "average": self.data[["z1", "z2"]].mean(),
                "standard_deviation": [
                    stats.tstd(self.data["z1"]),
                    stats.tstd(self.data["z2"]),
                ],
                "skewness": [
                    stats.skew(self.data["z1"], bias=False),
                    stats.skew(self.data["z2"], bias=False),
                ],
                "kurtosis": [
                    stats.kurtosis(self.data["z1"], bias=False, fisher=False),
                    stats.kurtosis(self.data["z2"], bias=False, fisher=False),
                ],
            }
        )

        for s in ["average", "standard_deviation", "skewness", "kurtosis"]:
            self.assertTrue(
                check_if_almost_equal(
                    estimator2[s],
                    truth2[s],
                    absolute_tolerance=1e-12,
                    relative_tolerance=1e-12,
                )
            )

    def test_compare_noisy_data_and_clean_data(self):
        """
        The SummaryStatistics should debias the summary statistics when applied to noisy data.
        The results should be similar to what the off-the-shelf functions provide when applied
        to the underlying non-noisy data.
        """
        df_data = DataFrameProcessor(self.data)
        estimator = (
            SummaryStatistics(["x1", "x2"], [2.0**2, 1.0**2])
            .estimate_summary_statistics(df_data)
            .summary_statistics
        )
        truth = pd.DataFrame(
            {
                "average": self.data[["z1", "z2"]].mean(),
                "standard_deviation": [
                    stats.tstd(self.data["z1"]),
                    stats.tstd(self.data["z2"]),
                ],
                "skewness": [stats.skew(self.data["z1"]), stats.skew(self.data["z2"])],
                "kurtosis": [
                    stats.kurtosis(self.data["z1"], fisher=False),
                    stats.kurtosis(self.data["z2"], fisher=False),
                ],
            }
        )
        estimator.reset_index(drop=True, inplace=True)
        truth.reset_index(drop=True, inplace=True)
        for s in ["average", "standard_deviation", "skewness", "kurtosis"]:
            self.assertTrue(
                check_if_almost_equal(
                    estimator[s],
                    truth[s],
                    absolute_tolerance=1e-12,
                    relative_tolerance=1e-1,
                )
            )

    def test_compare_database_and_dataframe_version(self):
        """
        The DatabaseProcessor and DataFrameProcessor should return the same intermediate
        results for SummaryStatistics, when applied the same training data. It will further
        ensure the same final results.
        """
        df_data = DataFrameProcessor(self.data)
        df_moments, df_n = SummaryStatistics(
            ["x1", "x2"], [2.0**2, 1.0**2]
        )._preprocess_data(df_data)

        connection = sqlite3.connect(":memory:")
        self.data.to_sql("db_data", con=connection)
        db_data = DatabaseProcessor(connection, "db_data")
        db_moments, db_n = SummaryStatistics(
            ["x1", "x2"], [2.0**2, 1.0**2]
        )._preprocess_data(db_data)

        for i in range(2):
            self.assertTrue(
                check_if_almost_equal(
                    df_moments[i],
                    db_moments[i],
                    absolute_tolerance=1e-12,
                    relative_tolerance=1e-12,
                )
            )
        self.assertTrue(
            check_if_almost_equal(
                df_n, db_n, absolute_tolerance=1e-12, relative_tolerance=1e-12
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
