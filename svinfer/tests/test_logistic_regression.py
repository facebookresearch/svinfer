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

import statsmodels.api as sm

from ..linear_model.logistic_regression import LogisticRegression
from .utilities import check_if_almost_equal, simulate_test_data


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.data = simulate_test_data()

    def test_on_clear_data(self):
        """
        When applied to non-noisy data, the svinfer's LogisticRegegression
        is expected to return the same results as a classic logistic regression.
        The benchmark results are from statsmodels.api.GLM
        """
        predictors = ["z1", "z2"]
        response = "y_binary"
        # fit by svinfer
        svinfer_model = LogisticRegression(predictors, response, [0, 0]).fit(self.data)
        svinfer_beta = svinfer_model.beta
        svinfer_vcov = svinfer_model.beta_vcov

        # fit by statsmodels
        sm_model = sm.GLM(
            self.data[response].values,
            sm.add_constant(self.data[predictors].values),
            family=sm.families.Binomial(),
        ).fit(
            cov_type="HC0"
        )  # use basic sandwich
        sm_beta = sm_model.params
        sm_vcov = sm_model.cov_params()

        self.assertTrue(
            check_if_almost_equal(
                svinfer_beta,
                sm_beta,
                absolute_tolerance=1e-12,
                relative_tolerance=1e-12,
            )
        )

        self.assertTrue(
            check_if_almost_equal(
                svinfer_vcov,
                sm_vcov,
                absolute_tolerance=1e-12,
                relative_tolerance=1e-12,
            )
        )

    def test_dataframe_version(self):
        """
        When applied to noisy data, the svifner's LogisticRegegression
        is expected to provide confidence interval that covers the truth
        in most of the time.
        """
        predictors = ["x1", "x2"]
        response = "y_binary"
        x_s2 = [2.0 ** 2, 1.0 ** 2]
        beta_true = [1, 2, -0.5]

        model = LogisticRegression(predictors, response, x_s2).fit(self.data)
        beta, se = model.beta, model.beta_standarderror
        ci_lower = beta - 1.96 * se
        ci_upper = beta + 1.96 * se

        self.assertTrue(((ci_lower < beta_true) & (beta_true < ci_upper)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
