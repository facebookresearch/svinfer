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

from ..processor.commons import DataFrameProcessor
from ..linear_model.logistic_regression import LogisticRegression
from .utilities import (
    check_if_almost_equal,
    simulate_test_data,
    simulate_test_data_misspecified_model,
)


class Wrapper:
    class BaseTestLogisticRegression(unittest.TestCase):
        def __init__(self, *args, **kwargs):
            super(Wrapper.BaseTestLogisticRegression, self).__init__(*args, **kwargs)
            self.data = None
            self.predictors_clear = None
            self.predictors_noisy = None
            self.response = None
            self.x_s2 = None
            self.beta_true = None

        def test_on_clear_data(self):
            """
            When applied to non-noisy data, the svinfer's LogisticRegression
            is expected to return the same results as a classic logistic regression.
            The benchmark results are from statsmodels.api.GLM
            """
            # fit by svinfer
            svinfer_model = LogisticRegression(
                self.predictors_clear, self.response, [0] * len(self.predictors_clear)
            ).fit(DataFrameProcessor(self.data))
            svinfer_beta = svinfer_model.beta
            svinfer_vcov = svinfer_model.beta_vcov

            # fit by statsmodels
            sm_model = sm.GLM(
                self.data[self.response].values,
                sm.add_constant(self.data[self.predictors_clear].values),
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
            When applied to noisy data, the svinfer's LogisticRegression
            is expected to provide confidence interval that covers the truth
            in most of the time.
            """
            model = LogisticRegression(
                self.predictors_noisy, self.response, self.x_s2
            ).fit(DataFrameProcessor(self.data))
            beta, se = model.beta, model.beta_standarderror
            ci_lower = beta - 1.96 * se
            ci_upper = beta + 1.96 * se

            self.assertTrue(
                ((ci_lower < self.beta_true) & (self.beta_true < ci_upper)).all()
            )


class TestLogisticRegression(Wrapper.BaseTestLogisticRegression):
    def setUp(self):
        self.data = simulate_test_data()
        self.predictors_clear = ["z1", "z2"]
        self.predictors_noisy = ["x1", "x2"]
        self.response = "y_binary"
        self.x_s2 = [2.0 ** 2, 1.0 ** 2]
        self.beta_true = [1, 2, -0.5]


class TestLogisticRegression2(Wrapper.BaseTestLogisticRegression):
    def setUp(self):
        self.data = simulate_test_data_misspecified_model()
        self.predictors_clear = ["z1", "z2"]
        self.predictors_noisy = ["x1_squared", "x2_squared"]
        self.response = "y_binary"
        self.x_s2 = [1.0 ** 2, 2.0 ** 2]
        self.beta_true = [1, 2, -0.5]


if __name__ == "__main__":
    unittest.main(verbosity=2)
