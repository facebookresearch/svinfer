#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
from scipy import optimize

from ..processor.commons import AbstractProcessor

from ..processor.matrix import get_result


class LogisticRegression:
    """
    Use the conditional score estimator proposed in theorem 4 in
    Stefanski, L. A., & Carroll, R. J. (1985). Covariate measurement error in
    logistic regression. The Annals of Statistics, 1335-1351.
    https://www.jstor.org/stable/2241358
    """

    def __init__(
        self,
        x_columns,
        y_column,
        x_s2,
        fit_intercept=True,
    ):
        self.x_columns = x_columns
        self.y_column = y_column
        self.x_s2 = np.array([0.0] + x_s2 if fit_intercept else x_s2)
        self.fit_intercept = fit_intercept

        self.success = None
        self.beta = None
        self.beta_vcov = None
        self.beta_standarderror = None

    @staticmethod
    def _score(beta, x, y, x_s2, query_runner):
        """
        score(beta) = avg_{i = 1}^{n} (y_i - p_i(beta)) c_i(beta)
        where p_i(beta) = (1 + exp(-c_i(beta)^T beta))^{-1},
        and c_i(beta) = x_i + (y_i - 0.5) diag(x_s2) beta.

        For Jacobian, the element at the i-th row and the j-th column is
        the partial derivative of the i-th component in score(beta)
        with respect to the j-th component in beta.

        jacobian(beta) = avg_{i = 1}^{n} (
            (y_i - p_i(beta)) (y_i - 0.5) diag(x_s2)
            - p_i(beta) (1 - pi(beta)) (y_i - 0.5) c_i(beta) beta^T diag(x_s2)
            - p_i(beta) (1 - pi(beta)) c_i(beta) c_i(beta)^T
        )
        """
        c = x + (y - 0.5).outer(x_s2 * beta)
        score = c * (y - 1.0 / (1.0 + (-c.dot(beta)).exp()))
        p = 1.0 / (1 + (-c.dot(beta)).exp())
        term1_part = (y - p) * (y - 0.5)
        term2_part = c * (p * (1 - p) * (y - 0.5))
        term3_part = (c * (p * (1 - p))).cross(c)
        z = get_result(
            {
                "score": score,
                "term1_part": term1_part,
                "term2_part": term2_part,
                "term3_part": term3_part,
            },
            query_runner,
        )
        score = z["score"]
        term1 = z["term1_part"] * np.diag(x_s2)
        term2 = np.outer(z["term2_part"], x_s2 * beta)
        term3 = z["term3_part"]
        jacobian = term1 - term2 - term3
        return score, jacobian

    @staticmethod
    def _meat(beta, x, y, x_s2, query_runner):
        c = x + (y - 0.5).outer(x_s2 * beta)
        score = c * (y - 1.0 / (1.0 + (-c.dot(beta)).exp()))
        meat = score.cross(score)
        z = get_result(
            {
                "meat": meat,
            },
            query_runner,
        )
        meat = z["meat"]
        sample_size = z["sample_size"]
        return meat, sample_size

    @staticmethod
    def _get_coefficients(x, y, x_s2, query_runner):
        naive = optimize.root(
            LogisticRegression._score,
            np.zeros(x_s2.shape),
            args=(x, y, np.zeros(x_s2.shape), query_runner),
            method="lm",
            jac=True,
        )
        if naive.success:
            initial = naive.x
        else:
            initial = np.zeros(x_s2.shape)
        final = optimize.root(
            LogisticRegression._score,
            initial,
            args=(x, y, x_s2, query_runner),
            method="lm",
            jac=True,
        )
        beta_est = final.x
        success = final.success
        return beta_est, success

    @staticmethod
    def _get_covariance(beta, x, y, x_s2, query_runner):
        meat, n = LogisticRegression._meat(beta, x, y, x_s2, query_runner)
        jacobian = LogisticRegression._score(beta, x, y, x_s2, query_runner)[1]
        bread = np.linalg.inv(jacobian)
        return bread.dot(meat).dot(bread.T) / n

    def fit(self, data):
        assert isinstance(data, AbstractProcessor)
        x, y = data.prepare_xy(self.x_columns, self.y_column, self.fit_intercept)
        beta_est, success = LogisticRegression._get_coefficients(
            x, y, self.x_s2, data.run_query
        )
        if not success:
            logging.warning("optimization does not converge!")
        var_est = LogisticRegression._get_covariance(
            beta_est, x, y, self.x_s2, data.run_query
        )
        self.success = success
        self.beta = beta_est
        self.beta_vcov = var_est
        self.beta_standarderror = np.sqrt(np.diag(var_est))
        return self
