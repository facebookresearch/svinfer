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

import logging
import numpy as np
from scipy import optimize
from sklearn.linear_model import LogisticRegression as NaiveLogisticRegression


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
        self.x_s2 = ([0.0] if fit_intercept else []) + x_s2
        self.fit_intercept = fit_intercept

        self.success = None
        self.beta = None
        self.beta_vcov = None
        self.beta_standarderror = None

    @staticmethod
    def _score_all(beta, y, x, x_s2):
        c = x + np.outer(y - 0.5, x_s2 * beta)
        return c.T * (y - 1.0 / (1 + np.exp(-c.dot(beta))))

    @staticmethod
    def _score(beta, y, x, x_s2):
        return np.mean(LogisticRegression._score_all(beta, y, x, x_s2), axis=1)

    @staticmethod
    def _jacobian(beta, y, x, x_s2):
        n, p = x.shape
        c = x + np.outer(y - 0.5, x_s2 * beta)
        p = 1.0 / (1 + np.exp(-c.dot(beta)))
        term1 = np.mean((y - p) * (y - 0.5)) * np.diag(x_s2)
        term2_stag = np.mean(c.T * p * (1 - p) * (y - 0.5), axis=1)
        term2 = np.outer(term2_stag, beta) * x_s2
        term3 = (c.T * p * (1 - p)).dot(c) / n
        return term1 - term2 - term3

    @staticmethod
    def _get_coefficients(x, y, x_s2):
        naive_model = NaiveLogisticRegression(
            fit_intercept=False, penalty='none', random_state=0).fit(x, y)
        initial_value = naive_model.coef_
        model = optimize.root(
            LogisticRegression._score,
            initial_value,
            args=(y, x, x_s2),
            method='lm',
            jac=LogisticRegression._jacobian)
        return model.x, model.success

    @staticmethod
    def _get_covariance(x, y, x_s2, beta):
        n = x.shape[0]
        score_est = LogisticRegression._score_all(beta, y, x, x_s2)
        var_est = score_est.dot(score_est.T) / n
        j_est = LogisticRegression._jacobian(beta, y, x, x_s2)
        j_est_inv = np.linalg.inv(j_est)
        return j_est_inv.T.dot(var_est).dot(j_est_inv) / n

    def fit(self, data):
        x = data[self.x_columns].values
        if self.fit_intercept:
            x = np.insert(x, 0, 1.0, axis=1)
        y = data[self.y_column].values
        beta_est, success = LogisticRegression._get_coefficients(x, y, self.x_s2)
        if not success:
            logging.warning("optimization does not converge!")
        var_est = LogisticRegression._get_covariance(x, y, self.x_s2, beta_est)
        self.success = success
        self.beta = beta_est
        self.beta_vcov = var_est
        self.beta_standarderror = np.sqrt(np.diag(var_est))
        return self
