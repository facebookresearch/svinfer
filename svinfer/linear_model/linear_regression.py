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

from ..processor.commons import AbstractProcessor
from ..processor.matrix import get_result


class LinearRegressionCoefficients:
    """
    Internal class to be used by LinearRegression
    You should not call this class directly.
    """

    def __init__(self, n, xtx, xty, yty, x_s2, df_corrected=True):
        if xtx.shape[0] != xtx.shape[1]:
            raise ValueError("xtx should be square matrix!")
        if xtx.shape[0] != xty.shape[0]:
            raise ValueError("xtx and xty have unmatched dimension")
        if any(s2 < 0.0 for s2 in x_s2):
            raise ValueError("x_s2 should not contain negative elements")
        self.n = n
        self.xtx = xtx
        self.xty = xty
        self.yty = yty
        self.x_s2 = x_s2
        self.df_corrected = df_corrected
        self.k = self.xtx.shape[0]
        self.beta = None
        self.sigma_sq = None
        self.omega = None

    def estimate_beta(self):
        self.omega = self.xtx.copy()
        self.omega[np.diag_indices(self.k)] = np.diag(self.omega) - self.x_s2

        # check whether omega is positive-definite or not
        # reference on threshold https://fburl.com/7y4wyaev
        eigenvalues = np.linalg.eigvals(self.omega)
        if eigenvalues.max() < 0 or (
            eigenvalues.min() < eigenvalues.max() * self.k * np.finfo(np.float_).eps
        ):
            logging.warning("omega is not positive definite")
            self.beta = np.linalg.lstsq(self.omega, self.xty, rcond=None)[0]
        else:
            self.beta = np.linalg.solve(self.omega, self.xty)

    def estimate_residual_var(self):
        if self.yty is None:
            self.sigma_sq = None
            return
        df = self.n
        if self.df_corrected:
            # I think it should be df -= self.k
            # but use df -= self.k - 1 for now to be aligned with the R package
            df -= self.k - 1
        term1 = (
            (
                self.yty
                - 2 * self.beta.T.dot(self.xty)
                + self.beta.T.dot(self.xtx).dot(self.beta)
            )
            * self.n
            / df
        )
        term2 = (self.beta**2 * self.x_s2).sum()
        sigma_sq = term1 - term2
        # check whether variance is positive or not
        if sigma_sq < 0.0:
            logging.warning("negative residual variance!")
        self.sigma_sq = sigma_sq

    def estimate_all(self):
        self.estimate_beta()
        self.estimate_residual_var()
        return self.beta, self.sigma_sq, self.omega


class LinearRegressionVariance:
    """
    Internal class to be used by LinearRegressionFit
    You should not call this class directly.
    """

    def __init__(
        self,
        n,
        xtx,
        xty,
        yty,
        sigma_sq,
        omega,
        x_s2,
        n_replications=500,
        random_state=None,
    ):
        self.n = n
        self.xtx = xtx
        self.xty = xty
        self.yty = yty
        self.sigma_sq = sigma_sq
        self.omega = omega
        self.x_s2 = x_s2
        self.n_replications = n_replications
        self.random_state = random_state
        self.k = self.xtx.shape[0]

    def estimate_vcov_xx_xx(self, k, l, j, m):
        return (
            self.omega[k, l] * self.x_s2[j] * (j == m)
            + self.omega[k, m] * self.x_s2[j] * (j == l)
            + self.omega[j, l] * self.x_s2[k] * (k == m)
            + self.omega[j, m] * self.x_s2[k] * (k == l)
            + self.x_s2[k] * (k == l) * self.x_s2[j] * (j == m)
            + self.x_s2[k] * (k == m) * self.x_s2[j] * (j == l)
        ) / self.n

    def estimate_vcov_xy_xy(self, k, j):
        return (
            self.sigma_sq * self.omega[k, j] + self.x_s2[k] * (k == j) * self.yty
        ) / self.n

    def estimate_vcov_xy_xx(self, k, j, m):
        return (
            self.x_s2[k] * (k == m) * self.xty[j]
            + self.x_s2[k] * (k == j) * self.xty[m]
        ) / self.n

    def simulate_distribution(self):
        index = np.triu_indices(self.k)
        t = np.concatenate((self.xtx[index], self.xty), axis=0)

        d1, d2 = self.k * (self.k + 1) // 2, self.k
        v_t = np.zeros((d1 + d2, d1 + d2))
        for i1 in range(d1 + d2):
            for i2 in range(i1 + 1):
                if i1 < d1 and i2 < d1:
                    k, j = index[0][i1], index[1][i1]
                    l, m = index[0][i2], index[1][i2]
                    v_t[i1, i2] = self.estimate_vcov_xx_xx(k, l, j, m)
                elif i2 < d1 <= i1:
                    k = i1 - d1
                    j, m = index[0][i2], index[1][i2]
                    v_t[i1, i2] = self.estimate_vcov_xy_xx(k, j, m)
                else:
                    k = i1 - d1
                    j = i2 - d1
                    v_t[i1, i2] = self.estimate_vcov_xy_xy(k, j)
                v_t[i2, i1] = v_t[i1, i2]

        return t, v_t

    def transform_vector_to_matrix(self, sample):
        k = self.k
        d = k * (k + 1) // 2
        xty = sample[d:]
        xtx = np.zeros((k, k))
        xtx[np.triu_indices(k)] = sample[:d]
        i, j = np.triu_indices(k, 1)
        xtx[(j, i)] = xtx[(i, j)]
        return xtx, xty

    def simulate_beta_vcov(self):
        t, v_t = self.simulate_distribution()
        np.random.seed(self.random_state)
        t_samples = np.random.multivariate_normal(
            t, v_t, self.n_replications, check_valid="ignore"
        )

        simu_beta_list = []
        for i in range(self.n_replications):
            simu_xtx, simu_xty = self.transform_vector_to_matrix(t_samples[i, :])
            simu_beta, _, _ = LinearRegressionCoefficients(
                self.n, simu_xtx, simu_xty, None, self.x_s2
            ).estimate_all()
            simu_beta_list.append(simu_beta)

        beta_vcov = np.cov(np.array(simu_beta_list), rowvar=False)
        return beta_vcov


class LinearRegression:
    def __init__(
        self,
        x_columns,
        y_column,
        x_s2,
        fit_intercept=True,
        df_corrected=True,
        n_replications=500,
        random_state=None,
    ):
        """
        LinearRegression fits a linear regression where some variables
        in the training data are subject to naturally occurring measurement
        error, and implements a computationally efficient algorithm for
        statistical inference in large datasets.

        :param x_columns: list of strings.
        Specify the column names for the features in the data.
        :param y_column: string
        Specify the column name for the response in the data.
        :param x_s2: list of doubles
        Specify the variance of noise that was added to each of the features.
        It’s length should be equal to that of x_columns. If no noise is
        added to a feature, use 0.0 in the corresponding place.
        :param fit_intercept: bool, optional, default to True
        Whether to include the intercept into the model. If set to False,
        the design matrix will only consist of the features specified.
        If set to True, a column of 1 will be added to the design matrix
        besides features specified.
        :param df_corrected: bool, optional, default to True
        Whether to adjust the degree of freedom when estimating the error
        variance. If set to False, the degree of freedom is n, where n is
        the sample size. If set to True, the degree of freedom is n - p,
        where p is the number of columns in the design matrix.
        :param n_replications: int, optional, default to 500
        The number of simulation replicates. It should be a positive integer.
        :param random_state: int or None, optional, default to None
        Determines random number generation for dataset creation. Pass an
        int for reproducible output across multiple function calls.
        """
        self.x_columns = x_columns
        self.y_column = y_column
        self.x_s2 = ([0.0] if fit_intercept else []) + x_s2
        self.fit_intercept = fit_intercept
        self.df_corrected = df_corrected
        self.n_replications = n_replications
        self.random_state = random_state

        self.beta = None
        self.sigma_sq = None
        self.beta_vcov = None
        self.beta_standarderror = None

    def _preprocess_data(self, data: AbstractProcessor):
        logging.info("data is an instance of %s", type(data))
        x, y = data.prepare_xy(self.x_columns, self.y_column, self.fit_intercept)
        z = get_result(
            {
                "xtx": x.cross(x),
                "xty": x.cross(y),
                "yty": y.cross(y),
            },
            data.run_query,
        )
        n = z["sample_size"]
        xtx = z["xtx"]
        xty = z["xty"][:, 0]
        yty = z["yty"][0, 0]
        return n, xtx, xty, yty

    def fit(self, data):
        """
        fit model
        :return: a tuple; the 1st element is the estimated beta,
        the 2nd element is the estimated sigma squared, the 3rd element is
        the estimated variance-covariance matrix of the estimated beta.
        """
        n, xtx, xty, yty = self._preprocess_data(data)
        self.beta, self.sigma_sq, omega = LinearRegressionCoefficients(
            n, xtx, xty, yty, self.x_s2, df_corrected=self.df_corrected
        ).estimate_all()

        self.beta_vcov = LinearRegressionVariance(
            n,
            xtx,
            xty,
            yty,
            self.sigma_sq,
            omega,
            self.x_s2,
            n_replications=self.n_replications,
            random_state=self.random_state,
        ).simulate_beta_vcov()

        self.beta_standarderror = np.sqrt(np.diag(self.beta_vcov))
        return self
