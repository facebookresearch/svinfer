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
from scipy import linalg, special
import pandas as pd
from ..processor.abstract_processor import AbstractProcessor


class SummaryStatisticsForOneColumn:
    def __init__(self, x_moments, s2):
        self.x_moments = x_moments
        self.s2 = s2

    def estimate_moments(self, x_moments):
        """
        Estimate the 1st, 2nd, 3rd, 4th (raw) moments for underlying the data,
        given the moments of the corresponding noisy data.
        :param moments: a list of numeric values
        :return: a list of numeric values
        """
        # the 0th, 1st, 2nd, 3rd, 4th moments of a normal distribution
        # with variance self.s2
        noise_moments = [1, 0, self.s2, 0, 3 * self.s2 ** 2]
        a = np.zeros([4, 4])
        for i in range(4):
            for j in range(i + 1):
                a[i, j] = special.comb(i + 1, j + 1) * noise_moments[i - j]
        b = np.array(x_moments) - np.array(noise_moments[1:])
        z_moments = linalg.solve(a, b)
        return z_moments

    def estimate_summary_statistics(self):
        """
        Compute standard mean, sample standard deviation, sample skewness and sample kortosis
        given the sample moments of the noisy data and the sample size
        """
        mu1, mu2, mu3, mu4 = self.estimate_moments(self.x_moments)
        # get sample average
        average = mu1
        # compute central moments given raw moments
        # formula: https://en.wikipedia.org/wiki/Central_moment
        central_mu2 = mu2 - mu1 ** 2
        central_mu3 = mu3 - 3 * mu1 * mu2 + 2 * mu1 ** 3
        central_mu4 = mu4 - 4 * mu1 * mu3 + 6 * mu1 ** 2 * mu2 - 3 * mu1 ** 4
        # get sample standard deviation
        # warning: this estimation approach cannot guarantee that the sample std is positive.
        if central_mu2 < 0:
            logging.warning("the estimated central moment is negative!")
        standard_deviation = np.sqrt(central_mu2)
        # compute skewness via method of moments
        skewness = central_mu3 / standard_deviation ** 3
        # compute kortosis via method of moments
        kortosis = central_mu4 / standard_deviation ** 4
        return average, standard_deviation, skewness, kortosis


class SummaryStatistics:
    def __init__(self, columns, x_s2):
        self.columns = columns
        self.x_s2 = x_s2

    def _preprocess_data(self, data: AbstractProcessor):
        logging.info("data is an instance of %s", type(data))
        return data.prepare_for_summary_statistics(self.columns)

    def estimate_summary_statistics(self, data):
        x_moments = self._preprocess_data(data)
        tmp = []
        for i in range(len(self.columns)):
            summary_for_one_col = SummaryStatisticsForOneColumn(
                x_moments[i], self.x_s2[i]
            ).estimate_summary_statistics()
            tmp.append(summary_for_one_col)
        return pd.DataFrame(
            data=tmp,
            columns=["average", "standard_deviation", "skewness", "kortosis"],
            index=self.columns)
