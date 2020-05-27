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
import pandas as pd
from .abstract_processor import AbstractProcessor


class DataFrameProcessor(AbstractProcessor):
    def __init__(self, data: pd.DataFrame):
        # pyre-fixme[25]: Assertion will always fail.
        if not isinstance(data, pd.DataFrame):
            logging.info("data is an instance of %s", type(data))
        self.data = data

    def check_input(self, x_columns, y_column):
        result = np.isin(x_columns, self.data.columns)
        if not result.all():
            raise ValueError(
                "cannot identify {} in data".format(
                    np.array(x_columns)[~result])
            )
        if y_column not in self.data.columns:
            raise ValueError("cannot identify {} in data".format(y_column))

    def prepare_for_linear_regression(self, x_columns, y_column, fit_intercept):
        x = self.data[x_columns].values
        if fit_intercept:
            x = np.insert(x, 0, 1.0, axis=1)
        y = self.data[y_column].values

        n = len(y)
        xtx = x.T.dot(x)
        xty = x.T.dot(y)
        yty = y.T.dot(y)

        return n, xtx, xty, yty
