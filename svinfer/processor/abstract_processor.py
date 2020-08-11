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

from abc import ABC, abstractmethod


class AbstractProcessor(ABC):
    @abstractmethod
    def check_input(self, x_columns, y_column):
        """
        :param x_columns: list of strings
        Specify the column names for the features in the data.
        :param y_column: string
        Specify the column name for the response in the data.
        """
        pass

    @abstractmethod
    def prepare_for_linear_regression(self, x_columns, y_columns, fit_interept):
        """
        :param x_columns: list of strings
        Specify the column names for the features in the data.
        :param y_column: string
        Specify the column name for the response in the data.
        :param fit_intercept: bool
        Whether to include the intercept into the model.
        :return: a tuple with length 4.
        Its elements includes a scalar, an array with shape (k, k),
        an array with shape (k, ), and a scalar.
        """
        pass

    @abstractmethod
    def prepare_for_summary_statistics(self, columns):
        """
        :param columns: list of strings
        Specify the column name of the features in the data
        :return: an array with shape (k, 4)
        Its elements are the 1st, 2nd, 3rd and 4th moments of the features specified in columns.
        """
        pass
