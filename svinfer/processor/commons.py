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

import abc
import logging
import numpy as np
import pandas as pd
import sqlalchemy

from .matrix import NumpyMatrix, SqlMatrix

class AbstractProcessor(abc.ABC):
    @abc.abstractmethod
    def prepare_xy(self, x_columns, y_column, fit_intercept=True):
        """
        :param x_columns: list of strings
        Specify the column names for the features in the data.
        :param y_column: string or None
        Specify the column name for the response in the data.
        :param fit_intercept: bool
        Whether to include the intercept into the model.
        :return: a tuple with length 2 of x and y as AbstractMatrix.
        """
        pass

    @abc.abstractmethod
    def run_query(self, query):
        pass


class DataFrameProcessor(AbstractProcessor):
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            logging.info("data is an instance of %s", type(data))
        self.data = data

    def prepare_xy(self, x_columns, y_column, fit_intercept=True):
        x = self.data[x_columns].values
        if fit_intercept:
            x = np.insert(x, 0, 1.0, axis=1)
        y = self.data[y_column].values
        return NumpyMatrix(x), NumpyMatrix(y)

    def run_query(self, query):
        pass


class DatabaseProcessor(AbstractProcessor):
    def __init__(self, connection, table_name):
        self.connection = connection
        self.table_name = table_name

    def run_query(self, query):
        df = pd.read_sql(query, self.connection)
        return df.values[0, :]

    def prepare_xy(self, x_columns, y_column, fit_intercept=True):
        x_columns = [sqlalchemy.Column(j, sqlalchemy.FLOAT) for j in x_columns]
        y_column = sqlalchemy.Column(y_column, sqlalchemy.FLOAT)
        sqlalchemy.Table(
            self.table_name, sqlalchemy.MetaData(), y_column, *x_columns
        )
        need = [y_column.label("y")]
        if fit_intercept:
            need.append(sqlalchemy.literal(1.0).label("x0"))
        for i in range(len(x_columns)):
            need.append(x_columns[i].label("x" + str(i + 1)))
        work = sqlalchemy.select(need)
        y_part = list(work.columns)[:1]
        x_part = list(work.columns)[1:]
        return SqlMatrix(x_part), SqlMatrix(y_part)
