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

from .abstract_processor import AbstractProcessor
from .query_generator import QueryGenerator
from .query_executor import QueryExecutor


class DatabaseProcessor(AbstractProcessor):
    def __init__(self, connection, table_name):
        self.connection = connection
        self.table_name = table_name

    def _execute_query(self, query):
        """
        Execute a SQL query. Fetch query results if any.
        :param query: string, the SQL query
        :return: a DataFrame
        """
        logging.debug("running query")
        logging.debug(query)
        cursor = self.connection.cursor()
        data, colnames = QueryExecutor(cursor).execute_query(query)
        if len(data) == 0:
            raise RuntimeError("query returns no data!")
        return data, colnames

    def check_input(self, x_columns, y_column):
        x_columns = [col.lower() for col in x_columns]
        y_column = y_column.lower()
        _, colnames = self._execute_query(
            QueryGenerator.generate_query_for_one_row_data(self.table_name)
        )
        table_columns = [x.lower() for x in colnames]
        index = np.isin(x_columns, table_columns)
        if not index.all():
            raise ValueError(
                "cannot identify {} in data".format(np.array(x_columns)[~index])
            )
        if y_column not in table_columns:
            raise ValueError("cannot identify {} in data".format(y_column))

    def prepare_for_linear_regression(self, x_columns, y_column, fit_intercept):
        if fit_intercept:
            x_columns = ["_intercept"] + x_columns

        query = QueryGenerator.generate_query_for_linear_regression(
            x_columns, y_column, self.table_name,
        )
        data, _ = self._execute_query(query)
        query_result = np.array(data)[0]

        n, xtx, xty, yty = QueryGenerator.split_query_result(x_columns, query_result)
        return n, xtx, xty, yty
