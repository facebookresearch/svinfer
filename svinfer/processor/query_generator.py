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
import sqlalchemy


class QueryGenerator:
    @classmethod
    def generate_query_for_one_row_data(cls, table_name, database=None):
        my_table = sqlalchemy.Table(table_name, sqlalchemy.MetaData(schema=database))
        query = str(
            sqlalchemy.select("*", from_obj=my_table)
            .limit(1)
            .compile(compile_kwargs={"literal_binds": True})
        )
        logging.info("query is \n%s", query)
        return query

    @classmethod
    def generate_query_for_linear_regression(
        cls, x_columns, y_column, table_name, database=None, filters=None
    ):
        x_cols = [sqlalchemy.Column(k, sqlalchemy.Float) for k in x_columns]
        y_col = sqlalchemy.Column(y_column, sqlalchemy.Float)

        filters_cols = {}
        if filters is not None:
            for k in filters:
                filters_cols[k] = sqlalchemy.Column(k, sqlalchemy.String)

        sqlalchemy.Table(
            table_name,
            sqlalchemy.MetaData(schema=database),
            *x_cols,
            y_col,
            *filters_cols.values(),
        )

        k = len(x_columns)
        item_n = [sqlalchemy.func.count(1)]
        item_yty = [sqlalchemy.func.sum(1.0 * y_col * y_col)]

        item_xty = []
        for i1 in range(k):
            if x_columns[i1] == "_intercept":
                item_xty.append(sqlalchemy.func.sum(1.0 * y_col))
            else:
                item_xty.append(sqlalchemy.func.sum(1.0 * x_cols[i1] * y_col))

        item_xtx = []
        for i1 in range(k):
            for i2 in range(i1, k):
                if x_columns[i1] == "_intercept" and x_columns[i2] == "_intercept":
                    item_xtx.append(sqlalchemy.func.sum(1.0))
                elif x_columns[i1] == "_intercept":
                    item_xtx.append(sqlalchemy.func.sum(1.0 * x_cols[i2]))
                elif x_columns[i2] == "_intercept":
                    item_xtx.append(sqlalchemy.func.sum(1.0 * x_cols[i1]))
                else:
                    item_xtx.append(sqlalchemy.func.sum(1.0 * x_cols[i1] * x_cols[i2]))

        items = item_n + item_yty + item_xty + item_xtx

        where_clauses = (
            [filters_cols[k].in_(filters[k]) for k in filters_cols.keys()]
            if filters is not None
            else []
        )
        query = str(
            sqlalchemy.select(columns=items)
            .where(sqlalchemy.and_(*where_clauses))
            .compile(compile_kwargs={"literal_binds": True})
        )
        logging.info("query is \n%s", query)
        return query

    @classmethod
    def split_query_result_for_linear_regression(cls, x_columns, query_result):
        k = len(x_columns)
        n = query_result[0]
        yty = query_result[1]
        xty = query_result[2 : (2 + k)]
        xtx = np.zeros((k, k))
        xtx[np.triu_indices(k)] = query_result[(2 + k) :]
        i, j = np.triu_indices(k, 1)
        xtx[(j, i)] = xtx[(i, j)]

        logging.info("number n is \n%s", n)
        logging.info("number yty is \n%s", yty)
        logging.info("vector xty is \n%s", xty)
        logging.info("matrix xtx is \n%s", xtx)

        return n, xtx, xty, yty

    @classmethod
    def generate_query_for_summary_statistics(
        cls, columns, table_name, database=None
    ):
        cols = [sqlalchemy.Column(k, sqlalchemy.Float) for k in columns]
        sqlalchemy.Table(
            table_name,
            sqlalchemy.MetaData(schema=database),
            *cols,
        )

        items = []
        for col in cols:
            items.append(col)
            for _ in range(3):
                items.append(items[-1] * col)
        avg_items = [sqlalchemy.func.avg(t) for t in items]

        query = str(
            sqlalchemy.select(columns=avg_items)
            .compile(compile_kwargs={"literal_binds": True})
        )
        logging.info("query is \n%s", query)
        return query

    @classmethod
    def split_query_result_for_summary_statistics(cls, columns, query_result):
        k = len(columns)
        moments = []
        for i in range(k):
            moments.append(query_result[4 * i : 4 * (i + 1)])

        logging.info("moments for each columns are \n%s", moments)
        return moments
