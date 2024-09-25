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

import numpy as np
import sqlalchemy


class AbstractMatrix(abc.ABC):
    def __init__(self):
        self.value = None
        self.ncol = None
        self.dim = None

    @abc.abstractmethod
    def __pos__(self):
        return NotImplemented

    @abc.abstractmethod
    def __neg__(self):
        return NotImplemented

    @abc.abstractmethod
    def __add__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __radd__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __sub__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __rsub__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __mul__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __rmul__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __truediv__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __rtruediv__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def dot(self, b):
        """
        self b,
        where self is n by k, b is a numpy array with shape (k,),
        and the result is n by 1.
        """
        return NotImplemented

    @abc.abstractmethod
    def outer(self, b):
        """
        self b^T,
        where self is n by 1, b is a numpy array with shape (k,),
        and the result is n by k.
        """
        return NotImplemented

    @abc.abstractmethod
    def cross(self, other):
        """
        self^T other,
        where self is n by p, other is n by q,
        and the result is p by q.
        """
        return NotImplemented

    @abc.abstractmethod
    def exp(self):
        return NotImplemented

    @abc.abstractmethod
    def log(self):
        return NotImplemented


class NumpyMatrix(AbstractMatrix):
    def __init__(self, x, dim=None):
        assert isinstance(x, np.ndarray)
        super().__init__()
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        self.value = x
        self.ncol = x.shape[1]
        if dim is None:
            dim = (self.ncol,)
        else:
            assert np.prod(dim) == self.ncol
        self.dim = dim

    def __pos__(self):
        return NumpyMatrix(self.value)

    def __neg__(self):
        return NumpyMatrix(-self.value)

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return NumpyMatrix(self.value + other)
        if isinstance(other, NumpyMatrix):
            return NumpyMatrix(self.value + other.value)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return NumpyMatrix(self.value * other)
        if isinstance(other, NumpyMatrix):
            return NumpyMatrix(self.value * other.value)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return NumpyMatrix(self.value / other)
        if isinstance(other, NumpyMatrix):
            return NumpyMatrix(self.value / other.value)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return NumpyMatrix(other / self.value)
        if isinstance(other, NumpyMatrix):
            return NumpyMatrix(other.value / self.value)
        return NotImplemented

    def dot(self, b):
        assert isinstance(b, np.ndarray)
        assert len(b.shape) == 1
        assert self.ncol == b.size
        return NumpyMatrix(np.dot(self.value, b))

    def outer(self, b):
        assert isinstance(b, np.ndarray)
        assert len(b.shape) == 1
        assert self.ncol == 1
        return NumpyMatrix(np.outer(self.value, b))

    def cross(self, other):
        assert isinstance(other, NumpyMatrix)
        result = []
        for j in range(other.ncol):
            result.append(self.value * other.value[:, j : (j + 1)])
        result = np.concatenate(result, axis=1)
        return NumpyMatrix(result, dim=(self.ncol, other.ncol))

    def exp(self):
        return NumpyMatrix(np.exp(self.value))

    def log(self):
        return NumpyMatrix(np.log(self.value))


class SqlMatrix(AbstractMatrix):
    def __init__(self, x, dim=None):
        assert isinstance(x, list)
        super().__init__()
        self.value = x.copy()
        self.ncol = len(x)
        if dim is None:
            dim = (self.ncol,)
        else:
            assert np.prod(dim) == self.ncol
        self.dim = dim

    def __pos__(self):
        return SqlMatrix(self.value)

    def __neg__(self):
        return SqlMatrix([-self.value[j] for j in range(self.ncol)])

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return SqlMatrix([self.value[j] + other for j in range(self.ncol)])
        if isinstance(other, SqlMatrix):
            assert other.ncol == self.ncol or other.ncol == 1
            return SqlMatrix(
                [self.value[j] + other.value[j % other.ncol] for j in range(self.ncol)]
            )
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return SqlMatrix([self.value[j] * float(other) for j in range(self.ncol)])
        if isinstance(other, SqlMatrix):
            assert other.ncol == self.ncol or other.ncol == 1
            return SqlMatrix(
                [self.value[j] * other.value[j % other.ncol] for j in range(self.ncol)]
            )
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return SqlMatrix([self.value[j] / float(other) for j in range(self.ncol)])
        if isinstance(other, SqlMatrix):
            assert other.ncol == self.ncol or other.ncol == 1
            return SqlMatrix(
                [self.value[j] / other.value[j % other.ncol] for j in range(self.ncol)]
            )
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return SqlMatrix([float(other) / self.value[j] for j in range(self.ncol)])
        if isinstance(other, SqlMatrix):
            assert other.ncol == self.ncol or other.ncol == 1
            return SqlMatrix(
                [other.value[j % other.ncol] / self.value[j] for j in range(self.ncol)]
            )
        return NotImplemented

    def dot(self, b):
        assert isinstance(b, np.ndarray)
        assert len(b.shape) == 1
        assert self.ncol == b.size
        result = sqlalchemy.literal(0.0)
        for j in range(self.ncol):
            result += self.value[j] * b[j]
        return SqlMatrix([result])

    def outer(self, b):
        assert isinstance(b, np.ndarray)
        assert len(b.shape) == 1
        assert self.ncol == 1
        result = []
        for j in range(b.size):
            result.append(self.value[0] * b[j])
        return SqlMatrix(result)

    def cross(self, other):
        assert isinstance(other, SqlMatrix)
        result = []
        for j in range(other.ncol):
            for i in range(self.ncol):
                result.append(self.value[i] * other.value[j])
        return SqlMatrix(result, dim=(self.ncol, other.ncol))

    def exp(self):
        result = []
        for j in range(self.ncol):
            result.append(sqlalchemy.func.exp(self.value[j]))
        return SqlMatrix(result)

    def log(self):
        result = []
        for j in range(self.ncol):
            result.append(sqlalchemy.func.log(self.value[j]))
        return SqlMatrix(result)


def get_result(tags, query_runner=None):
    result = {}
    sample_size = None
    sql_columns = [sqlalchemy.func.count(1)]
    sql_parts = []
    for k, v in tags.items():
        if k == "sample_size":
            raise ValueError("'sample_size' is reserved")
        if isinstance(v, SqlMatrix):
            start = len(sql_columns)
            for j in range(v.ncol):
                sql_columns.append(sqlalchemy.func.avg(v.value[j]))
            stop = len(sql_columns)
            sql_parts.append({"key": k, "start": start, "stop": stop, "dim": v.dim})
        elif isinstance(v, NumpyMatrix):
            if sample_size is None:
                sample_size = v.value.shape[0]
            else:
                assert sample_size == v.value.shape[0]
            result[k] = np.array(v.value.mean(axis=0)).reshape(v.dim)
        else:
            raise TypeError("Unknown value type with key = " + k)
    if len(sql_columns) > 1:
        if query_runner is None:
            raise ValueError("'query_runner' is required")
        query_text = str(
            sqlalchemy.select(sql_columns).compile(
                compile_kwargs={"literal_binds": True}
            )
        )
        raw = query_runner(query_text)
        if sample_size is None:
            sample_size = raw[0]
        else:
            assert sample_size == raw[0]
        for part in sql_parts:
            result[part["key"]] = np.array(raw[part["start"] : part["stop"]]).reshape(
                part["dim"]
            )
    result["sample_size"] = sample_size
    return result
