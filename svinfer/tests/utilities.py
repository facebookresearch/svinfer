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

import numpy as np
import pandas as pd


def simulate_test_data(n = 100000, seed=0):
    """
    A simulation setting from the following paper https://gking.harvard.edu/dpd
    The z1, z2 are features,
    while x1, x2 are features with designed noise;
    The response y is the generated based on z1, z2.
    """
    np.random.seed(seed)
    z1 = np.random.poisson(lam=7, size=n)
    z2 = np.random.poisson(lam=9, size=n) + (2 * z1)
    return pd.DataFrame(
        {
            "y": 10
            + (12 * z1)
            - (3 * z2)
            + (2 * np.random.standard_normal(size=n)),
            "z1": z1,
            "z2": z2,
            "x1": z1 + np.random.standard_normal(size=n) * 2.0,
            "x2": z2 + np.random.standard_normal(size=n) * 1.0,
        }
    )

def check_if_almost_equal(x1, x2, absolute_tolerance=1e-12, relative_tolerance=1e-12):
    # logic: absolute(x1 - x2) <= (absolute_tolerance + relative_tolerance * absolute(x2))
    return np.isclose(x1, x2, rtol=relative_tolerance, atol=absolute_tolerance).all()
