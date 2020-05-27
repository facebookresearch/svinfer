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

import setuptools


if __name__ == "__main__":
    setuptools.setup(
        name="svinfer",
        version="0.1.3",
        author="Facebook",
        author_email="researchtool-help@fb.com",
        description="Statistical models with valid inference "
                    "for differentially private data",
        packages=[
            "svinfer",
            "svinfer.linear_model",
            "svinfer.processor",
        ],
        python_requires=">=3.6",
        classifiers=["Programming Language :: Python :: 3"],
    )
