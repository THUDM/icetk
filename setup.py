# Copyright (c) Wendi Zheng and Ming Ding, et al. in KEG, Tsinghua University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

from setuptools import find_packages, setup

def _requirements():
    return Path("requirements.txt").read_text()

setup(
    name="icetk",
    version='0.0.6',
    description="A unified tokenization tool for Images, Chinese and English.",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    install_requires=_requirements(),
    entry_points={},
    packages=find_packages(),
    url="https://github.com/THUDM/icetk",
    author="Wendi Zheng and Ming Ding",
    scripts={},
    include_package_data=True,
    python_requires=">=3.5",
    license="MIT license"
)
