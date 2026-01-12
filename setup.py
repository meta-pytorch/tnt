# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
from datetime import date
from typing import List

from setuptools import find_packages, setup
from torchtnt import __version__


def current_path(file_name: str) -> str:
    return os.path.abspath(os.path.join(__file__, os.path.pardir, file_name))


def read_requirements(file_name: str) -> List[str]:
    with open(current_path(file_name), encoding="utf8") as f:
        return [r for r in f.read().strip().split() if not r.startswith("-")]


def get_nightly_version() -> str:
    return date.today().strftime("%Y.%m.%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchtnt setup")
    parser.add_argument(
        "--nightly",
        dest="nightly",
        action="store_true",
        help="enable settings for nightly package build",
        default=False,
    )
    parser.add_argument(
        "--append-to-version",
        dest="append_version",
        help="append string to end of version number (e.g. a1)",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    with open(current_path("README.md"), encoding="utf8") as f:
        readme: str = f.read()

    custom_args, setup_args = parse_args()
    package_name = "torchtnt" if not custom_args.nightly else "torchtnt-nightly"
    version = __version__ if not custom_args.nightly else get_nightly_version()
    if custom_args.append_version:
        version = f"{version}{custom_args.append_version}"

    print(f"using package_name={package_name}, version={version}")

    sys.argv = [sys.argv[0]] + setup_args

    setup(
        name=package_name,
        version=version,
        author="PyTorch",
        author_email="daniellepintz@fb.com",
        description="A lightweight library for PyTorch training tools and utilities",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/pytorch/tnt",
        license="BSD-3",
        keywords=["pytorch", "torch", "training", "tools", "utilities"],
        python_requires=">=3.7",
        install_requires=read_requirements("requirements.txt"),
        packages=find_packages(),
        package_data={"torchtnt": ["py.typed"]},
        zip_safe=True,
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        extras_require={"dev": read_requirements("dev-requirements.txt")},
    )
