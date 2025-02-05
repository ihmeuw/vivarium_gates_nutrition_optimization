#!/usr/bin/env python
import json
import os
import sys

from packaging.version import parse
from setuptools import find_packages, setup

with open("python_versions.json", "r") as f:
    supported_python_versions = json.load(f)

python_versions = [parse(v) for v in supported_python_versions]
min_version = min(python_versions)
max_version = max(python_versions)
if not (
    min_version <= parse(".".join([str(v) for v in sys.version_info[:2]])) <= max_version
):
    py_version = ".".join([str(v) for v in sys.version_info[:3]])
    # NOTE: Python 3.5 does not support f-strings
    error = (
        "\n--------------------------------------------\n"
        "Error: Vivarium Gates Nutrition Optimization runs under python {min_version}-{max_version}.\n"
        "You are running python {py_version}.\n".format(
            min_version=min_version.base_version,
            max_version=max_version.base_version,
            py_version=py_version,
        )
        + "--------------------------------------------\n"
    )
    print(error, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(
        os.path.join(src_dir, "vivarium_gates_nutrition_optimization", "__about__.py")
    ) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        "gbd_mapping>=3.1.1",
        "vivarium>=3.0.0, <4.0.0",
        "vivarium_public_health>=3.0.0, <4.0.0",
        "click",
        "jinja2",
        "loguru",
        "numpy",
        "pandas",
        "pyyaml",
        "scipy",
        "tables",
    ]

    # use "pip install -e .[dev]" to install required components + extra components
    data_requirements = ["vivarium_inputs[data]>=4.1.1"]
    cluster_requirements = ["vivarium_cluster_tools>=2.0.0, <3.0.0"]
    test_requirements = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
    ]
    lint_requirements = [
        "black==22.3.0",
        "isort==5.13.2",
    ]

    setup(
        name=about["__title__"],
        version=about["__version__"],
        description=about["__summary__"],
        long_description=long_description,
        license=about["__license__"],
        url=about["__uri__"],
        author=about["__author__"],
        author_email=about["__email__"],
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,
        install_requires=install_requirements,
        extras_require={
            "test": test_requirements,
            "cluster": cluster_requirements,
            "data": data_requirements + cluster_requirements,
            "dev": test_requirements + cluster_requirements + lint_requirements,
        },
        zip_safe=False,
        entry_points="""
            [console_scripts]
            make_artifacts=vivarium_gates_nutrition_optimization.tools.cli:make_artifacts
        """,
    )
