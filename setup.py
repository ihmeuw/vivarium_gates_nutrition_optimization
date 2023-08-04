#!/usr/bin/env python
import os

from setuptools import find_packages, setup

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
        "gbd_mapping==3.1.1",
        "vivarium==1.2.2",
        "vivarium_public_health==1.0.0",
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
    data_requires = [
        "vivarium_cluster_tools==1.3.10",
        "vivarium_inputs[data]==4.1.1",
    ]

    test_requirements = ["pytest"]

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
            "data": data_requires,
            "dev": test_requirements + data_requires,
        },
        zip_safe=False,
        entry_points="""
            [console_scripts]
            make_artifacts=vivarium_gates_nutrition_optimization.tools.cli:make_artifacts
            make_results=vivarium_gates_nutrition_optimization.tools.cli:make_results
        """,
    )
