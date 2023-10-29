"""Setup script for package."""
# pylint: disable=consider-using-with, no-self-use
import os
import re
import sys
from pathlib import Path

from setuptools import Command, find_packages, setup

match = re.search(
    r'^VERSION\s*=\s*"(.*)"',
    Path("offerexpert/version.py").read_text(encoding="UTF-8"),
    re.M,
)
VERSION = match.group(1) if match else "???"
LONG_DESCRIPTION = Path("README.md").read_text(encoding="UTF-8")


class VerifyVersion(Command):
    """Command for verifying that git tag matches package version."""

    description = "verify that the git tag matches package version"
    user_options = []

    def initialize_options(self):
        """Implement required method for Command."""

    def finalize_options(self):
        """Implement required method for Command."""

    def run(self):
        """
        Check that the git tag matches the package version.

        If it doesn't match, exit.
        """
        tag = os.getenv("CIRCLE_TAG")
        if not _validate_version(tag, VERSION):
            info = f"Git tag: '{tag}' does not match package version: {VERSION}"  # noqa: E501
            sys.exit(info)


def _validate_version(tag: str | None, version: str) -> bool:
    if not tag:
        return version == "0.0.0"

    if tag[0] != "v":
        return False
    return tag[1:] == version


setup(
    name="OfferXpert",
    version=VERSION,
    description="",
    long_description=LONG_DESCRIPTION,
    author="Apdullah Yayik",
    author_email="apdullahyayik@gmail.com",
    url="",
    keywords="",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "pandas",
        "sentence-transformers",
        "faiss-cpu",
        "gensim",
        "matplotlib",
        "nltk",
        "regex",
        "seaborn",
        "scikit-learn",
        "torch",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "bandit",
            "black",
            "coverage",
            "flake8-annotations",
            "flake8-plus",
            "flake8-pytest-style",
            "flake8",
            "jupyter",
            "lxml-stubs",
            "pycodestyle",
            "pydocstyle",
            "pyenchant",
            "pylint",
            "pytest-cov",
            "pytest-mock",
            "pytest",
            "rope",
            "tox",
        ],
        "test": [
            "coverage",
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "tox",
        ],
    },
    classifiers=["Programming Language :: Python :: 3.10"],
    entry_points={
        "console_scripts": [
            "offerexpert = offerexpert.__main__:main",
        ]
    },
    cmdclass={
        "verify": VerifyVersion,
    },
)
