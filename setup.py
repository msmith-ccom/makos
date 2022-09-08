import codecs
import os
import re

from setuptools import setup, find_packages

# ------------------------------------------------------------------
#                         HELPER FUNCTIONS

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M, )
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")


# ------------------------------------------------------------------
#                          POPULATE SETUP

setup(
    name="makos",
    version=find_version("makos", "__init__.py"),
    license="None",

    namespace_packages=[
    ],
    packages=find_packages(exclude=[
        "*.tests", "*.tests.*", "tests.*", "tests", "*.test*",
    ]),
    package_data={
        "": [
        ],
    },
    zip_safe=False,
    ext_modules=None,
    setup_requires=[
        "setuptools",
        "wheel"
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib-base"
    ],
    python_requires='>=3.8',
    entry_points={
        "gui_scripts": [

        ],
        "console_scripts": [

        ],
    },
    test_suite="tests",

    description="Library for modeling transmit radiation patterns and acoustic "
                "modeling.",
    long_description=read(os.path.join(here, "README.rst")),
    url="None",
    classifiers=[
        "Development Status :: 0 - Beta",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: None",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    keywords="ocean mapping acoustic sonar arrays radiation pattern",
    author="Michael Smith,",
    author_email="msmith@ccom.unh.edu, ",
)