# Pyccel : write Python code,  get Fortran speed

 [![Linux unit tests](https://github.com/pyccel/pyccel/actions/workflows/linux.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/linux.yml) [![MacOSX unit tests](https://github.com/pyccel/pyccel/actions/workflows/macosx.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/macosx.yml) [![Windows unit tests](https://github.com/pyccel/pyccel/actions/workflows/windows.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/windows.yml) [![Anaconda-Linux](https://github.com/pyccel/pyccel/actions/workflows/anaconda_linux.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/anaconda_linux.yml) [![Anaconda-Windows](https://github.com/pyccel/pyccel/actions/workflows/anaconda_windows.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/anaconda_windows.yml) [![Intel unit tests](https://github.com/pyccel/pyccel/actions/workflows/intel.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/intel.yml)

[![Documentation](https://github.com/pyccel/pyccel/actions/workflows/documentation-deploy.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/documentation-deploy.yml) [![codacy](https://app.codacy.com/project/badge/Grade/9723f47b95db491886a0e78339bd4698)](https://www.codacy.com/gh/pyccel/pyccel?utm_source=github.com&utm_medium=referral&utm_content=pyccel/pyccel&utm_campaign=Badge_Grade) [![DOI](https://joss.theoj.org/papers/10.21105/joss.04991/status.svg)](https://doi.org/10.21105/joss.04991)

**Pyccel** stands for Python extension language using accelerators.

The aim of **Pyccel** is to provide a simple way to generate automatically, parallel low level code. The main uses would be:

1.  Convert a _Python_ code (or project) into a Fortran or C code.
2.  Accelerate _Python_ functions by converting them to _Fortran_ or _C_ functions.

**Pyccel** can be viewed as:

-   _Python-to-Fortran/C_ converter
-   a compiler for a _Domain Specific Language_ with _Python_ syntax

Pyccel comes with a selection of **extensions** allowing you to convert calls to some specific Python packages to Fortran/C. The following packages will be (partially) covered:

-   `numpy`
-   `scipy`

Pyccel's acceleration capabilities lead to much faster code. Comparisons of Python vs Pyccel or other tools can be found in the [benchmarks](https://github.com/pyccel/pyccel-benchmarks) repository.
The results for the `devel` branch currently show the following performance on Python 3.10:
![Pyccel execution times for devel branch](https://raw.githubusercontent.com/pyccel/pyccel-benchmarks/main/version_specific_results/devel_performance_310_execution.svg)

If you are eager to try Pyccel out, we recommend reading our [quick-start guide](https://pyccel.github.io/pyccel/docs/quickstart.html).

## Citing Pyccel

If Pyccel has been significant in your research, and you would like to acknowledge the project in your academic publication, we would ask that you cite the following paper:

Bourne, Güçlü, Hadjout and Ratnani (2023). Pyccel: a Python-to-X transpiler for scientific high-performance computing. Journal of Open Source Software, 8(83), 4991, <https://doi.org/10.21105/joss.04991>

The associated bibtex can be found [here](https://github.com/pyccel/pyccel/blob/devel/pyccel.bib).

## Installation

Pyccel has a few system requirements to ensure that the system where it is installed is capable of compiling Fortran code.
These requirements are detailed in the [documentation](https://pyccel.github.io/pyccel/docs/installation.html).
Once all requirements are satisfied, we recommend installing Pyccel into a Python virtual environment, which can be created with [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).
Once the Python virtual environment is ready and activated, Pyccel can be easily installed using [pip](https://github.com/pypa/pip?tab=readme-ov-file#pip---the-python-package-installer), the Python package installer.
The simple command

```sh
pip install pyccel
```

will download the latest release of Pyccel from [PyPI](https://pypi.org/project/pyccel/), the Python package index.
Alternative installation methods such as installing from source, or installing with a docker, are described in the [documentation](https://pyccel.github.io/pyccel/docs/installation.html).

## Testing

It is good practice to test that Pyccel works as intended on the machine where it is installed.
To this end Pyccel provides an extended test suite which can be downloaded from the official repository.
Assuming the Python virtual environment is in the directory `<ENV-PATH>`, we activate it with

```sh
source <ENV-PATH>/bin/activate
```

and install the `test` component of the Pyccel package:

```sh
pip install "pyccel[test]"
```

This installs a few additional Python packages which are necessary for running the unit tests and getting a coverage report.

The recommended way of running the unit tests is simply using the command line tool `pyccel test` which is installed with Pyccel.
This runs all unit tests using Pytest under the hood.

Alternatively, if more fine-grained control over which tests are run is desired, e.g. for debugging local modifications to Pyccel, Pytest can be called directly using the commands provided in the [documentation](https://pyccel.github.io/pyccel/docs/testing.html).

## Contributing

We welcome any and all contributions.

There are many ways to help with the Pyccel project which are more or less involved.
A summary can be found in the [documentation](https://pyccel.github.io/pyccel/docs/CONTRIBUTING.html).

We can also be contacted via the [Pyccel Discord Server](https://discord.gg/2Q6hwjfFVb).
