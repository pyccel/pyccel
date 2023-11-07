# Pyccel : write Python code,  get Fortran speed

 [![Linux unit tests](https://github.com/pyccel/pyccel/actions/workflows/linux.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/linux.yml) [![MacOSX unit tests](https://github.com/pyccel/pyccel/actions/workflows/macosx.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/macosx.yml) [![Windows unit tests](https://github.com/pyccel/pyccel/actions/workflows/windows.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/windows.yml) [![Anaconda-Linux](https://github.com/pyccel/pyccel/actions/workflows/anaconda_linux.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/anaconda_linux.yml) [![Anaconda-Windows](https://github.com/pyccel/pyccel/actions/workflows/anaconda_windows.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/anaconda_windows.yml) [![Intel unit tests](https://github.com/pyccel/pyccel/actions/workflows/intel.yml/badge.svg?branch=devel&event=push)](https://github.com/pyccel/pyccel/actions/workflows/intel.yml) [![codacy](https://app.codacy.com/project/badge/Grade/9723f47b95db491886a0e78339bd4698)](https://www.codacy.com/gh/pyccel/pyccel?utm_source=github.com&utm_medium=referral&utm_content=pyccel/pyccel&utm_campaign=Badge_Grade) [![DOI](https://joss.theoj.org/papers/10.21105/joss.04991/status.svg)](https://doi.org/10.21105/joss.04991)

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

If you are eager to try Pyccel out, we recommend reading our [quick-start guide](https://github.com/pyccel/pyccel/blob/devel/docs/quickstart.md).

## Citing Pyccel

If Pyccel has been significant in your research, and you would like to acknowledge the project in your academic publication, we would ask that you cite the following paper:

Bourne, Güçlü, Hadjout and Ratnani (2023). Pyccel: a Python-to-X transpiler for scientific high-performance computing. Journal of Open Source Software, 8(83), 4991, https://doi.org/10.21105/joss.04991

The associated bibtex can be found [here](https://github.com/pyccel/pyccel/blob/devel/pyccel.bib).

## Installation

Pyccel has a few system requirements to ensure that the system where it is installed is capable of compiling Fortran code.
These requirements are detailed in the [documentation](https://github.com/pyccel/pyccel/blob/devel/docs/installation.md).
Once all requirements are satisfied, the simplest way to install Pyccel is using PyPI.
Simply run:

```sh
python3 -m pip install --user pyccel
```

Alternative installation methods such as installing from source, or installing with a docker are described in the [documentation](https://github.com/pyccel/pyccel/blob/devel/docs/installation.md).

## Contributing

We welcome any and all contributions.

There are many ways to help with the Pyccel project which are more or less involved.
A summary can be found in the [documentation](https://github.com/pyccel/pyccel/blob/devel/docs/CONTRIBUTING.md).

We can also be contacted via the [Pyccel Discord Server](https://discord.gg/2Q6hwjfFVb).

## User Documentation

-   [Quick-start Guide](https://github.com/pyccel/pyccel/blob/devel/docs/quickstart.md)

-   [Installation](https://github.com/pyccel/pyccel/blob/devel/docs/installation.md)

-   [Contributing](https://github.com/pyccel/pyccel/blob/devel/docs/CONTRIBUTING.md)

-   [C/Fortran Compilers](https://github.com/pyccel/pyccel/blob/devel/docs/compiler.md)

-   [Decorators](https://github.com/pyccel/pyccel/blob/devel/docs/decorators.md)

-   [Header Files](https://github.com/pyccel/pyccel/blob/devel/docs/header-files.md)

-   [Templates](https://github.com/pyccel/pyccel/blob/devel/docs/templates.md)

-   [N-dimensional Arrays](https://github.com/pyccel/pyccel/blob/devel/docs/ndarrays.md)

-   [Function-pointers as arguments](https://github.com/pyccel/pyccel/blob/devel/docs/function-pointers-as-arguments.md)

-   [Const keyword](https://github.com/pyccel/pyccel/blob/devel/docs/const_keyword.md)

-   Supported libraries/APIs
    -   [OpenMP](https://github.com/pyccel/pyccel/blob/devel/docs/openmp.md)
    -   [NumPy](https://github.com/pyccel/pyccel/blob/devel/docs/numpy-functions.md)

## Developer Documentation

-   [Overview](https://github.com/pyccel/pyccel/blob/devel/developer_docs/overview.md)
-   [How to solve an issue](https://github.com/pyccel/pyccel/blob/devel/developer_docs/how_to_solve_an_issue.md)
-   [Review Process](https://github.com/pyccel/pyccel/blob/devel/developer_docs/review_process.md)
-   [Development Conventions](https://github.com/pyccel/pyccel/blob/devel/developer_docs/development_conventions.md)
-   [Tips and Tricks](https://github.com/pyccel/pyccel/blob/devel/developer_docs/tips_and_tricks.md)
-   [Scope](https://github.com/pyccel/pyccel/blob/devel/developer_docs/scope.md)
-   [Type Inference](https://github.com/pyccel/pyccel/blob/devel/developer_docs/type_inference.md)
-   [Array Ordering](https://github.com/pyccel/pyccel/blob/devel/developer_docs/order_docs.md)
