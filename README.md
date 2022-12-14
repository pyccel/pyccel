# Welcome to Pyccel

 [![master_tests](https://github.com/pyccel/pyccel/actions/workflows/master.yml/badge.svg)](https://github.com/pyccel/pyccel/actions/workflows/master.yml) [![codacy](https://app.codacy.com/project/badge/Grade/9723f47b95db491886a0e78339bd4698)](https://www.codacy.com/gh/pyccel/pyccel?utm_source=github.com&utm_medium=referral&utm_content=pyccel/pyccel&utm_campaign=Badge_Grade)

**Pyccel** stands for Python extension language using accelerators.

The aim of **Pyccel** is to provide a simple way to generate automatically, parallel low level code. The main uses would be:

1.  Convert a _Python_ code (or project) into a Fortran or C code.
2.  Accelerate _Python_ functions by converting them to _Fortran_ or _C_ functions.

**Pyccel** can be viewed as:

-   _Python-to-Fortran/C_ converter
-   a compiler for a _Domain Specific Language_ with _Python_ syntax

Pyccel comes with a selection of **extensions** allowing you to convert calls to some specific Python packages to Fortran/C. The following packages will be covered (partially):

-   numpy
-   scipy
-   mpi4py (not available yet)
-   h5py (not available yet)

Pyccel's acceleration capabilities lead to much faster code. Comparisons of Python vs Pyccel or other tools can be found in the [benchmarks](https://github.com/pyccel/pyccel-benchmarks) repository.
The results for the master branch currently show the following performance on python 3.10:
![Pyccel execution times for master branch](https://github.com/pyccel/pyccel-benchmarks/blob/main/version_specific_results/devel_performance_310_execution.png)

If you are eager to try Pyccel out, we recommend reading our [quick-start guide](./tutorial/quickstart.md)

## Table of contents

-   [User Documentation](#User-documentation)

-   [Developer Documentation](#Developer-documentation)

-   [Pyccel Installation Methods](#Pyccel-Installation-Methods)

-   [Requirements](#Requirements)
    -   [Linux-Debian-Ubuntu-Mint](#Linux-Debian-Ubuntu-Mint)
    -   [Linux Fedora-CentOS-RHEL](#Linux-Fedora-CentOS-RHEL)
    -   [Mac OS X](#Mac-OS-X)
    -   [Windows](#Windows)

-   [Installation](#Installation)
    -   [From PyPi](#From-PyPi)
    -   [From sources](#From-sources)
    -   [On a read-only system](#On-a-read-only-system)

-   [Additional packages](#Additional-packages)

-   [Testing](#Testing)

-   [Pyccel Container Images](#Pyccel-Container-Images)

## User Documentation

-   [Quick-start Guide](./tutorial/quickstart.md)

-   [C/Fortran Compilers](./tutorial/compiler.md)

-   [Decorators](./tutorial/decorators.md)

-   [Header Files](./tutorial/header-files.md)

-   [Templates](./tutorial/templates.md)

-   [N-dimensional Arrays](./tutorial/ndarrays.md)

-   [Function-pointers as arguments](./tutorial/function-pointers-as-arguments.md)

-   [Const keyword](./tutorial/const_keyword.md)

-   Supported libraries/APIs
    -   [OpenMP](./tutorial/openmp.md)
    -   [Numpy](./tutorial/numpy-functions.md)

## Pyccel Installation Methods

Pyccel can be installed on virtually any machine that provides Python 3, the pip package manager, a C/Fortran compiler, and an Internet connection.
Some advanced features of Pyccel require additional non-Python libraries to be installed, for which we provide detailed instructions below.

Alternatively, Pyccel can be deployed through a **Linux Docker image** that contains all dependencies, and which can be setup with any version of Pyccel.
For more information, please read the section on [Pyccel container images](#Pyccel-Container-Images).

It is possible to use pyccel with anaconda but this is generally not advised as anaconda modifies paths used for finding executables, shared libraries and other objects.
Support is provided for anaconda on linux/macos.

On Windows support is limited to examples which do not use external libraries.
This is because we do not know of a way to reliably avoid [DLL hell](https://en.wikipedia.org/wiki/DLL_Hell).
As a result DLLs managed by conda are always loaded before DLLs related to the compiler.

## Requirements

First of all, Pyccel requires a working Fortran/C compiler

For Fortran it supports

-   [GFortran](https://gcc.gnu.org/fortran/)
-   [Intel® Fortran Compiler](https://software.intel.com/en-us/fortran-compilers)
-   [PGI Fortran](https://www.pgroup.com/index.htm)

For C it supports

-   [GCC](https://gcc.gnu.org/)
-   [Intel® Compiler](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/dpc-compiler.html)
-   [PGI](https://www.pgroup.com/index.htm)

In order to perform fast linear algebra calculations, Pyccel uses the following libraries:

-   [BLAS](http://www.netlib.org/blas/)(Basic Linear Algebra Subprograms)
-   [LAPACK](http://www.netlib.org/lapack/)(Linear Algebra PACKage)

Finally, Pyccel supports distributed-memory parallel programming through the Message Passing Interface (MPI) standard; hence it requires an MPI library like

-   [Open-MPI](https://www.open-mpi.org/)
-   [MPICH](https://www.mpich.org/)
-   [Intel® MPI Library](https://software.intel.com/en-us/mpi-library)

We recommend using GFortran/GCC and Open-MPI.

Pyccel also depends on several Python3 packages, which are automatically downloaded by pip, the Python Package Installer, during the installation process. In addition to these, unit tests require additional packages which are installed as optional dependencies with pip, while building the documentation requires [Sphinx](http://www.sphinx-doc.org/).

### Linux Debian-Ubuntu-Mint

To install all requirements on a Linux Ubuntu machine, just use APT, the Advanced Package Tool:

```sh
sudo apt update
sudo apt install gcc
sudo apt install gfortran
sudo apt install libblas-dev liblapack-dev
sudo apt install libopenmpi-dev openmpi-bin
sudo apt install libomp-dev libomp5
```

### Linux Fedora-CentOS-RHEL

Install all requirements using the DNF software package manager:

```sh
su
dnf check-update
dnf install gcc
dnf install gfortran
dnf install blas-devel lapack-devel
dnf install openmpi-devel
dnf install libgomp
exit
```

Similar commands work on Linux openSUSE, just replace `dnf` with `zypper`.

### Mac OS X

On an Apple Macintosh machine we recommend using [Homebrew](https://brew.sh/):

```sh
brew update
brew install gcc
brew install openblas
brew install lapack
brew install open-mpi
brew install libomp
```

This requires that the Command Line Tools (CLT) for Xcode are installed.

### Windows

Support for Windows is still experimental, and the installation of all requirements is more cumbersome.
We recommend using [Chocolatey](https://chocolatey.org/) to speed up the process, and we provide commands that work in a git-bash sh.
In an Administrator prompt install git-bash (if needed), a Python3 distribution, and a GCC compiler:

```sh
choco install git
choco install python3
choco install mingw
```

Download x64 BLAS and LAPACK DLLs from <https://icl.cs.utk.edu/lapack-for-windows/lapack/>:

```sh
WEB_ADDRESS=https://icl.cs.utk.edu/lapack-for-windows/libraries/VisualStudio/3.7.0/Dynamic-MINGW/Win64
LIBRARY_DIR=/c/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/lib
curl $WEB_ADDRESS/libblas.dll -o $LIBRARY_DIR/libblas.dll
curl $WEB_ADDRESS/liblapack.dll -o $LIBRARY_DIR/liblapack.dll
```

Generate static MS C runtime library from corresponding dynamic link library:

```sh
cd "$LIBRARY_DIR"
cp $SYSTEMROOT/SysWOW64/vcruntime140.dll .
gendef vcruntime140.dll
dlltool -d vcruntime140.def -l libmsvcr140.a -D vcruntime140.dll
cd -
```

Download MS MPI runtime and SDK, then install MPI:

```sh
WEB_ADDRESS=https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1
curl -L $WEB_ADDRESS/msmpisetup.exe -o msmpisetup.exe
curl -L $WEB_ADDRESS/msmpisdk.msi -o msmpisdk.msi
./msmpisetup.exe
msiexec //i msmpisdk.msi
```

At this point, close and reopen your terminal to refresh all environment variables!

In Administrator git-bash, generate mpi.mod for gfortran according to <https://abhilashreddy.com/writing/3/mpi_instructions.html>:

```sh
cd "$MSMPI_INC"
sed -i 's/mpifptr.h/x64\/mpifptr.h/g' mpi.f90
sed -i 's/mpifptr.h/x64\/mpifptr.h/g' mpif.h
gfortran -c -D_WIN64 -D INT_PTR_KIND\(\)=8 -fno-range-check mpi.f90
cd -
```

Generate static libmsmpi.a from msmpi.dll:

```sh
cd "$MSMPI_LIB64"
cp $SYSTEMROOT/SysWOW64/msmpi.dll .
gendef msmpi.dll
dlltool -d msmpi.def -l libmsmpi.a -D msmpi.dll
cd -
```

On Windows it is important that all locations containing DLLs are on the PATH. If you have added any variables to locations which are not on the PATH then you need to add them:
```sh
echo $PATH
export PATH=$LIBRARY_DIR;$PATH
```

[As of Python 3.8](https://docs.python.org/3/whatsnew/3.8.html#bpo-36085-whatsnew) it is also important to tell Python which directories contain trusted DLLs. In order to use Pyccel this should include all folders containing DLLs used by your chosen compiler. The function which communicates this to Python is: [`os.add_dll_directory`](https://docs.python.org/3/library/os.html#os.add_dll_directory).
E.g:
```python
import os
os.add_dll_directory(C://ProgramData/chocolatey/lib/mingw/tools/install/mingw64/lib')
os.add_dll_directory('C://ProgramData/chocolatey/lib/mingw/tools/install/mingw64/bin')
```

These commands must be run every time a Python instance is opened which will import a Pyccel-generated library.

If you use Pyccel often and aren't scared of debugging any potential DLL confusion from other libraries. You can use a `.pth` file to run the necessary commands automatically. The location where the `.pth` file should be installed is described in the [python docs](https://docs.python.org/3/library/site.html). Once the site is located you can run:
```sh
echo "import os; os.add_dll_directory('C://ProgramData/chocolatey/lib/mingw/tools/install/mingw64/lib'); os.add_dll_directory('C://ProgramData/chocolatey/lib/mingw/tools/install/mingw64/bin')" > $SITE_PATH/dll_path.pth
```
(The command may need adapting for your installation locations)

## Installation

On Windows and/or Anaconda Python, use `pip` instead of `pip3` for the Installation of Pyccel below.

### From PyPi

Simply run, for a user-specific installation:

```sh
pip3 install --user pyccel
```

or:

```sh
sudo pip3 install pyccel
```

for a system-wide installation.

### From sources

-   **Standard mode**:

    ```sh
    git clone git@github.com:pyccel/pyccel.git
    cd pyccel
    pip3 install --user .
    ```

-   **Development mode**:

    ```sh
    git clone git@github.com:pyccel/pyccel.git
    cd pyccel
    pip3 install --user -e .[test]
    ```

this will install a _python_ library **pyccel** and a _binary_ called **pyccel**.
Any required Python packages will be installed automatically from PyPI.

### On a read-only system

If the folder where Pyccel is saved is read only, it may be necessary to run an additional command after installation or updating:
```sh
sudo pyccel-init
```

This step is necessary in order to [pickle header files](./tutorial/header-files.md#Pickling-header-files).
If this command is not run then Pyccel will still run correctly but may be slower when using [OpenMP](./tutorial/openmp.md) or other supported external packages.
A warning, reminding the user to execute this command, will be printed to the screen when pyccelizing files which rely on these packages if the pickling step has not been executed.

## Additional packages

In order to run the unit tests and to get a coverage report, a few additional Python packages should be installed:

```sh
pip install --user -e .[test]
```

Most of the unit tests can also be run in parallel.

## Testing

To test your Pyccel installation please run the script _tests/run\_tests\_py3.sh_ (Unix), or _tests/run\_tests.bat_ (Windows).

Continuous testing runs on github actions: <https://github.com/pyccel/pyccel/actions?query=branch%3Amaster>

## Pyccel Container Images

Pyccel container images are available through both Docker Hub (docker.io) and the GitHub Container Registry (ghcr.io).

The images:

-   are based on ubuntu:latest
-   use distro packaged python3, gcc, gfortran, blas and openmpi
-   support all pyccel releases except the legacy "0.1"

Image tags match Pyccel releases.

In order to implement your Pyccel-accelerated code, you can use a host based volume during the Pyccel container creation.

For example:

```sh
docker pull pyccel/pyccel:v1.0.0
docker run -it -v $PWD:/data:rw  pyccel/pyccel:v1.0.0 bash
```

If you are using SELinux, you will need to set the right context for your host based volume.
Alternatively you may have docker or podman set the context using -v $PWD:/data:rwz instead of -v $PWD:/data:rw .

## Developer Documentation

-   [Overview](./developer_docs/overview.md)
-   [How to solve an issue](./developer_docs/how_to_solve_an_issue.md)
-   [Review Process](./developer_docs/review_process.md)
-   [Tips and Tricks](./developer_docs/tips_and_tricks.md)
