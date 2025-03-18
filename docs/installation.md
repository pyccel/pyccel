# Installing Pyccel

## Pyccel Installation Methods

Pyccel can be installed on virtually any machine that provides Python 3, the pip package manager, a C/Fortran compiler, and an Internet connection.
Some advanced features of Pyccel require additional non-Python libraries to be installed, for which we provide detailed instructions below.

Alternatively, Pyccel can be deployed through a **Linux Docker image** that contains all dependencies, and which can be setup with any version of Pyccel.
For more information, please read the section on [Pyccel container images](#Pyccel-Container-Images).

It is possible to use Pyccel with anaconda but this is generally not advised as anaconda modifies paths used for finding executables, shared libraries and other objects.
Support is provided for anaconda on linux/macOS.

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

In order to install Pyccel from source, CMake (>=3.13) is additionally required to build the gFTL dependence.

### Linux Debian-Ubuntu-Mint

To install all requirements on a Linux Ubuntu machine, just use APT, the Advanced Package Tool:

```sh
sudo apt update
sudo apt install gcc
sudo apt install gfortran
sudo apt install libblas-dev liblapack-dev
sudo apt install libopenmpi-dev openmpi-bin
sudo apt install libomp-dev libomp5
sudo apt install cmake
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
dnf install cmake
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
brew install cmake
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
choco install cmake
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

In Administrator git-bash, generate `mpi.mod` for GFortran according to <https://abhilashreddy.com/writing/3/mpi_instructions.html>:

```sh
cd "$MSMPI_INC"
sed -i 's/mpifptr.h/x64\/mpifptr.h/g' mpi.f90
sed -i 's/mpifptr.h/x64\/mpifptr.h/g' mpif.h
gfortran -c -D_WIN64 -D INT_PTR_KIND\(\)=8 -fno-range-check mpi.f90
cd -
```

Generate static `libmsmpi.a` from `msmpi.dll`:

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

If you use Pyccel often and aren't scared of debugging any potential DLL confusion from other libraries. You can use a `.pth` file to run the necessary commands automatically. The location where the `.pth` file should be installed is described in the [Python docs](https://docs.python.org/3/library/site.html). Once the site is located you can run:
```sh
echo "import os; os.add_dll_directory('C://ProgramData/chocolatey/lib/mingw/tools/install/mingw64/lib'); os.add_dll_directory('C://ProgramData/chocolatey/lib/mingw/tools/install/mingw64/bin')" > $SITE_PATH/dll_path.pth
```
(The command may need adapting for your installation locations)

## Installation

We recommend creating a clean Python virtual environment using [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment):
```sh
python3 -m venv <ENV-PATH>
```
where `<ENV-PATH>` is the location to create the virtual environment.
(A new directory will be created at the required location.)

In order to activate the environment from a new terminal session just run the command
```sh
source <ENV-PATH>/bin/activate
```

At this point Pyccel may be installed in **standard mode**, which copies the relevant files to the correct locations of the virtual environment, or in **editable mode**, which only installs symbolic links to the Pyccel directory.
The latter mode allows one to affect the behaviour of Pyccel by modifying the source files.

In both cases we use **`pip`** to install a _Python_ library **`pyccel`** and three _binary files_ called **`pyccel`**, **`pyccel-clean`**, and **`pyccel-init`**.
The **`pyccel`** command translates the given Python file to a Fortran or C file, and then compiles the generated code to a Python C extension module or a simple executable.
The **`pyccel-clean`** command is a user helper tool which cleans up the environment of the temporary files generated by Pyccel.
The **`pyccel-init`** command is necessary in order to [pickle header files](./header-files.md#Pickling-header-files) on read-only systems as explained below.

### Standard install from PyPI

In order to install the latest release of Pyccel on PyPI, the Python package index, just run
```sh
source <ENV-PATH>/bin/activate
pip install pyccel
```
Pip automatically downloads any required Python packages from PyPI and installs them.
The flags `--upgrade` and `--force-reinstall` may be needed in order to overwrite a pre-existing installation of Pyccel.
It is also possible to install a specific release of Pyccel, for example `pip install pyccel==1.11.2`.

### Editable install from sources

For those who want to have the most recent version ("trunk") of Pyccel, and possibly modify its source code, the following commands clone our Git repository from GitHub, checkout the `devel` branch, and install symbolic links to the `pyccel` directory:
```sh
source <ENV-PATH>/bin/activate
git clone --recurse-submodules https://github.com/pyccel/pyccel.git
cd pyccel
pip install --editable ".[test]"
```
This installs a few additional Python packages which are necessary for running the unit tests and getting a coverage report.

### On a read-only system

If the folder where Pyccel is saved is read-only, it may be necessary to run an additional command after installation or updating:
```sh
sudo pyccel-init
```

This step is necessary in order to [pickle header files](./header-files.md#Pickling-header-files).
If this command is not run then Pyccel will still run correctly but may be slower when using [OpenMP](./openmp.md) or other supported external packages.
A warning, reminding the user to execute this command, will be printed to the screen when pyccelising files which rely on these packages if the pickling step has not been executed.

## Additional packages

In order to run the unit tests and to get a coverage report, a few additional Python packages should be installed:

```sh
pip install --user -e ".[test]"
```

Most of the unit tests can also be run in parallel.

## Testing

To test your Pyccel installation please run the script `tests/run\_tests\_py3.sh` (Unix), or `tests/run\_tests.bat` (Windows).

Continuous testing runs on GitHub actions: <https://github.com/pyccel/pyccel/actions?query=branch%3Adevel>

Most of the unit tests can also be run in parallel.

## Pyccel Container Images

Pyccel container images are available through both Docker Hub (<docker.io>) and the GitHub Container Registry (<ghcr.io>).

The images:

-   are based on `ubuntu:latest`
-   use distro packaged Python3, GCC, GFortran, BLAS and OpenMPI
-   support all Pyccel releases except the legacy "0.1"

Image tags match Pyccel releases.

In order to implement your Pyccel-accelerated code, you can use a host based volume during the Pyccel container creation.

For example:

```sh
docker pull pyccel/pyccel:v1.0.0
docker run -it -v $PWD:/data:rw  pyccel/pyccel:v1.0.0 bash
```

If you are using SELinux, you will need to set the right context for your host based volume.
Alternatively you may have docker or podman set the context using `-v $PWD:/data:rwz` instead of `-v $PWD:/data:rw` .
