# Different compilers in Pyccel
## Compilers supported by Pyccel

Pyccel provides default compiler settings for 4 different compiler families:
-   **GNU** : `gcc` / `gfortran`
-   **Intel** : `icx` / `ifx`
-   **NVIDIA** : `nvc` / `nvfort`

**Warning** : The **NVIDIA** compiler is not currently tested regularly

## Specifying a compiler

The default compiler family is **GNU**. To use a different compiler, the compiler family should be passed to either `pyccel` or `epyccel`.
E.g.
```shell
pyccel example.py --compiler=intel
```
or
```python
epyccel(my_func, compiler='intel')
```

It is also possible to change the default compiler family by setting the environment variable `PYCCEL_DEFAULT_COMPILER`.

## Using an unsupported compiler or custom compiler flags

Pyccel's compiler settings are described internally by a dictionary. This makes it easy for a user to define their own compiler settings. This is done via a JSON file. The location of the JSON file is passed to the _compiler_ argument of the Pyccel command line tool or `epyccel`. The JSON file must define the following:

-   `exec` : The name of the executable
-   `mpi_exec` : The name of the MPI executable
-   `language` : The language handled by this compiler
-   `module_output_flag` : This flag is only required when the language is Fortran. It specifies the flag which indicates where .mod files should be saved (e.g. '-J' for `gfortran`)
-   `debug_flags` : A list of flags used when compiling in debug mode \[optional\]
-   `release_flags` : A list of flags used when compiling in release mode \[optional\]
-   `general_flags` : A list of flags used when compiling in any mode \[optional\]
-   `standard_flags` : A list of flags used to impose the expected language standard \[optional\]
-   `libs` : A list of libraries necessary for compiling \[optional\]
-   `libdirs` : A list of library directories necessary for compiling \[optional\]
-   `includes` : A list of include directories necessary for compiling \[optional\]
  
In addition, for each accelerator (`mpi`/`openmp`/`openacc`/`python`) that you will use the JSON file must define the following:
  
-   `flags` : A list of flags used to impose the expected language standard \[optional\]
-   `libs` : A list of libraries necessary for compiling \[optional\]
-   `libdirs` : A list of library directories necessary for compiling \[optional\]
-   `includes` : A list of include directories necessary for compiling \[optional\]

Python is considered to be an accelerator and must additionally specify `shared_suffix`.

The default compilers can provide examples compatible with your system once Pyccel has been executed at least. To export the JSON file describing your setup, use the `--export-compile-info` flag and provide a target file name.
E.g.
```shell
pyccel --compiler=intel --language=c --export-compile-info=icc.json
```
once this file has been modified it can then be used with:
```shell
pyccel --compiler=icc.json --language=c <file_to_translate>
```

## Utilising Pyccel within Anaconda Environment
While Anaconda is a popular way to install Python as it simplifies package management, it can introduce challenges when working with compilers.

Upon installation Anaconda modifies your shell's environment variables.

This can lead to packages installed in conda/anaconda usurping the packages manually installed by the user.

This in turn easily leads to nasty surprises (e.g. using libraries compiled for serial execution instead of optimised parallel implementations).

To avoid these problems, Pyccel ignores Conda paths when searching for compilers in the system's PATH.

By using the expected compiler, the expected libraries are correctly linked.

Pyccel offers the `--conda-warnings` flag which takes one of the following options : `off`, `basic`, `verbose`.
This controls the visibility of Conda-related warnings. The default value is `basic` which indicates when folders are ignored.
The `verbose` option additionally outputs a list of all ignored folders. If these folders should not be ignored then this can be avoided by using a user-defined compiler file as described above.
The `off` option suppresses all Conda-related warnings.
