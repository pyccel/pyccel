# Different compilers in Pyccel
## Compilers supported by Pyccel

Pyccel provides default compiler settings for 4 different compiler families:
-   **GNU** : `gcc` / `gfortran`
-   **intel** : `icc` / `ifort`
-   **PGI** : `pgcc` / `pgfortran`
-   **nvidia** : `nvc` / `nvfort`

**Warning** : The **GNU** compiler is currently the only compiler which is tested regularly

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

## User-defined compiler

The user can also define their own compiler in a JSON file. To use this definition, the location of the JSON file must be passed to the _compiler_ argument. The JSON file must define the following:

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

Python is considered to be an accelerator and must additionally specify shared\_suffix.

The default compilers can provide examples compatible with your system once Pyccel has been executed at least. To export the JSON file describing your setup, use the `--export-compile-info` flag and provide a target file name.
E.g.
```shell
pyccel --compiler=PGI --language=c --export-compile-info=icc.json
```
## Utilising Pyccel within a Conda Environment
While Conda simplifies Python package management, it can sometimes introduce challenges when working with Pyccel. By default, Pyccel ignores Conda paths when searching for compilers in the system's PATH. Pyccel offers flags to manage Conda-related warnings for a smoother experience: `--disable-conda-warnings` hides warnings, and `--detailed-conda-warnings` lists the ignored Conda paths.

### There are a few cons you should be aware of when working with Conda

-   Dependency conflicts: Conda manages package dependencies, but sometimes conflicts can occur between different packages or versions. This can result in unexpected behaviour or an inability to install certain packages.
-   Environment activation: Conda allows you to create and manage multiple environments. However, it's essential to activate the desired environment before running your code or executing commands. Failing to activate the correct environment can lead to using the wrong package versions or encountering compatibility issues.
-   Activation scripts: Conda modifies your shell's environment variables when you activate an environment. If you have custom scripts or configurations that depend on specific environment variables, they may not work correctly when switching between Conda environments.
-   Mixing Conda and pip: Conda is primarily focused on managing packages within its own ecosystem, while pip is the standard package manager for Python. Mixing Conda and pip installations within the same environment can lead to conflicts or unexpected behaviour.
-   Limited package availability: Although Conda provides a wide range of packages, there may be cases where certain packages or specific versions are not available through Conda channels.
-   Conda environment size: Conda environments can occupy a significant amount of disk space due to the duplication of certain dependencies across different environments.
-   Conda update and package conflicts: When updating Conda or packages within an environment, conflicts may arise between package versions. These conflicts can lead to broken environments or inconsistent behaviour.