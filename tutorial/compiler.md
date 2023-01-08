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

-   **`exec`** : The name of the executable
-   **`mpi\_exec`** : The name of the MPI executable
-   **`language`** : The language handled by this compiler
-   **`module\_output\_flag`** : This flag is only required when the language is fortran. It specifies the flag which indicates where .mod files should be saved (e.g. '-J' for gfortran)
-   **`debug\_flags`** : A list of flags used when compiling in debug mode \[optional\]
-   **`release\_flags`** : A list of flags used when compiling in release mode \[optional\]
-   **`general\_flags`** : A list of flags used when compiling in any mode \[optional\]
-   **`standard\_flags`** : A list of flags used to impose the expected language standard \[optional\]
-   **`libs`** : A list of libraries necessary for compiling \[optional\]
-   **libdirs** : A list of library directories necessary for compiling \[optional\]
-   **includes** : A list of include directories necessary for compiling \[optional\]
  
In addition, for each accelerator (`mpi`/`openmp`/`openacc`/`python`) that you will use the JSON file must define the following:
  
-   **flags** : A list of flags used to impose the expected language standard \[optional\]
-   **libs** : A list of libraries necessary for compiling \[optional\]
-   **libdirs** : A list of library directories necessary for compiling \[optional\]
-   **includes** : A list of include directories necessary for compiling \[optional\]

Python is considered to be an accelerator and must additionally specify shared\_suffix.

The default compilers can provide examples compatible with your system once pyccel has been executed at least. To export the JSON file describing your setup, use the `--export-compile-info` flag and provide a target file name.
E.g.
```shell
pyccel --compiler=PGI --language=c --export-compile-info=icc.json
```
