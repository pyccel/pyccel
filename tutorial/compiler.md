# Different compilers in Pyccel
## Compilers supported by Pyccel

Pyccel provides default compiler settings for 4 different compiler families:
-   **GNU** : gcc / gfortran
-   **intel** : icc / ifort
-   **PGI** : pgcc / pgfortran
-   **nvidia** : nvc / nvfort

**Warning** : The **GNU** compiler is currently the only compiler which is tested regularly

## Specifying a compiler

The default compiler family is **GNU**. To use a different compiler, the compiler family should be passed to either pyccel or epyccel.
E.g.
```shell
pyccel example.py --compiler=intel
```
or
```python
epyccel(my_func, compiler='intel')
```

## User-defined compiler

The user can also define their own compiler in a json file. To use this definition, the location of the json file must be passed to the _compiler_ argument. The json file must define the following:

-   **exec** : The name of the executable
-   **mpi\_exec** : The name of the mpi executable
-   **language** : The language handled by this compiler
-   **module\_output\_flag** : This flag is only required when the language is fortran. It specifies the flag which indicates where .mod files should be saved (e.g. '-J' for gfortran)
-   **debug\_flags** : A list of flags used when compiling in debug mode \[optional\]
-   **release\_flags** : A list of flags used when compiling in release mode \[optional\]
-   **general\_flags** : A list of flags used when compiling in any mode \[optional\]
-   **standard\_flags** : A list of flags used to impose the expected language standard \[optional\]
-   **libs** : A list of libraries necessary for compiling \[optional\]
-   **libdirs** : A list of library directories necessary for compiling \[optional\]
-   **includes** : A list of include directories necessary for compiling \[optional\]
  
In addition, for each accelerator (mpi/openmp/openacc/python) that you will use the json file must define the following:
  
-   **flags** : A list of flags used to impose the expected language standard \[optional\]
-   **libs** : A list of libraries necessary for compiling \[optional\]
-   **libdirs** : A list of library directories necessary for compiling \[optional\]
-   **includes** : A list of include directories necessary for compiling \[optional\]

Python is considered to be an accelerator and must additionally specify shared\_suffix.

The default compilers can provide examples compatible with your system once pyccel has been executed at least. These json files can be found in the folder pyccel/compilers/
