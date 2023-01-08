# Header files

## Using header files

A header file in Pyccel is a file with a name ending with `.pyh`, which contains function/variable declarations, macro definitions, templates and metavariable declarations.\
Header files serve two purposes:
-   Link external libraries in the targeted languages by providing their function declarations;
-   Accelerate the parsing process of an imported Python module by parsing only its header file (automatically generated) instead of the full module.

### Examples
#### Link with OpenMP
We create the file `header.pyh` that contains an OpenMP function definition:

```python
#$ header metavar module_name = 'omp_lib'
#$ header metavar import_all  = True

#$ header function omp_get_num_threads() results(int)
```
We then create `example.py` file:

```python
from header import omp_get_num_threads
print('number of threads is :', omp_get_num_threads())
```
Pyccel can compile the Python file with the following command: `pyccel example.py --openmp`
, It will then create the executable file `example`
#### Link with a static library
We have the following Fortran Module that we put in the file `funcs.f90`  

```fortran
module funcs

use ISO_C_BINDING

implicit none

contains

!........................................
recursive function fib(n) result(result)

implicit none

integer(C_LONG_LONG) :: result
integer(C_LONG_LONG), value :: n

if (n < 2_C_LONG_LONG) then
  result = n
  return
end if
result = fib(n - 1_C_LONG_LONG) + fib(n - 2_C_LONG_LONG)
return

end function fib
!........................................

end module funcs
```

We then create a static library using these commands:
-   `gfortran -c funcs.f90`
-   `ar rcs libfuncs.a funcs.o`

In order to use this library the user needs to create a header file, we call it  `funcs_headers.pyh`
```python
#$ header metavar module_name      = "funcs"
#$ header metavar ignore_at_import = True

#$ header function fib(int) results(int)
```
After that we can create a Python file `test_funcs.py`,where we can import the Fortran functions and use them

```python
from pyccel.decorators import types

@types('int')
def print_fib(x):
    from  funcs_headers import fib
    print(fib(x))
```
To compile this file we execute the following command `pyccel test_funcs.py --libs=funcs --libdir=$PWD`, this will create the shared library `test_funcs.so`

## Pickling header files
Parsing a large Pyccel header file with hundreds of function declarations may require a significant amount of time, therefore it is important that this process is only done once when pyccelizing multiple Python source files in a large project.

To this end, Pyccel uses the [pickle](https://docs.python.org/3/library/pickle.html) Python module to store the result of the parser to a `.pyccel` binary file, which is created in the same directory as the header file.
Afterwards Pyccel will load the precompiled parser from the `.pyccel` file, instead of parsing the header file again.
This results in a performance gain.

Pyccel will generate a new `.pyccel` binary if the corresponding header file was modified, or if the installed version of Pyccel does not match the one used to parse the header.
