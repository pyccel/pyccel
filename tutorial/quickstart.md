# Pyccel's Quickstart Guide

Pyccel is a **static compiler** for Python 3, using Fortran or C as backend language, with a focus on high-performance computing (HPC) applications.

Pyccel's main goal is to accelerate the transition from **prototype** to **production** in scientific computing, where programmers usually develop their prototype code in a user-friendly interactive language like Python, but they later need a statically compiled language like Fortran/C/C++ (as well as SIMD vectorization, parallel multi-threading, MPI parallelization, GPU offloading, etc.) in order to achieve the performance required by their final application.

Pyccel generates very fast Fortran or C code which is **human-readable**, hence the expert programmer can easily profile the code on the target machine and further optimize it.

## Some Useful Background

We recall that Python 3 is an **interpreted** language, **dynamically typed** and **garbage-collected**.
In particular, it is worth clarifying the difference between an object and a variable (or name):

### Python object

-   Is created by the Python interpreter when `object.__new__()` is invoked (e.g. as a result of an expression).
-   Can be either **mutable** or **immutable**, but its type never changes.
-   Resides in memory and has a **reference count**.
-   Is accessed through one or more Python variables.
-   Is destroyed by the garbage collector when its reference count drops to zero.

### Python variable

-   Is a reference to a Python object in memory (which is the variable's **value**).

-   Is created with an assignment operation `x = expr`, where `x` is the new variable and `expr` is any expression:

    -   `x` will reference the **value** of `expr`, therefore increasing the object's reference count
    -   If the variable `x` already existed, the interpreter reduces the reference count of its old value

-   Can be destroyed with the command `del x`, which reduces its value's reference count

-   Since a variable can be reassigned to any object, its type could change at run-time; hence we say that Python is **dynamically typed**.

### Statically Typed Languages
	
A language is statically typed if the type of a variable is known at compile-time instead of run-time.
Common examples of statically typed languages include Java, C, C++, FORTRAN, Pascal and Scala.
These languages provide a set of built-in types, and provide the means for creating additional user-defined types.
It is the programmer's responsibility to explicitly declare the type of every variable used in the code.

### Further Reading

-   https://docs.python.org/3/tutorial/classes.html#a-word-about-names-and-objects
-   https://docs.python.org/3/reference/datamodel.html#objects-values-and-types
-   https://en.wikipedia.org/wiki/Type_system
-   https://android.jlelse.eu/magic-lies-here-statically-typed-vs-dynamically-typed-languages-d151c7f95e2b

## How does Pyccel work?

In order to translate Python 3 code (dynamically typed) to efficient Fortran/C code (statically typed), Pyccel makes a certain number of *assumptions*, needs *additional information* from the user, and imposes a few *restrictions* on the code.
The fundamental rule that guides Pyccel's design is that the Python 3 code and the generated Fortran/C code should behave in the same way.
Up to round-off errors, this means that a function must return the same output value if the input arguments are the same.
Moreover, Pyccel raises some warnings/errors for the conflicts that can occur in some corner cases.

### Type Inference

Pyccel uses type inference to calculate the type of all variables through a static analysis of the Python code.
It understands all Python literals, and computes the type of an expression that contains previously processed variables.
For example:
```python
x = 3        # int
y = x * 0.5  # float
z = y + 1j   # complex
```

### Assumptions and Restrictions

Because the type of a variable must be unique within a given **scope**, Pyccel cannot support some of the flexibility that Python provides.
The following basic restrictions apply:

1.  The type of a variable cannot be changed
2.  The type of a variable cannot depend on an if condition

For example:
```python
if condition:
    y = 1    # int
    z = 1.0  # float
else:
    y = 2    # int     : OK
    z = 3j   # complex : Pyccel raises an error!
```

### Type Annotations

When parsing a function, Pyccel needs to know the type of the input arguments in order to perform type inference, and ultimately compute the type of the output result.
(In the case of recursive functions, the return type should also be declared.)
Pyccel can then perform type inference on other Python code that uses that function, because the type of its result will already be known.

The programmer has various ways to provide the argument type information to Pyccel.
We recommend using Python-style annotations, which have the syntax:
```python
def fun(arg1: 'type1', arg2: 'type2', ..., argN: 'typeN') -> 'return_type':
```
or to declare Numpy arrays
```python
def fun(arg1: 'type1[:]', arg2: 'type2[:,:]', ..., argN: 'typeN[dimensions]') -> 'return_type':
```
The number of dimensions of an array is equal to the number of comma-separated colons in the square brackets.
So `arr[:]` means that the array `arr` has 1 dimension, `arr[:,:]` means that it has 2 dimensions and so on.
In general string type hints must be used to provide Pyccel with information about the number of dimensions.

For scalar variables and arrays Pyccel supports the following data types:

-   built-in datatypes: `bool`, `int`, `float`, `complex`
-   Numpy integer types: `int8`, `int16`, `int32`, `int64`
-   Numpy real types: `float32`, `float64`, `double`
-   Numpy complex types: `complex64`, `complex128`

## How to use Pyccel?

### Installation

Pyccel's official releases can be downloaded from PyPI (the Python Package Index) using `pip`.
To get the latest (trunk) version of Pyccel, one can clone the `git` repository from GitHub and checkout the `master` branch.
Detailed installation instructions are found in the [README](https://github.com/pyccel/pyccel/blob/master/README.rst) file.

### Command Line Usage

After installation, the `pyccel` command will be available on a terminal app (iterm or terminal for MacOs, terminal for Linux).
After typing `pyccel`, the usage should be displayed on the terminal; if this is the case then the installation has succeeded.
In essence the `pyccel` command translates the given Python file to a Fortran or C file, and then compiles the generated code to a Python C extension module or a simple executable.

#### Example 1: "Hello World" program

In this first example we create the simplest Python script `hello.py`, which prints `Hello, world!` followed by an empty line.
We use the `pyccel` command to translate the Python code to a C program in file `hello.c`, which is placed in the new `__pyccel__` directory.
By default Pyccel also compiles the C code into an executable named `hello`, which is placed in the same directory as the original file:
```bash
$ echo 'print("Hello, world!\n")' > hello.py
$ python3 hello.py
Hello, world!

$ pyccel hello.py --language c
$ ./hello
Hello, world!

```
Here are the contents of the current directory:
```bash
$ tree
.
├── hello
├── hello.py
└── __pyccel__
    └── hello.c

1 directory, 4 files
```
And this is the C file generated by Pyccel:
```C
#include <stdlib.h>
#include <stdio.h>

int main()
{
    printf("%s\n", "Hello, world!\n");
    return 0;
}
```
#### Example 2: extension module

If the Python file to be accelerated only contains functions and class definitions, Pyccel will treat it as a Python module instead of just a script.
Accordingly, it will not generate a program, but rather a Python C extension module which can be imported from Python.

For example, we now consider the Python module `mod.py`, which reads
```python
def factorial(n : int) -> int:
    if n <= 0: return 1
    else     : return n * factorial(n - 1)

def binomial_coefficient(n : int, k : int):
    num = factorial(n)
    den = factorial(k) * factorial(n - k)
    return num // den
```
We use the `pyccel` command to translate `mod.py` to the C files `mod.c` and `mod.h`, which are placed in the new `__pyccel__` directory:
```bash
$ pyccel mod.py --language c
```
By default Pyccel also compiles the C code into a Python C extension module named `mod.<TAG>.so`, which is placed in the same directory as `mod.py`.
To achieve this Pyccel generates the additional files `mod_wrapper.c` (which interacts directly with the CPython API) and `setup_mod.py` (which defines the build procedure for the extension module), as well as a `build` directory.

If the command `import mod` is now given to the Python interpreter, this will import the Python C extention module `mod.<TAG>.so` instead of the pure Python module `mod.py`.

These are the contents of the current directory:
```bash
$ tree
.
├── mod.cpython-36m-x86_64-linux-gnu.so
├── mod.py
└── __pyccel__
    ├── build
    │   ├── lib.linux-x86_64-3.6
    │   └── temp.linux-x86_64-3.6
    │       └── mod_wrapper.o
    ├── mod.c
    ├── mod.h
    ├── mod.o
    ├── mod_wrapper.c
    └── setup_mod.py

4 directories, 10 files

```

This is the C file `mod.c` generated by Pyccel:
```c
#include <mod.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

/*........................................*/
int64_t factorial(int64_t n)
{
    int64_t Out_0001;
    if (n <= 0)
    {
        Out_0001 = 1;
        return Out_0001;
    }
    else if (1)
    {
        Out_0001 = n * factorial(n - 1);
        return Out_0001;
    }
}
/*........................................*/

/*........................................*/
int64_t binomial_coefficient(int64_t n, int64_t k)
{
    int64_t num;
    int64_t den;
    int64_t Out_0001;
    num = factorial(n);
    den = factorial(k) * factorial(n - k);
    Out_0001 = floor((double)(num) / (double)(den));
    return Out_0001;
}
/*........................................*/
```
And this is the header file `mod.h`:
```c
#ifndef MOD_H
#define MOD_H

#include <stdlib.h>
#include <stdint.h>

int64_t factorial(int64_t n);

int64_t binomial_coefficient(int64_t n, int64_t k);

#endif // MOD_Hy
```

#### Example 3: Numpy arrays
   
   file_name.py:
   
   ```python
   from numpy import array
   from numpy import empty
   from numpy import ones

   x = array([1, 2, 3])
   y = empty((10, 10))

   a = ones(3)
   b = ones((4,3))
   ```

  file_name.c:
   
  ```c
  #include <stdint.h>
  #include <stdlib.h>
  #include <ndarrays.h>
  int main()
  {
   t_ndarray x;
   t_ndarray y;
   t_ndarray a;
   t_ndarray b;

   x = array_create(1, (int32_t[]){3}, nd_int64);
   int64_t array_dummy_0001[] = {1, 2, 3};
   memcpy(x.nd_int64, array_dummy_0001, x.buffer_size);

   y = array_create(2, (int32_t[]){10, 10}, nd_double);

   a = array_create(1, (int32_t[]){3}, nd_double);
   array_fill((double)1.0, a);

   b = array_create(2, (int32_t[]){4, 3}, nd_double);
   array_fill((double)1.0, b);

   free_array(x);
   free_array(y);
   free_array(a);
   free_array(b);
   return 0;
  }
  ```

  file_name.f90:

  ```Fortran
  program prog_test

  use ISO_C_BINDING

  implicit none

  integer(C_INT64_T), allocatable :: x(:)
  real(C_DOUBLE), allocatable :: y(:,:)
  real(C_DOUBLE), allocatable :: a(:)
  real(C_DOUBLE), allocatable :: b(:,:)

  allocate(x(0:3_C_INT64_T - 1_C_INT64_T))
  x = [1_C_INT64_T, 2_C_INT64_T, 3_C_INT64_T]
  allocate(y(0:10_C_INT64_T - 1_C_INT64_T, 0:10_C_INT64_T - 1_C_INT64_T))
  allocate(a(0:3_C_INT64_T - 1_C_INT64_T))
  a = 1.0_C_DOUBLE
  allocate(b(0:3_C_INT64_T - 1_C_INT64_T, 0:4_C_INT64_T - 1_C_INT64_T))
  b = 1.0_C_DOUBLE

  end program prog_test
  ```

### Interactive Usage with `epyccel`

In addition to the `pyccel` command, the Pyccel library provides the `epyccel` Python function, whose name stands for "embedded Pyccel": given a pure Python function `f` with type annotations, `epyccel` returns a "pyccelized" function `f_fast` that can be used in the same Python session.
For example:
```python
from pyccel.epyccel import epyccel
from mod import f

f_fast = epyccel(f)
```
In practice `epyccel` copies the contents of `f` into a temporary python file in the `__epyccel__` directory, then it calls the `pyccel` command to generate a Python C extension module that contains a single pyccelized function, and finally it imports this function and returns it to the caller.

#### Example 4: quicksort algorithm

Let's assume that we have a `quicksort` function in a pure Python module `mod.py`:
```python
def quicksort(a: 'float[:]', lo: int, hi: int):
    i = lo
    j = hi
    while i < hi:
        pivot = a[(lo + hi) // 2]
        while i <= j :
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                tmp  = a[i]
                a[i] = a[j]
                a[j] = tmp
                i += 1
                j -= 1
        if lo < j:
            quicksort(a, lo, j)
        lo = i
        j = hi
```
We now import this function from an interactive IPython terminal and pyccelize it with the `epyccel` command.
We then use the two functions (original and pyccelized) to sort a random array of 100 elements.
Finally we compare the timings obtained on an Intel Core 3 architecture.
```
In [1]: from numpy.random import random

In [2]: from mod import quicksort

In [3]: from pyccel.epyccel import epyccel

In [4]: quicksort_fast = epyccel(quicksort)

In [5]: x = random(100)

In [6]: %timeit y = x.copy()
435 ns ± 4.75 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

In [7]: %timeit y = x.copy(); quicksort(y, 0, 99)
280 µs ± 1.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

In [8]: %timeit y = x.copy(); quicksort_fast(y, 0, 99)
1.76 µs ± 10 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

In [9]: (280 - 0.435) / (1.76 - 0.435)
Out[9]: 210.99245283018868
```
After subtracting the amount of time required to create an array copy from the given times, we can conclude that the pyccelized function is approximately 210 times faster than the original Python function.

## Other Features

Pyccel's generated code can use parallel multi-threading through [OpenMP](https://en.wikipedia.org/wiki/OpenMP); please read [our documentation](https://github.com/pyccel/pyccel/blob/master/tutorial/openmp.md) for more details.

We are also working on supporting [MPI](https://en.wikipedia.org/wiki/Open_MPI), [LAPACK](https://en.wikipedia.org/wiki/LAPACK)/[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms), and [OpenACC](https://en.wikipedia.org/wiki/OpenACC).

In the future we plan to support GPU programming with [CUDA](https://en.wikipedia.org/wiki/CUDA) and [task-based parallelism](https://en.wikipedia.org/wiki/Task_parallelism).

## Getting Help

If you face problems with pyccel, please take the following steps:

1.  Consult our documention in the tutorial directory;
2.  Send an email message to pyccel@googlegroups.com;
3.  Open an issue on GitHub.

Thank you!
