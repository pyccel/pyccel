# Pyccel's Quickstart Guide

Pyccel is a **static compiler** for Python 3, using Fortran or C as a backend language, with a focus on high-performance computing (HPC) applications.

Pyccel's main goal is to resolve the principal bottleneck in scientific computing: the transition from **prototype** to **production**.
Programmers usually develop their prototype code in a user-friendly interactive language like Python, but their final application requires an HPC implementation and therefore a new production code.
In most cases this is written in a statically compiled language like Fortran/C/C++, and it uses SIMD vectorisation, parallel multi-threading, MPI parallelisation, GPU offloading, etc.

We believe that this expensive process can be avoided, or at least drastically reduced, by using Pyccel to accelerate the most computationally intensive parts of the Python prototype.
Not only is the Pyccel-generated Fortran or C code very fast, but it is **human-readable**; hence the expert programmer can easily profile the code on the target machine and further optimise it.

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

-   <https://docs.python.org/3/tutorial/classes.html#a-word-about-names-and-objects>
-   <https://docs.python.org/3/reference/datamodel.html#objects-values-and-types>
-   <https://en.wikipedia.org/wiki/Type_system>
-   <https://android.jlelse.eu/magic-lies-here-statically-typed-vs-dynamically-typed-languages-d151c7f95e2b>

## How Pyccel works

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

or to declare NumPy arrays

```python
def fun(arg1: 'type1[:]', arg2: 'type2[:,:]', ..., argN: 'typeN[dimensions]') -> 'return_type':
```

The number of dimensions of an array is equal to the number of comma-separated colons in the square brackets.
So `arr[:]` means that the array `arr` has 1 dimension, `arr[:,:]` means that it has 2 dimensions and so on.
In general string type hints must be used to provide Pyccel with information about the number of dimensions.

For scalar variables and arrays Pyccel supports the following data types:

-   built-in data types: `bool`, `int`, `float`, `complex`
-   NumPy integer types: `int8`, `int16`, `int32`, `int64`
-   NumPy real types: `float32`, `float64`, `double`
-   NumPy complex types: `complex64`, `complex128`

Scalar variables can also be specified without using string type annotations:

```python
def fun(arg1: type1, arg2: type2, ..., argN: typeN) -> return_type:
```

Such type annotations are compatible with mypy and as such are recommended for use where this is possible. NumPy does not yet provide mypy compatible type annotations which provide sufficient information for Pyccel so string annotations are still necessary for arrays.

:warning: When running Pyccel in interactive mode with `epyccel` anything imported before the function being translated is not copied into the file which will be translated. Using non-string types here is therefore likely to generate errors as the necessary imports for these objects will be missing.

## How to use Pyccel

### Installation

Pyccel's official releases can be downloaded and installed from PyPI (the Python Package Index) using `pip`.
To get the latest (trunk) version of Pyccel, one can clone the `git` repository from GitHub and checkout the `devel` branch.
Detailed installation instructions are found [here](https://github.com/pyccel/pyccel/blob/devel/docs/installation.md).

### Command Line Usage

After installation, the `pyccel` command will be available on a terminal app (iterm or terminal for macOS, terminal for Linux).
After typing `pyccel`, the usage should be displayed on the terminal; if this is the case then the installation has succeeded.
In essence the `pyccel` command translates the given Python file to a Fortran or C file, and then compiles the generated code to a Python C extension module or a simple executable.

#### Example 1: "Hello World" program

In this first example we create the simplest Python script `hello.py`, which prints `Hello, world!` followed by an empty line.
We use the `pyccel` command to translate the Python code to a C program in file `hello.c`, which is placed in the new `__pyccel__` directory.
By default Pyccel also compiles the C code into an executable named `hello`, which is placed in the same directory as the original file:

```bash
$ printf 'if __name__ == "__main__": \n print("Hello, world!\\n")' > hello.py
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
├── __pyccel__
│   ...
│   ├── prog_hello.c
│   ...
├── hello
└── hello.py
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
pyccel mod.py --language c
```

By default Pyccel also compiles the C code into a Python C extension module named `mod.<TAG>.so`, which is placed in the same directory as `mod.py`.
To achieve this Pyccel generates the additional files `mod_wrapper.c` (which interacts directly with the CPython API) and `setup_mod.py` (which defines the build procedure for the extension module), as well as a `build` directory.

If the command `import mod` is now given to the Python interpreter, this will import the Python C extension module `mod.<TAG>.so` instead of the pure Python module `mod.py`.

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

#### Example 3: matrix-matrix multiplication with OpenMP

Let's now see a more complicated example, where the Python module `mod.py` contains a function that performs the matrix-matrix multiplication between two arrays `a` and `b`, and writes the result into the array `c`:

-   The three function's arguments are 2D NumPy arrays of double-precision floating point numbers
-   Matrices `a` and `c` use C ordering (row-major), matrix `b` uses Fortran ordering (column-major)
-   Since matrix `c` is modified by the function, it has `intent(inout)` in Fortran
-   Comments starting with `#$ omp` are translated to OpenMP pragmas

```python
def matmul(a: 'float[:,:](order=C)',
           b: 'float[:,:](order=F)',
           c: 'float[:,:](order=C)'):

    m, p = a.shape
    q, n = b.shape
    r, s = c.shape

    if p != q or m != r or n != s:
        return -1

    #$ omp parallel
    #$ omp for schedule(runtime)
    for i in range(m):
        for j in range(n):
            c[i, j] = 0.0
            for k in range(p):
                c[i, j] += a[i, k] * b[k, j]
    #$ omp end parallel

    return 0
```

We now translate this file to Fortran, and compile it to a Python C extension module, using the command

```bash
pyccel mod.py --language fortran
```

The flag `--language fortran` may be omitted, as Pyccel uses Fortran as the default backend language.
These are the contents of the current directory:

```bash
$ tree .
.
├── mod.cpython-36m-x86_64-linux-gnu.so
├── mod.py
└── __pyccel__
    ├── bind_c_mod.f90
    ├── bind_c_mod.o
    ├── build
    │   ├── lib.linux-x86_64-3.6
    │   └── temp.linux-x86_64-3.6
    │       └── mod_wrapper.o
    ├── mod.f90
    ├── mod.mod
    ├── mod.o
    ├── mod_wrapper.c
    └── setup_mod.py

4 directories, 10 files
```

And this is the Fortran file `mod.f90`:

```fortran
module mod


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , f64 => &
        C_DOUBLE
  implicit none

  contains

  !........................................
  function matmul(a, b, c) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    real(f64), intent(in) :: a(0:,0:)
    real(f64), intent(in) :: b(0:,0:)
    real(f64), intent(inout) :: c(0:,0:)
    integer(i64) :: m
    integer(i64) :: p
    integer(i64) :: q
    integer(i64) :: n
    integer(i64) :: r
    integer(i64) :: s
    integer(i64) :: i
    integer(i64) :: j
    integer(i64) :: k

    m = size(a, 2_i64, i64)
    p = size(a, 1_i64, i64)
    q = size(b, 1_i64, i64)
    n = size(b, 2_i64, i64)
    r = size(c, 2_i64, i64)
    s = size(c, 1_i64, i64)
    if (p /= q .or. m /= r .or. n /= s) then
      Out_0001 = -1_i64
      return
    end if
    !$omp parallel
    !$omp do schedule(runtime)
    do i = 0_i64, m - 1_i64
      do j = 0_i64, n - 1_i64
        c(j, i) = 0.0_f64
        do k = 0_i64, p - 1_i64
          c(j, i) = c(j, i) + a(k, i) * b(k, j)
        end do
      end do
    end do
    !$omp end do
    !$omp end parallel
    Out_0001 = 0_i64
    return

  end function matmul
  !........................................

end module mod
```

### Interactive Usage with `epyccel`

In addition to the `pyccel` command, the Pyccel library provides the `epyccel` Python function, whose name stands for "embedded Pyccel": given a pure Python function `f` with type annotations, `epyccel` returns a "pyccelised" function `f_fast` that can be used in the same Python session.
For example:

```python
from pyccel import epyccel
from mod import f

f_fast = epyccel(f)
```

In practice `epyccel` copies the contents of `f` into a temporary Python file in the `__epyccel__` directory.
Although some basic information (scalar literals, type hints, imports of supported libraries) can be deduced from the context, it is important that all other objects are defined inside the function when using `epyccel`.
Once the file has been copied, `epyccel` calls the `pyccel` command to generate a Python C extension module that contains a single pyccelised function.
Then finally, it imports this function and returns it to the caller.

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

We now import this function from an interactive IPython terminal and pyccelise it with the `epyccel` command.
We then use the two functions (original and pyccelised) to sort a random array of 100 elements.
Finally we compare the timings obtained on an Intel Core 3 architecture.

```bash
In [1]: from numpy.random import random
In [2]: from mod import quicksort
In [3]: from pyccel import epyccel

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

After subtracting the amount of time required to create an array copy from the given times, we can conclude that the pyccelised function is approximately 210 times faster than the original Python function.

### Interactive Usage with `lambdify`

While Pyccel is usually used to accelerate Python code, it is also possible to accelerate other expressions. The Pyccel library provides the `lambdify` Python function. This function is similar to SymPy's [`lambdify`](https://docs.sympy.org/latest/modules/utilities/lambdify.html) function, given a SymPy expression `f` and type annotations, `lambdify` returns a "pyccelised" function `f_fast` that can be used in the same Python session.
For example:

```python
from typing import TypeVar
import numpy as np
import sympy as sp
from pyccel import lambdify

T = TypeVar('T', 'float[:]', 'float[:,:]')

x = sp.Symbol('x')
expr = x**2 + x*5
f = lambdify(expr, {x : 'float'})
print(f(3.0))

expr = x-x
f2 = lambdify(expr, {x : 'float'}, result_type = 'float')
print(f2(3.0))

expr = x**2 + x*5 + 4.5
f3 = lambdify(expr, {x : 'T'}, context_dict = {'T': T})
x_1d = np.ones(4)
x_2d = np.ones((4,2))
print(f3(x_1d))
print(f3(x_2d))
```

In practice `lambdify` uses SymPy's `NumPyPrinter` to generate code which is passed to the `epyccel` function. The `epyccel` function copies the code into a temporary Python file in the `__epyccel__` directory.
Once the file has been copied, `epyccel` calls the `pyccel` command to generate a Python C extension module that contains a single pyccelised function.
Then finally, it imports this function and returns it to the caller.

In order to make functions even faster it may be desirable to avoid unnecessary allocations inside the function. This functionality is similar to using an `out` argument in NumPy. Pyccel makes this functionality possible through the use of the `use_out` argument.
For example:

```python
import numpy as np
import sympy as sp
from pyccel import lambdify

x = sp.Symbol('x')
expr = x**2 + x*5
f = lambdify(expr, {x : 'float[:,:]'}, result_type = 'float[:,:]', use_out = True)
x_2d = np.ones((4,2))
y_2d = np.empty_like(x_2d)
f(x_2d, y_2d)
print(y_2d)
```

## Other Features

Pyccel's generated code can use parallel multi-threading through [OpenMP](https://en.wikipedia.org/wiki/OpenMP); please read [our documentation](https://github.com/pyccel/pyccel/blob/devel/docs/openmp.md) for more details.

We are also working on supporting [MPI](https://en.wikipedia.org/wiki/Open_MPI), [LAPACK](https://en.wikipedia.org/wiki/LAPACK)/[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms), and [OpenACC](https://en.wikipedia.org/wiki/OpenACC).

In the future we plan to support GPU programming with [CUDA](https://en.wikipedia.org/wiki/CUDA) and [task-based parallelism](https://en.wikipedia.org/wiki/Task_parallelism).

## Cleaning up the environment

Pyccel generates various files, in order to help clean up the environment after using it, we therefore also provide the command line tool: `pyccel-clean`.
This tool removes all folders whose name begins with `__pyccel__` or `__epyccel__` and can also be used to remove shared libraries and programs.

## Getting Help

If you face problems with Pyccel, please take the following steps:

1.  Consult our documentation in the  [`docs/`](https://github.com/pyccel/pyccel/blob/devel/docs) directory;
2.  Send an email message to <pyccel@googlegroups.com>;
3.  Open an issue on GitHub.

Thank you!
