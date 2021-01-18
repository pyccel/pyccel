# Welcome to Pyccel

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

-   Is a reference to a Python object in memory.

-   Is created with an assignment operation `x = expr`:

    -   If the variable `x` already exists, the interpreter reduces the reference count of its object
    -   Otherwise a new variable `x` is created, which references the value of expr.
    -   The variable `x` is then modified to reference the object referenced by `expr` and the reference count of this object is increased

-   The type of the variable can be changed at run-time, because Python is a dynamically typed language.

-   Can be destroyed with the command del `x`.

### Statically Typed Languages
	
A language is statically-typed if the type of a variable is known at compile-time instead of run-time.
Common examples of statically-typed languages include Java, C, C++, FORTRAN, Pascal and Scala.
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
Pyccel can then perform type inference on other Python code that uses that function, because the type of its result will be already known.

As described in the next section, the programmer has various ways to provide the argument type information to Pyccel.
For example, Python-style type annotations can be used:
```python
def factorial(n : int):
    r = 1
    for i in range(1, n):
    	r *= i
    return r

def binomial_coefficient(n : int, k : int):
    num = factorial(n)
    den = factorial(k) * factorial(n - k)
    return num // den
```

## How to use Pyccel?

### Installation

Pyccel's official releases can be downloaded from PyPI (the Python Package Index) using `pip`.
To get the latest (trunk) version of Pyccel, just clone the `git` repository from GitHub and checkout the `master` branch.
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
#### Example 2: test module

E.g (using `@types` decorator/python type hints and a recursive function with a typed return), To specify the types of the function arguments and its return, we need to import the `@types` decorator from pyccel.decorators (as you can see in `file_name.py` first line) and then specify the types for each function argument in `@types` using the following:
-   The syntax for the decorator is: `@types('1stArgType', '2ndArgType', 'NthArgType', results='return_type')`, or to declare arrays: `@types('1stArgType[:]', '2ndArgType[:,:]', 'NthArgType[dimensions]', results='return_type')`, The expression `[:]` means that the array has 1 dimension. 2 dimensions would be specified with `[:,:]`. The number of dimensions of an array is equal to the number of comma-separated colons in the square brackets. So `arr[:,:,:]` means that the array `arr` has 3 dimensions and so on.
-   In the function we just use python syntax `def fun(1stArg, 2ndArg, NthArg)`.
-   Also, You can specify the function arguments types using python type hints, `def fun(1stArg: '1stArgType', 2ndArg: '2ndArgType', NthArg: 'NthArgType') -> 'returnType'`, For arrays the syntax is the same as for the decorator and string type hints must be used to provide pyccel with information about the number of dimensions.
   
In `@types` decorator, pyccel supports the following data types: real, double, float, float32, float64, complex, complex64, complex128, int8, int16, int32, int64, int, bool.

For the moment, Pyccel supports `@types` decorator(recommended) and python type hints as approaches to provide type informations to the function arguments and its return type.
   
  python code:
  Specifying the types using `@types` decorator.
  file_name.py
  ```python
  from pyccel.decorators import types

  @types('int', results='int')
  def factorial(n):
	  if n == 0: return 1
	  else : return n * factorial(n - 1)
   ```
  
  Specifying the types using python type hints.
  file_name.py
  ```python
  def factorial(n: int) -> int:
	  if n == 0: return 1
	  else : return n * factorial(n - 1)
   ``` 
   C code:
   
   file_name.c
   
   ```c
   #include <test.h>
   #include <stdlib.h>
   #include <stdint.h>

   /*........................................*/
   int64_t factorial(int64_t n)
   {
	   int64_t Out_0001;
	   if (n == 0)
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
   ```

   Fortran code:
   
   file_name.f90

   ```Fortran
   module test

   use ISO_C_BINDING

   implicit none

   contains

   !........................................
   recursive function factorial(n) result(Out_0001)

   implicit none

   integer(C_INT64_T) :: Out_0001
   integer(C_INT64_T), value :: n

   if (n == 0_C_INT64_T) then
   Out_0001 = 1_C_INT64_T
   return
   else if (.True._C_BOOL) then
   Out_0001 = n * factorial(n - 1_C_INT64_T)
   return
   end if

   end function factorial
   !........................................

   end module test
   ```

   another example with Numpy arrays:
   
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

Also, we are working on supporting [openmp](https://en.wikipedia.org/wiki/OpenMP), [openmpi](https://en.wikipedia.org/wiki/Open_MPI), [lapack](https://en.wikipedia.org/wiki/LAPACK)/[blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms), [cuda](https://en.wikipedia.org/wiki/CUDA), [openacc](https://en.wikipedia.org/wiki/OpenACC), [task-based parallelism](https://en.wikipedia.org/wiki/Task_parallelism).

Feel free to open issues for any problem you faced with pyccel, thank you.
