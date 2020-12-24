Welcome to
# Pyccel

 ## What is Pyccel

-   Static compiler for Python 3, using Fortran or C as backend language.
-   Started as small open-source project in 2018 at IPP Garching.
-   Public repository is now hosted on GitHub, freely available for download.

 ## Python’s objects, variables, and garbage collection

 Python is an **interpreted** language, **dynamically typed** and **garbage-collected**.

 ### Python object

-   Is created by the Python interpreter when `object.__new__()` is invoked (e.g. as a result of an expression).
-   Can be either mutable or immutable, but its type never changes.
-   Resides in memory and has a **reference count**.
-   Is accessed through one or more Python variables.
-   Is destroyed by the garbage collector when its reference count drops to zero.

For more details about Python object, see [this](https://docs.python.org/3/tutorial/classes.html).

 ### Python variable

-   Is a reference to a Python object in memory.

-   Is created with an assignment operation `x = expr`:
    -   If the variable `x` already exists, the interpreter reduces the reference count of its object
    -   Otherwise a new variable `x` is created, which references the value of expr.
    -   The reference count increased with an assignment operator, in argument passing or appending an object to a list.

-   The type of the variable can be changed at run-time, because python is a dynamically typed language.

-   Tan be destroyed with the command del `x`.

For more details about Python variables, see [this](https://www.w3schools.com/python/python_variables.asp).

 ## Statically typed languages
	
A language is statically-typed if the type of a variable is known at compile-time instead of run-time. Common examples of statically-typed languages include Java, C, C++, FORTRAN, Pascal and Scala, on the other hand, in python the type of a variable is known at run-time, that's why we need to collect the garbage in the generated code, and raise some warnings/errors for the conflicts that can occur between dynamically typed languages (python) and statically typed languages(C/Fortran). See [this](https://en.wikipedia.org/wiki/Type_system#:~:text=In%20programming%20languages%2C%20a%20type,%2C%20expressions%2C%20functions%20or%20modules.) and [this](https://android.jlelse.eu/magic-lies-here-statically-typed-vs-dynamically-typed-languages-d151c7f95e2b#:~:text=Static%20typed%20languages,%2C%20FORTRAN%2C%20Pascal%20and%20Scala.) for more details.

 ## Installation (see [README](https://github.com/pyccel/pyccel/blob/master/README.rst) file)

 ## Command line usage

-   Open a terminal app, iterm or terminal for MacOs, terminal for Linux.

-   After the installation, type `pyccel`, the usage should be shown, If this is the case then the installation has succeeded.

-   Create a Python file that contains simple lines of code to see what will happen.
    -   To create the file `touch file_name.py`.
    -   Use your favorite text editor to fill the file with some lines of code or just type `echo 'print("hello, world!\n")' > file_name.py` for a quick test, `cat file_name.py` to make sure that your lines in the file.
    -   To generate the C/Fortran code form your Python code, type `pyccel file_name.py` or `pyccel file_name.py --language fortran` to generate Fortran code, and `pyccel file_name.py --language c` to generate C code.
    -   No problems ? You should discover `__pyccel__` the directory that contains your generated code and some other stuff.

E.g (using `@types` decorator and a recursive function with a typed return), To specify the types of the function arguments and its return, we need to import the `@types` decorator from pyccel.decorators (as you can see in `file_name.py` first line) and then specify the types for each function argument in `@types` using the following:
-   The syntax for the decorator is: `@types('1stArgType', '2ndArgType', 'NthArgType', results='return_type')`, or to declare arrays: `@types('1stArgType[:]', '2ndArgType[:,:]', 'NthArgType[dimensions]', results='return_type')`, The expression `[:]` means that the array has 1 dimension. 2 dimensions would be specified with `[:,:]`. The number of dimensions of an array is equal to the number of comma-separated colons in the square brackets. So `arr[:,:,:]` means that the array `arr` has 3 dimensions and so on.
-   In the function we just use python syntax `def fun('1stArg', '2ndArg', 'NthArg')`.
   
In `@types` decorator, pyccel supports the following data types: real, double, float, pythonfloat, float32, float64, pythoncomplex, complex, complex64, complex128, int8, int16, int32, int64, int, pythonint, integer, bool, pythonbool.
   
  python code:

  file_name.py
  ```python
  from pyccel.decorators import types

  @types('int', results='int')
  def factorial(n):
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
