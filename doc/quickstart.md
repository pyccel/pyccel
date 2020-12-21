# Pyccel

 ## What is Pyccel

  - static compiler for Python 3, using Fortran or C as backend language.
  - started as small open-source project in 2018 at IPP Garching.
  - public repository is now hosted on GitHub, freely available for download.

## Python’s objects, variables, and garbage collection

 Python is an **interpreter** language, **dynamically typed** and **garbage-collected**.

 ###### Python object:

- is created by the Python interpreter when `object.__new__()` is invoked (e.g. as a result of an expression).
- can be either mutable or immutable, but its type never changes.
- resides in memory and has a **reference count**.
- is accessed through one or more Python variables.
- is destroyed by the garbage collector when its reference count drops to zero.

For more details about Python object, see [this](https://docs.python.org/3/tutorial/classes.html).

 ###### Python variable:
- is a reference to a Python object in memory.
- is created with an assignment operation `x = expr`:
  1. if the variable `x` already exists, the interpreter reduces the reference count of its object
  2. a new variable `x` is created, which references the value of expr.
can be destroyed with the command del `x`.

For more details about Python variable, see [this](https://www.w3schools.com/python/python_variables.asp).

 ## Static typed languages

A language is statically-typed if the type of a variable is known at compile-time instead of at run-time. Common examples of statically-typed languages include Java, C, C++, FORTRAN, Pascal and Scala. See [this](https://en.wikipedia.org/wiki/Type_system#:~:text=In%20programming%20languages%2C%20a%20type,%2C%20expressions%2C%20functions%20or%20modules.) and [this](https://android.jlelse.eu/magic-lies-here-statically-typed-vs-dynamically-typed-languages-d151c7f95e2b#:~:text=Static%20typed%20languages,%2C%20FORTRAN%2C%20Pascal%20and%20Scala.) for more details.

 ## Installation (see [README](https://github.com/pyccel/pyccel/blob/master/README.rst) file)

 ## Command line usage
- Open a terminal app, iterm or terminal for MacOs, terminal for Linux. 
- After the installation, type `pyccel`, the usage should be shown, then all good.
- Create a Python file that contains simple lines of code to see what will happen.
  1. To create the file `touch file_name.py`.
  2. Use your favorite text editor to fill the file with some lines of code or just type `echo 'print("hello, world!\n")' > file_name.py`        for a quick test, `cat file_name.py` to make sure that your lines in the file.
  3. To generate the C/Fortran code form your Python code, type `pyccel file_name.py` or `pyccel file_name.py --language fortran`
     to generate Fortran code, and `pyccel file_name.py --language c` to generate C code
  4. No problems ?, So you should discover `__pyccel__` the directory that contains your generated code and some other stuff.

E.g (using `@types` decorator and a recursive function with a typed return):
   
   python code:
    file_name.py
   ```
   from pyccel.decorators import types

   @types('int', results='int')
   def factorial(n):
    if n == 0: return 1
    else : return n * factorial(n - 1)
   ```
   C code:
    file_name.c
   
    ```
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
    
    ```
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

    
