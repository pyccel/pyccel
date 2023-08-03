# Function-pointers as arguments

Note: before reading this you should have read [Installation and Command Line Usage](https://github.com/pyccel/pyccel/blob/master/tutorial/quickstart.md#installation)

In order to support passing [function-pointers](https://en.wikipedia.org/wiki/Function_pointer) as arguments, Pyccel needs the user to define the type of the passed function-pointers. This can be done by using the syntax `def function_name(func1_name : '(func1_return_type)(func1_arguments_types)', func2_name : '(func2_return_type)(func2_arguments_types)', ..., arg1, arg2, ...)` or using a function-header `#$ header function function_name((func1_return_type)(func1_arguments), (func2_return_type)(func2_arguments), ..., var1_type, var2_type, ...)`. Here is how Pyccel converts that feature:

In this example we will use short syntax for this feature:

```python
def high_int_int_1(function1: '(int)(int)', function2: '(int)(int)', a: 'int'):
    x = function1(a)
    y = function2(a)
    return x + y
```

Here is the C generated code:

```C
#include "boo.h"
#include <stdlib.h>
#include <stdint.h>


/*........................................*/
int64_t high_int_int_1(int64_t (*function1)(int64_t ), int64_t (*function2)(int64_t ), int64_t a)
{
    int64_t x;
    int64_t y;
    x = function1(a);
    y = function2(a);
    return x + y;
}
/*........................................*/
```

Here is the Fortran equivalent:

```Fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  function high_int_int_1(function1, function2, a) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    integer(i64), value :: a
    integer(i64) :: x
    integer(i64) :: y

    interface
      function function1(in_0000) result(out_0000)
        use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
        integer(i64) :: out_0000
        integer(i64), value :: in_0000
      end function function1

      function function2(in_0001) result(out_0001)
        use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
        integer(i64) :: out_0001
        integer(i64), value :: in_0001
      end function function2
    end interface

    x = function1(a)
    y = function2(a)
    Out_0001 = x + y
    return

  end function high_int_int_1
  !........................................

end module boo
```

Here is the version using a function-header:

```Python
#$ header function high_int_int_1((int)(int), (int)(int), int)
def high_int_int_1(function1, function2, a):
    x = function1(a)
    y = function2(a)
    return x + y
```

Here is the generated C code:

```C
#include "boo.h"
#include <stdlib.h>
#include <stdint.h>


/*........................................*/
int64_t high_int_int_1(int64_t (*function1)(int64_t ), int64_t (*function2)(int64_t ), int64_t a)
{
    int64_t x;
    int64_t y;
    x = function1(a);
    y = function2(a);
    return x + y;
}
/*........................................*/
```

And the Fortran equivalent:

```Fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  function high_int_int_1(function1, function2, a) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    integer(i64), value :: a
    integer(i64) :: x
    integer(i64) :: y

    interface
      function function1(in_0000) result(out_0000)
        use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
        integer(i64) :: out_0000
        integer(i64), value :: in_0000
      end function function1

      function function2(in_0001) result(out_0001)
        use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
        integer(i64) :: out_0001
        integer(i64), value :: in_0001
      end function function2
    end interface

    x = function1(a)
    y = function2(a)
    Out_0001 = x + y
    return

  end function high_int_int_1
  !........................................

end module boo
```

## Pyccel Optimisation Case

Now, we will see a special case that is optimised by Pyccel (not optimised in C yet):

In this example, Pyccel will recognise that foo doesn't change `x`, so it will automatically add `const` or `intent(in)` (depending on the language: C/Fortran) to the data type of `x`. This provides useful information for C/Fortran compilers to make optimisations to the code:

```python
def foo(x: 'int[:]', i: 'int'):
    #some code
    return x[i]

def func1(a: 'int[:]', b: 'int', func_arg : '(int)(int[:], int)' = foo):
    x = foo(a, b)
    return x
```

The Fortran equivalent:

```Fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  function foo(x, i) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    integer(i64), intent(in) :: x(0:)
    integer(i64), value :: i

    !some code
    Out_0001 = x(i)
    return

  end function foo
  !........................................

  !........................................
  function func1(a, b, func_arg) result(x)

    implicit none

    integer(i64) :: x
    integer(i64), intent(in) :: a(0:)
    integer(i64), value :: b

    interface
      function func_arg(in_0000, in_0001) result(out_0000)
        use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
        integer(i64) :: out_0000
        integer(i64), intent(inout) :: in_0000(0:)
        integer(i64), value :: in_0001
      end function func_arg
    end interface

    x = foo(a, b)
    return

  end function func1
  !........................................

end module boo
```

Note that the argument in the interface in `func1` has a different [intent](https://pages.mtu.edu/~shene/COURSES/cs201/NOTES/chap07/intent.html). The argument `x` in `foo` shouldn't be `intent(in)`, but rather `intent(inout)`. However as Pyccel detected that `x` won't change in `foo`, the perfect case for `x` is to be an `intent(in)` rather than `intent(inout)`
Now we will tell Pyccel to create a program by adding `if __name__ == '__main__':` to the Python code, and see what problem a mismatch in intent will cause:

```python
import numpy as np

def foo(x: 'int[:]', i: 'int'):
    #some code
    return x[i]

def func1(a: 'int[:]', b: 'int', func_arg : '(int)(int[:], int)' = foo):
    x = foo(a, b)
    return x

if __name__ == '__main__':
    a = np.ones(5, dtype=int)
    b = 4
    func1(a, b, foo)
```

After trying to pyccelise the Python code above, here are the generated codes:

The generated code of the Fortran module:

```fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  function foo(x, i) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    integer(i64), intent(in) :: x(0:)
    integer(i64), value :: i

    !some code
    Out_0001 = x(i)
    return

  end function foo
  !........................................

  !........................................
  function func1(a, b, func_arg) result(x)

    implicit none

    integer(i64) :: x
    integer(i64), intent(in) :: a(0:)
    integer(i64), value :: b

    interface
      function func_arg(in_0000, in_0001) result(out_0000)
        use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
        integer(i64) :: out_0000
        integer(i64), intent(inout) :: in_0000(0:)
        integer(i64), value :: in_0001
      end function func_arg
    end interface

    x = foo(a, b)
    return

  end function func1
  !........................................

end module boo
```

The generated code of the Fortran program:

```fortran
program prog_prog_boo

  use boo

  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  integer(i64), allocatable :: a(:)
  integer(i64) :: b
  integer(i64) :: Dummy_0001

  allocate(a(0:4_i64))
  a = 1_i64
  b = 4_i64
  Dummy_0001 = func1(a, b, foo)
  if (allocated(a)) then
    deallocate(a)
  end if

end program prog_prog_boo
```

The output summary:

```sh
Error: Interface mismatch in dummy procedure 'func_arg' at (1): INTENT mismatch in argument 'in_0000'
```

The Fortran compiler couldn't make a program out of the generated code because of the mismatch of `intent` between the argument `in_0000` of the interface (function-pointer in C) in `func1`
and the argument `x` (the correspondent of `in_000`) of `foo`. This error can be fixed by adding the `const` keyword to the correspondent argument of the array `x` in `func_arg` in the Python code:

```python
import numpy as np

def foo(x: 'int[:]', i: 'int'):
    #some code
    return x[i]

def func1(a: 'int[:]', b: 'int', func_arg : '(int)(const int[:], int)' = foo):
    x = foo(a, b)
    return x

if __name__ == '__main__':
    a = np.ones(5, dtype=int)
    b = 4
    func1(a, b, foo)
```

The generated code of the Fortran module:

```fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  function foo(x, i) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    integer(i64), intent(in) :: x(0:)
    integer(i64), value :: i

    !some code
    Out_0001 = x(i)
    return

  end function foo
  !........................................

  !........................................
  function func1(a, b, func_arg) result(x)

    implicit none

    integer(i64) :: x
    integer(i64), intent(in) :: a(0:)
    integer(i64), value :: b

    interface
      function func_arg(in_0000, in_0001) result(out_0000)
        use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
        integer(i64) :: out_0000
        integer(i64), intent(in) :: in_0000(0:)
        integer(i64), value :: in_0001
      end function func_arg
    end interface

    x = foo(a, b)
    return

  end function func1
  !........................................

end module boo
```

The generated code of the Fortran program:

```fortran
program prog_prog_boo

  use boo

  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  integer(i64), allocatable :: a(:)
  integer(i64) :: b
  integer(i64) :: Dummy_0001

  allocate(a(0:4_i64))
  a = 1_i64
  b = 4_i64
  Dummy_0001 = func1(a, b, foo)
  if (allocated(a)) then
    deallocate(a)
  end if

end program prog_prog_boo
```

## Getting Help

If you face problems with Pyccel, please take the following steps:

1.  Consult our documentation in the tutorial directory;
2.  Send an email message to pyccel@googlegroups.com;
3.  Open an issue on GitHub.

Thank you!
