# Function-pointers as arguments

Note: before reading this you should have read [Installation and Command Line Usage](https://github.com/pyccel/pyccel/blob/master/tutorial/quickstart.md#installation)

In order to support passing [function-pointers](https://en.wikipedia.org/wiki/Function_pointer) as arguments. Pyccel needs the user to define the type of the passed function-pointers, this can be done by using the following syntax `def function_name((func1_return_type)(func1_arguments), (func2_return_type)(func2_arguments), ..., var1_type, var2_type, ...)` or using a function-header `#$ header function function_name((func1_return_type)(func1_arguments), (func2_return_type)(func2_arguments), ..., var1_type, var2_type, ...)`. Here is how Pyccel converts that feature:

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

Now, We will see an advnaced example of this feature with `const`  keyword and using multiple arguments function-type:

Here is the python code:

```python
def high_int_int_1(function1: '(float)(int[:], const float[:])', function2: '(int)(int[:], float)', a: 'int[:]', b: 'const float[:]'):
    x = function1(a, b)
    y = function2(a, x)
    return x + y
```

The C generated code:

```C
#include "boo.h"
#include "ndarrays.h"
#include <stdint.h>
#include <stdlib.h>


/*........................................*/
double high_int_int_1(double (*function1)(t_ndarray , t_ndarray ), int64_t (*function2)(t_ndarray , double ), t_ndarray a, t_ndarray b)
{
    double x;
    int64_t y;
    x = function1(a, b);
    y = function2(a, x);
    return x + y;
}
/*........................................*/
```

The Fortran equivalent:

```Fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , f64 => &
      C_DOUBLE
  implicit none

  contains

  !........................................
  function high_int_int_1(function1, function2, a, b) result(Out_0001)

    implicit none

    real(f64) :: Out_0001
    integer(i64), intent(inout) :: a(0:)
    real(f64), intent(in) :: b(0:)
    real(f64) :: x
    integer(i64) :: y

    interface
      function function1(in_0000, in_0001) result(out_0000)
        use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , f64 &
      => C_DOUBLE
        real(f64) :: out_0000
        integer(i64), intent(inout) :: in_0000(0:)
        real(f64), intent(in) :: in_0001(0:)
      end function function1

      function function2(in_0002, in_0003) result(out_0001)
        use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , f64 &
      => C_DOUBLE
        integer(i64) :: out_0001
        integer(i64), intent(inout) :: in_0002(0:)
        real(f64), value :: in_0003
      end function function2
    end interface

    x = function1(a, b)
    y = function2(a, x)
    Out_0001 = x + y
    return

  end function high_int_int_1
  !........................................

end module boo
```

## Getting Help

If you face problems with pyccel, please take the following steps:

1.  Consult our documention in the tutorial directory;
2.  Send an email message to pyccel@googlegroups.com;
3.  Open an issue on GitHub.

Thank you!
