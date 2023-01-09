# Decorators

As Pyccel converts a dynamically typed language (Python) to statically typed languages, it has some *decorators* which the user can add in the code to provide access to low level optimisations. Here are the available decorators.

## Stack array

This decorator indicates that all arrays mentioned as arguments (of the decorator) should be stored
on the stack.

In order to store the array on the stack it is important that the size be known at the declaration.
In Fortran all declarations must occur at the start of the function.
As a result, Pyccel requires that the size of the stack array object is expressed as a function of arguments and [pure](#Pure) function results only.

This example shows how the decorators can affect the conversion of the array between the supported languages. Pyccel here is told by the decorator `stack_array` to store the array `array_in_stack` in the stack, for the array `array_in_heap` Pyccel is assuming that it should be stored in the heap:

```python
from pyccel.decorators import stack_array
import numpy as np

@stack_array('array_in_stack')
def fun1():

     #/////////////////////////
     #array stored in the stack
     #////////////////////////
     array_in_stack = np.array([1,2,3])
     #////////////////////////
     #array stored in the heap
     #////////////////////////
     array_in_heap = np.array([1,2,3])
```

This the C generated code:

```C
#include "boo.h"
#include <stdlib.h>
#include "ndarrays.h"
#include <stdint.h>
#include <string.h>


/*........................................*/
void fun1(void)
{
    int64_t array_dummy_0003[3];
    t_ndarray array_in_stack = (t_ndarray){
        .nd_int64=array_dummy_0003,
        .shape=(int64_t[]){3},
        .strides=(int64_t[1]){0},
        .nd=1,
        .type=nd_int64,
        .is_view=false
    };
    stack_array_init(&array_in_stack);
    t_ndarray array_in_heap;
    /*/////////////////////////*/
    /*array stored in the stack*/
    /*////////////////////////*/
    int64_t array_dummy_0001[] = {1, 2, 3};
    memcpy(array_in_stack.nd_int64, array_dummy_0001, array_in_stack.buffer_size);
    /*////////////////////////*/
    /*array stored in the heap*/
    /*////////////////////////*/
    array_in_heap = array_create(1, (int64_t[]){3}, nd_int64);
    int64_t array_dummy_0002[] = {1, 2, 3};
    memcpy(array_in_heap.nd_int64, array_dummy_0002, array_in_heap.buffer_size);
    free_array(array_in_heap);
}
/*........................................*/
```

This the Fortran equivalent code:

```Fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  subroutine fun1()

    implicit none

    integer(i64) :: array_in_stack(0:2_i64)
    integer(i64), allocatable :: array_in_heap(:)

    !/////////////////////////
    !array stored in the stack
    !////////////////////////
    array_in_stack = [1_i64, 2_i64, 3_i64]
    !////////////////////////
    !array stored in the heap
    !////////////////////////
    allocate(array_in_heap(0:2_i64))
    array_in_heap = [1_i64, 2_i64, 3_i64]
    if (allocated(array_in_heap)) then
      deallocate(array_in_heap)
    end if

  end subroutine fun1
  !........................................

end module boo
```

## Allow negative index

In python negative indexes allow a user to index an array starting from the back (e.g. the index -1 is the
last element of the array). Pyccel recreates this behaviour for literal indexes. However when an array is
indexed with a variable or an expression, it is impractical (and often impossible) to know at compile time whether the index is
positive or negative. As a result an if block must be added. This implies a (potentially large) performance
cost. Non-literal negative indexes are not especially common, therefore Pyccel does not add this costly
if block unless it is specifically requested. This can be done using the `allow_negative_index` decorator.

An example shows how Pyccel handles negative indexes between Python and C:

```python
from pyccel.decorators import allow_negative_index
from numpy import array

@allow_negative_index('a')
def fun1(i : int, j : int):
    #////////negative indexing allowed////////
    a = array([1,2,3,4,5,6])
    print(a[i - j])
    #////////negative indexing disallowed. the generated code can cause a crash/compilation error.////////
    b = array([1,2,3,4,5,6])
    print(b[i - j])
```

This is the generated C code:

```C
#include "boo.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "ndarrays.h"


/*........................................*/
void fun1(int64_t i, int64_t j)
{
    t_ndarray a;
    t_ndarray b;
    /*////////negative indexing allowed////////*/
    a = array_create(1, (int64_t[]){6}, nd_int64);
    int64_t array_dummy_0001[] = {1, 2, 3, 4, 5, 6};
    memcpy(a.nd_int64, array_dummy_0001, a.buffer_size);
    printf("%ld\n", GET_ELEMENT(a, nd_int64, i - j < 0 ? 6 + (i - j) : i - j));
    /*////////negative indexing disallowed. the generated can cause a crash/compilation error.////////*/
    b = array_create(1, (int64_t[]){6}, nd_int64);
    int64_t array_dummy_0002[] = {1, 2, 3, 4, 5, 6};
    memcpy(b.nd_int64, array_dummy_0002, b.buffer_size);
    printf("%ld\n", GET_ELEMENT(b, nd_int64, i - j));
    free_array(a);
    free_array(b);
}
/*........................................*/
```

And here is the equivalent Fortran code:

```fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  subroutine fun1(i, j)

    implicit none

    integer(i64), value :: i
    integer(i64), value :: j
    integer(i64), allocatable :: a(:)
    integer(i64), allocatable :: b(:)

    !////////negative indexing allowed////////
    allocate(a(0:5_i64))
    a = [1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64]
    print *, a(merge(6_i64 + (i - j), i - j, i - j < 0_i64))
    !////////negative indexing disallowed. the generated can cause a crash/compilation error.////////
    allocate(b(0:5_i64))
    b = [1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64]
    print *, b(i - j)
    if (allocated(a)) then
      deallocate(a)
    end if
    if (allocated(b)) then
      deallocate(b)
    end if

  end subroutine fun1
  !........................................

end module boo
```

## Elemental

In Python it is often the case that a function with scalar arguments and a single scalar output (if any) is also able to accept NumPy arrays with identical rank and shape - in such a case the scalar function is simply applied element-wise to the input arrays. In order to mimic this behaviour in the generated C or Fortran code, Pyccel provides the decorator `elemental`.

Important note: applying the `elemental` decorator to a function will not make a difference to the C translation of the function definition itself since C doesn't have the elementwise feature. However, Pyccel implements that functionality by calling the function in a `for` loop when an array argument is passed. In the following example, we will use the function `square` where `@elemental` will be useful:

Here is the python code:

```python
from pyccel.decorators import elemental
import numpy as np

@elemental
@types(float)
def square(x):
    s = x*x
    return s


def square_in_array():
    a = np.ones(5)
    square(a)
```

The generated C code:

```C
#include "boo.h"
#include "ndarrays.h"
#include <stdlib.h>
#include <stdint.h>


/*........................................*/
double square(double x)
{
    double s;
    s = x * x;
    return s;
}
/*........................................*/
/*........................................*/
void square_in_array(void)
{
    t_ndarray a;
    t_ndarray Dummy_0001;
    int64_t i_0001;
    a = array_create(1, (int64_t[]){5}, nd_double);
    array_fill((double)1.0, a);
    Dummy_0001 = array_create(1, (int64_t[]){5}, nd_double);
    for (i_0001 = 0; i_0001 < 5; i_0001 += 1)
    {
        GET_ELEMENT(Dummy_0001, nd_double, i_0001) = square(GET_ELEMENT(a, nd_double, i_0001));
    }
    free_array(a);
    free_array(Dummy_0001);
}
/*........................................*/
```

Fortran has the elementwise feature which is presented in the code as function prefix `elemental`. So any function marked as an elemental one can be used to operate on the arrays. See more about [elemental](https://www.fortran90.org/src/best-practices.html#element-wise-operations-on-arrays-using-subroutines-functions).

```Fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE , i64 => &
      C_INT64_T
  implicit none

  contains

  !........................................
  elemental function square(x) result(s)

    implicit none

    real(f64) :: s
    real(f64), value :: x

    s = x * x
    return

  end function square
  !........................................

  !........................................
  subroutine square_in_array()

    implicit none

    real(f64), allocatable :: a(:)
    real(f64), allocatable :: Dummy_0001(:)

    allocate(a(0:4_i64))
    a = 1.0_f64
    allocate(Dummy_0001(0:4_i64))
    Dummy_0001 = square(a)
    if (allocated(a)) then
      deallocate(a)
    end if
    if (allocated(Dummy_0001)) then
      deallocate(Dummy_0001)
    end if

  end subroutine square_in_array
  !........................................

end module boo
```

## Pure

The decorator `pure` indicates that the function below the decorator is a pure one. This means that the function should return identical output values for identical input arguments and that it has no side effects (e.g. print) in its application.

Here is a simple usage example:

```python
from pyccel.decorators import pure

@pure
@types(float)
def square(x):
    s = x*x
    return s
```

This decorator has no effect on the C code as the concept of a `pure` function does not exist in the language.
On the other hand, the `pure` decorator does affect the Python/Fortran conversion. The function prefix `pure`, which may allow the Fortran compiler to generate faster machine code, is added. See [here](http://www.lahey.com/docs/lfpro79help/F95ARPURE.htm#:~:text=Fortran%20procedures%20can%20be%20specified,used%20in%20the%20procedure%20declaration.) for more information about the `pure` keyword in Fortran:

```Fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE
  implicit none

  contains

  !........................................
  pure function square(x) result(s)

    implicit none

    real(f64) :: s
    real(f64), value :: x

    s = x * x
    return

  end function square
  !........................................

end module boo
```

## Inline

The `@inline` decorator indicates that the body of a function should be printed directly when it is called rather than passing through an additional function call. This can be useful for code optimisation.

### Basic Example

Here is a simple usage example:
```python
def f():
    @inline
    def cube(s : int):
        return s * s * s
    a = cube(3) + 2
    return a
```

The generated Fortran code:
```fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  function f() result(a)

    implicit none

    integer(i64) :: a

    a = 3_i64 * 3_i64 * 3_i64 + 2_i64
    return

  end function f
  !........................................

end module boo
```

The generated C code:
```c
#include "boo.h"
#include <stdlib.h>
#include <stdint.h>


/*........................................*/
int64_t f(void)
{
    int64_t a;

    a = 3 * 3 * 3 + 2;
    return a;
}
/*........................................*/
```

### Example with Local Variables

The following complicated example shows the handling of arrays and local variables

```python
import numpy as np
from pyccel.decorators import inline

@inline
def fill_pi(a : 'float[:]'):
    pi = 3.14159
    for i in range(a.shape[0]):
        a[i] = pi

def f():
    a = np.empty(4)
    fill_pi(a)
    pi = 3.14
    print(a,pi)
```

The generated Fortran code:
```fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE , i64 => &
      C_INT64_T
  implicit none

  contains

  !........................................
  !........................................

  !........................................
  subroutine f()

    implicit none

    real(f64), allocatable :: a(:)
    real(f64) :: pi
    real(f64) :: pi_0001
    integer(i64) :: i_0002

    allocate(a(0:3_i64))
    pi_0001 = 3.14159_f64
    do i_0002 = 0_i64, size(a, kind=i64) - 1_i64, 1_i64
      a(i_0002) = pi_0001
    end do
    pi = 3.14_f64
    print *, a, pi
    if (allocated(a)) then
      deallocate(a)
    end if

  end subroutine f
  !........................................

end module boo
```

The generated C code:
```c
#include "boo.h"
#include "ndarrays.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>


/*........................................*/
void f(void)
{
    t_ndarray a = {.shape = NULL};
    double pi;
    double pi_0001;
    int64_t i_0002;
    int64_t i_0003;
    a = array_create(1, (int64_t[]){4}, nd_double);
    pi_0001 = 3.14159;
    for (i_0002 = 0; i_0002 < a.shape[0]; i_0002 += 1)
    {
        GET_ELEMENT(a, nd_double, i_0002) = pi_0001;
    }
    pi = 3.14;
    printf("%s", "[");
    for (i_0003 = 0; i_0003 < 3; i_0003 += 1)
    {
        printf("%.12lf ", GET_ELEMENT(a, nd_double, i_0003));
    }
    printf("%.12lf]", GET_ELEMENT(a, nd_double, 3));
    printf("%.12lf\n", pi);
    free_array(a);
}
/*........................................*/
```

### Example with Optional Variables

Finally we present an example with optional variables:
```python
@inline
def get_val(x : int = None , y : int = None):
    if x is None :
        a = 3
    else:
        a = x
    if y is not None :
        b = 4
    else:
        b = 5
    return a + b

def f():
    a = get_val(2,7)
    b = get_val()
    c = get_val(6)
    d = get_val(y=0)
    return a,b,c,d
```

The generated Fortran code:
```fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  !........................................

  !........................................
  subroutine f(a, b, c, d)

    implicit none

    integer(i64), intent(out) :: a
    integer(i64), intent(out) :: b
    integer(i64), intent(out) :: c
    integer(i64), intent(out) :: d
    integer(i64) :: a_0001
    integer(i64) :: b_0002
    integer(i64) :: a_0003
    integer(i64) :: b_0004
    integer(i64) :: a_0005
    integer(i64) :: b_0006
    integer(i64) :: a_0007
    integer(i64) :: b_0008

    a_0001 = 2_i64
    b_0002 = 4_i64
    a = a_0001 + b_0002
    a_0003 = 3_i64
    b_0004 = 5_i64
    b = a_0003 + b_0004
    a_0005 = 6_i64
    b_0006 = 4_i64
    c = a_0005 + b_0006
    a_0007 = 3_i64
    b_0008 = 5_i64
    d = a_0007 + b_0008
    return

  end subroutine f
  !........................................

end module boo
```

The generated C code:
```c
```

## Getting Help

If you face problems with Pyccel, please take the following steps:

1.  Consult our documentation in the tutorial directory;
2.  Send an email message to pyccel@googlegroups.com;
3.  Open an issue on GitHub.

Thank you!
