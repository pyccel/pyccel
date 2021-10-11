# Decorators

Because Pyccel is converting a dynamically typed language (Python) to statically types ones, It imposes some *restrictions* on the code. One of them is using decorators that have certain options in order to specify how the functions or the variables should be treated in the conversion, Here are the available decorators.

## Stack array

This decorator indicates that all arrays mentioned as argements (of the decorator) should be stored
on the stack.

This example shows how the decorators can affect the conversion of the array between the supported languages, Pyccel here is told by the decorator `stack array` to store the array `arrat_in_stack` in the stack, For the array `array_in_heap` Pyccel is assuming that it should be stored in the heap:

```python
from pyccel.decorators import stack_array

@stack_array('array_in_stack')
def fun1():

     #/////////////////////////
     #array stored in the stack
     #////////////////////////
     array_in_stack = [1,2,3]
     #////////////////////////
     #array stored in the heap
     #////////////////////////
     array_in_heap = [1,2,3]
```

This the C generated code:

```C

#include "boo.h"
#include <stdint.h>
#include <stdlib.h>
#include "ndarrays.h"
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

  end subroutine fun1
  !........................................

end module boo
```

## Allow negative index

This one indicates that all arrays mentioned as argements (of the decorator) can be accessed with negative indexes. unlike fortran you can't use negative indexes in arrays in C, But with Pyccel you can work comfotabally with them in the Python code, For the generated C code Pyccel will take care of that.

An example shows how pyccel handles negative indexes beween Python and C:

```python
from pyccel.decorators import allow_negative_index

@allow_negative_index('a')
def fun1():
    a = [1,2,3,4,5,6]
    print(a[-1])

    b = [1,2,3,4,5,6]
    print(b[-1])
```

This is the generated C code:

```C

#include "boo.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "ndarrays.h"


/*........................................*/
void fun1(void)
{
    t_ndarray a;
    t_ndarray b;
    a = array_create(1, (int64_t[]){6}, nd_int64);
    int64_t array_dummy_0001[] = {1, 2, 3, 4, 5, 6};
    memcpy(a.nd_int64, array_dummy_0001, a.buffer_size);
    printf("%ld\n", GET_ELEMENT(a, nd_int64, 5));
    b = array_create(1, (int64_t[]){6}, nd_int64);
    int64_t array_dummy_0002[] = {1, 2, 3, 4, 5, 6};
    memcpy(b.nd_int64, array_dummy_0002, b.buffer_size);
    printf("%ld\n", GET_ELEMENT(b, nd_int64, 5));
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
  subroutine fun1()

    implicit none

    integer(i64), allocatable :: a(:)
    integer(i64), allocatable :: b(:)

    allocate(a(0:5_i64))
    a = [1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64]
    print *, a(size(a, kind=i64) - 1_i64)
    allocate(b(0:5_i64))
    b = [1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64]
    print *, b(size(b, kind=i64) - 1_i64)

  end subroutine fun1
  !........................................

end module boo
```

## Elemental and pure decorators

This decorator indicates that the function bellow is an elemental one, An elemental function is a function with a single scalar operator and a scalar return value which can also be called on an array. When it is called on an array it returns the result of the function called elementwise on the array.
This decorator `pure` indicates that the function bellow is a pure one. So that function should return identical return values for identical arguments and has no side effects in its application.

Here is a simple usage example:

```python
from pyccel.decorators import elemental, pure

@pure
@elemental
@types(float)
def square(x):
    s = x*x
    return s
```

This is the C generated code, this code can be generated even without these two decorators due to the C language itself that has not the elemenwise feature in the manipulation of the arrays nor a function prefix `pure`. So these decorators will not make a big deffirence in the Python/C conversion:

```C
#include "boo.h"
#include <stdlib.h>


/*........................................*/
double square(double x)
{
    double s;
    s = x * x;
    return s;
}
/*........................................*/
```

In the other hand `pure` and `elemental` decorators has thier effects in the Python/Fortran conversion, since Fortran has the elementwise feature, so any function marked as an elemental one can be used in the arrays and the function prefix `pure`, see more about [elemental](https://www.fortran90.org/src/best-practices.html#element-wise-operations-on-arrays-using-subroutines-functions) and [pure](http://www.lahey.com/docs/lfpro79help/F95ARPURE.htm#:~:text=Fortran%20procedures%20can%20be%20specified,used%20in%20the%20procedure%20declaration.):

```Fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE
  implicit none

  contains

  !........................................
  elemental pure function square(x) result(s)

  implicit none

  real(f64) :: s
  real(f64), value :: x

  s = x * x
  return

end function square
!........................................

end module boo
```
