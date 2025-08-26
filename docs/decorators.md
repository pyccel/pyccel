# Pyccel Decorators To Improve Performance

As Pyccel converts a dynamically typed language (Python) to statically typed languages, it has some *decorators* which the user can add in the code to provide access to low level optimisations. Here are the available decorators.

## Stack array

This decorator indicates that all arrays mentioned as arguments (of the decorator) should be stored
on the stack.

In order to store the array on the stack it is important that the size be known at the declaration.
In Fortran all declarations must occur at the start of the function.
As a result, Pyccel requires that the size of the stack array object is expressed as a function of arguments and [pure](#pure) function results only.

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
/*........................................*/
void fun1(void)
{
    int64_t array_in_stack_ptr[INT64_C(3)];
    array_int64_1d array_in_stack = cspan_md_layout(c_ROWMAJOR, array_in_stack_ptr, INT64_C(3));
    array_int64_1d array_in_heap = {0};
    int64_t* array_in_heap_ptr;
    /*/////////////////////////*/
    /*array stored in the stack*/
    /*////////////////////////*/
    (*cspan_at(&array_in_stack, INT64_C(0))) = INT64_C(1);
    (*cspan_at(&array_in_stack, INT64_C(1))) = INT64_C(2);
    (*cspan_at(&array_in_stack, INT64_C(2))) = INT64_C(3);
    /*////////////////////////*/
    /*array stored in the heap*/
    /*////////////////////////*/
    array_in_heap_ptr = malloc(sizeof(int64_t) * (INT64_C(3)));
    array_in_heap = (array_int64_1d)cspan_md_layout(c_ROWMAJOR, array_in_heap_ptr, INT64_C(3));
    (*cspan_at(&array_in_heap, INT64_C(0))) = INT64_C(1);
    (*cspan_at(&array_in_heap, INT64_C(1))) = INT64_C(2);
    (*cspan_at(&array_in_heap, INT64_C(2))) = INT64_C(3);
    free(array_in_heap.data);
    array_in_heap.data = NULL;
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
    if (allocated(array_in_heap)) deallocate(array_in_heap)

  end subroutine fun1
  !........................................

end module boo
```

## Allow negative index

In Python negative indexes allow a user to index an array starting from the back (e.g. the index -1 is the
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
/*........................................*/
void fun1(int64_t i, int64_t j)
{
    array_int64_1d a = {0};
    array_int64_1d b = {0};
    int64_t* a_ptr;
    int64_t* b_ptr;
    /*////////negative indexing allowed////////*/
    a_ptr = malloc(sizeof(int64_t) * (INT64_C(6)));
    a = (array_int64_1d)cspan_md_layout(c_ROWMAJOR, a_ptr, INT64_C(6));
    (*cspan_at(&a, INT64_C(0))) = INT64_C(1);
    (*cspan_at(&a, INT64_C(1))) = INT64_C(2);
    (*cspan_at(&a, INT64_C(2))) = INT64_C(3);
    (*cspan_at(&a, INT64_C(3))) = INT64_C(4);
    (*cspan_at(&a, INT64_C(4))) = INT64_C(5);
    (*cspan_at(&a, INT64_C(5))) = INT64_C(6);
    printf("%"PRId64"\n", (*cspan_at(&a, i - j < INT64_C(0) ? INT64_C(6) + (i - j) : i - j)));
    /*////////negative indexing disallowed. the generated code can cause a crash/compilation error.////////*/
    b_ptr = malloc(sizeof(int64_t) * (INT64_C(6)));
    b = (array_int64_1d)cspan_md_layout(c_ROWMAJOR, b_ptr, INT64_C(6));
    (*cspan_at(&b, INT64_C(0))) = INT64_C(1);
    (*cspan_at(&b, INT64_C(1))) = INT64_C(2);
    (*cspan_at(&b, INT64_C(2))) = INT64_C(3);
    (*cspan_at(&b, INT64_C(3))) = INT64_C(4);
    (*cspan_at(&b, INT64_C(4))) = INT64_C(5);
    (*cspan_at(&b, INT64_C(5))) = INT64_C(6);
    printf("%"PRId64"\n", (*cspan_at(&b, i - j)));
    free(b.data);
    b.data = NULL;
    free(a.data);
    a.data = NULL;
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
    if (allocated(a)) deallocate(a)
    if (allocated(b)) deallocate(b)

  end subroutine fun1
  !........................................

end module boo
```

## Elemental

In Python it is often the case that a function with scalar arguments and a single scalar output (if any) is also able to accept NumPy arrays with identical rank and shape - in such a case the scalar function is simply applied element-wise to the input arrays. In order to mimic this behaviour in the generated C or Fortran code, Pyccel provides the decorator `elemental`.

Important note: applying the `elemental` decorator to a function will not make a difference to the C translation of the function definition itself since C doesn't have the elementwise feature. However, Pyccel implements that functionality by calling the function in a `for` loop when an array argument is passed. In the following example, we will use the function `square` where `@elemental` will be useful:

Here is the Python code:

```python
from pyccel.decorators import elemental
import numpy as np

@elemental
def square(x : float):
    s = x*x
    return s


def square_in_array():
    a = np.ones(5)
    square(a)
```

The generated C code:

```C
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
    array_double_1d a = {0};
    array_double_1d Dummy_0000 = {0};
    int64_t i;
    double* a_ptr;
    double* Dummy_0000_ptr;
    a_ptr = malloc(sizeof(double) * (INT64_C(5)));
    a = (array_double_1d)cspan_md_layout(c_ROWMAJOR, a_ptr, INT64_C(5));
    c_foreach(Dummy_0001, array_double_1d, a) {
        *(Dummy_0001.ref) = 1.0;
    }
    Dummy_0000_ptr = malloc(sizeof(double) * (INT64_C(5)));
    Dummy_0000 = (array_double_1d)cspan_md_layout(c_ROWMAJOR, Dummy_0000_ptr, INT64_C(5));
    for (i = INT64_C(0); i < INT64_C(5); i += INT64_C(1))
    {
        (*cspan_at(&Dummy_0000, i)) = square((*cspan_at(&a, i)));
    }
    free(Dummy_0000.data);
    Dummy_0000.data = NULL;
    free(a.data);
    a.data = NULL;
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
    if (allocated(a)) deallocate(a)
    if (allocated(Dummy_0001)) deallocate(Dummy_0001)

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
def square(x : float):
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
Functions with the `@inline` decorator will not be exposed to the user in the shared library.
They are only parsed when encountered in a function call. As a result, type annotations are optional for functions with the `@inline` decorator.

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
    do i_0002 = 0_i64, size(a, kind=i64) - 1_i64
      a(i_0002) = pi_0001
    end do
    pi = 3.14_f64
    print *, a, pi
    if (allocated(a)) deallocate(a)

  end subroutine f
  !........................................

end module boo
```

The generated C code:

```c
#include "boo.h"


/*........................................*/
void f(void)
{
    array_double_1d a = {0};
    double pi;
    double* a_ptr;
    double Dummy_0000;
    int64_t Dummy_0001;
    int64_t i;
    a_ptr = malloc(sizeof(double) * (INT64_C(4)));
    a = (array_double_1d)cspan_md_layout(c_ROWMAJOR, a_ptr, INT64_C(4));
    Dummy_0000 = 3.14159;
    for (Dummy_0001 = INT64_C(0); Dummy_0001 < a.shape[INT64_C(0)]; Dummy_0001 += INT64_C(1))
    {
        (*cspan_at(&a, Dummy_0001)) = Dummy_0000;
    }
    pi = 3.14;
    printf("[");
    for (i = INT64_C(0); i < INT64_C(3); i += INT64_C(1))
    {
        printf("%.15lf ", (*cspan_at(&a, i)));
    }
    printf("%.15lf]", (*cspan_at(&a, INT64_C(3))));
    printf("%.15lf\n", pi);
    free(a.data);
    a.data = NULL;
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
    integer(i64) :: b_0001
    integer(i64) :: a_0002
    integer(i64) :: b_0002
    integer(i64) :: a_0003
    integer(i64) :: b_0003
    integer(i64) :: a_0004
    integer(i64) :: b_0004

    a_0001 = 2_i64
    b_0001 = 4_i64
    a = a_0001 + b_0001
    a_0002 = 3_i64
    b_0002 = 5_i64
    b = a_0002 + b_0002
    a_0003 = 6_i64
    b_0003 = 5_i64
    c = a_0003 + b_0003
    a_0004 = 3_i64
    b_0004 = 4_i64
    d = a_0004 + b_0004
    return

  end subroutine f
  !........................................

end module boo
```

The generated C code:

```c
#include "boo.h"


/*........................................*/
int64_t f(int64_t* a, int64_t* b, int64_t* c, int64_t* d)
{
    int64_t Dummy_0000;
    int64_t Dummy_0001;
    int64_t Dummy_0002;
    int64_t Dummy_0003;
    int64_t Dummy_0004;
    int64_t Dummy_0005;
    int64_t Dummy_0006;
    int64_t Dummy_0007;
    Dummy_0000 = INT64_C(2);
    Dummy_0001 = INT64_C(4);
    (*a) = Dummy_0000 + Dummy_0001;
    Dummy_0002 = INT64_C(3);
    Dummy_0003 = INT64_C(5);
    (*b) = Dummy_0002 + Dummy_0003;
    Dummy_0004 = INT64_C(6);
    Dummy_0005 = INT64_C(5);
    (*c) = Dummy_0004 + Dummy_0005;
    Dummy_0006 = INT64_C(3);
    Dummy_0007 = INT64_C(4);
    (*d) = Dummy_0006 + Dummy_0007;
    return 0;
}
/*........................................*/

```

### Import Error when imported from the shared library

Using the previous example, if we import the function `get_val`, we get this error:

```none
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: cannot import name 'get_val' from 'boo' (/home/__init__.py)
```

## Low-level

This decorator is not designed to be used in a `.py` file. Rather it should be used in `.pyi` files. These files are generated by Pyccel to speed up multi-file compilation. They can also be used with `pyccel-wrap` to expose existing Fortran or C code to Python. See [Header Files](./header-files.md) for more information.

_Added in version 2.1_
