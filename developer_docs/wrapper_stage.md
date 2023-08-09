# Wrapper stage

Python is written in C. In order to create a module which can be imported from Python, we therefore have to print a wrapper using the [Python-C API](https://docs.python.org/3/c-api/index.html). This code must call the Pyccel generated translation of the code. If that code was not generated in C then further wrappers are necessary in order to make the code compatible with C.

For example in Fortran generated code there are often array arguments. These do not exist in C, therefore three or more arguments must be passed to the function instead:
1.  The data
2.  The size(s) of the array in each dimension
3.  The stride(s) between elements in each dimension

When defining a slice in Python it is usual to pass a fourth parameter, namely the start of the slice. Internally NumPy simplifies these three integer quantities (`start`, `end`, `step`) into just a stride and a step. This is why the start is not stored or passed as an argument.

The wrapper stage takes the AST describing the generated code as an input and returns an AST which makes that code available to a target language (C or Python).

It should be noted that when code is translated to Python, no wrapper is required.

The entry point for the class `Wrapper` is the function `wrap`.

## `_wrap`

The `_wrap` function internally calls a function named `_wrap_X`, where `X` is the type of the object.
These functions must have the form:
```python
def _print_ClassName(self, stmt):
    ...
    return Y
```
Each of these `_wrap_X` functions should internally call the `_wrap` function on each of the elements relevant to the wrapper to obtain AST objects which describe the same information, but in a way that is accessible from the target language.

## Name Collisions

While creating the wrapper, it is often necessary to create multiple variables to fully describe something which was a single object in the original code. In order to facilitate reading the code these objects should have names similar to that of the original variable. As a result a **lot** of care must be taken to avoid name collisions.

In general the following rules should be respected:
-   Any variables which we wish to be able to retrieve from the scope using the `Scope.find` function must be added to the scope with `Scope.insert_symbol`. We cannot do this with 2 variables with the same name so the names must come from the AST where collisions have already been removed.
-   Any names saved via `Scope.insert_symbol` must be accessed using `Scope.get_expected_name` (as the name may have been changed to avoid other collisions).
-   Any new names must be created using `Scope.get_new_name`

## Fortran To C

The Fortran to C wrapper wraps Fortran code to make it callable from C. This module relies heavily on [`iso_c_binding`](https://stackoverflow.com/tags/fortran-iso-c-binding/info). It is used to ensure that the functions can be called correctly and that types are compatible. We do not use `iso_fortran_binding` as it was only introduced in Fortran 2018 and we do not expect compiler support for this in the near future.

### Scalar module variables

Scalar module variables are already accessible from C so no extra work is done here.

### Array module variables

Arrays are not compatible with C. Instead the wrapper prints a function which returns a pointer to the module variable as well as its size information to make the information available from C.

#### Example

When the following code is translated to Fortran:
```python
import numpy as np
x = np.empty(6)
```

The following Fortran translation is obtained:
```fortran
module tmp


  use, intrinsic :: ISO_C_Binding, only : b4 => C_BOOL , f64 => C_DOUBLE &
        , i64 => C_INT64_T
  implicit none

  integer(i64), bind(c) :: n
  real(f64), allocatable, target :: x(:)
  logical(b4), private :: initialised = .False._b4

  contains

  !........................................
  subroutine tmp__init()

    implicit none

    if (.not. initialised) then
      n = 6_i64
      allocate(x(0:n - 1_i64))
      initialised = .True._b4
    end if

  end subroutine tmp__init
  !........................................

  !........................................
  subroutine tmp__free()

    implicit none

    if (initialised) then
      if (allocated(x)) then
        deallocate(x)
      end if
      initialised = .False._b4
    end if

  end subroutine tmp__free
  !........................................

end module tmp
```

The array `x` is wrapped as follows:
```fortran
  subroutine bind_c_x(bound_x, x_shape_1) bind(c)

    use tmp, only: x

    implicit none

    type(c_ptr), intent(out) :: bound_x
    integer(i64), intent(out) :: x_shape_1

    x_shape_1 = size(x, kind=i64)
    bound_x = c_loc(x)

  end subroutine bind_c_x
```

### Functions

In the simplest case a wrapper around a function takes the same arguments, calls the function and then returns the result. The only difference between this function and the original function is the body (where the original function is called) and the `bind(c)` tag which ensures that the function name is accessible from C and is not mangled.

More complex cases include functions with array arguments or results, optional variables and functions as arguments.

Array arguments and results are handled similarly to array module variables, by adding additional variables for the shape and stride information. Strides are not used for results as pointers cannot be returned from functions.

Optional arguments are passed as C pointers. An if/else block then determines whether the pointer is assigned or not. This can be quite lengthy, however it is unavoidable for compilation with intel, or nvidia. It is also unavoidable for arrays as it is important not to index an array (to access the strides) which is not present.

Finally the most complex cases such as functions as arguments are simply not printed. Instead these cases raise warnings or errors to alert the user that support is missing.

#### Example 1 : function with scalar arguments

The following function:

```python
def f(x : int):
    return x + 3
```

is translated to the following Fortran code:
```fortran
  function f(x) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    integer(i64), value :: x

    Out_0001 = x + 3_i64
    return

  end function f
```

which is then wrapped as follows:
```fortran
  function bind_c_f(x) bind(c) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    integer(i64), value :: x

    Out_0001 = f(x = x)

  end function bind_c_f
```

This function has the following protoype in C:
```c
int64_t bind_c_f(int64_t);
```

#### Example 2 : function with array arguments

The following function:
```python
def f(x : 'int[:]'):
    return x + 3
```

is translated to the following Fortran code:
```fortran
  subroutine f(x, Out_0001)

    implicit none

    integer(i64), allocatable, intent(out) :: Out_0001(:)
    integer(i64), intent(in) :: x(0:)

    allocate(Out_0001(0:size(x, kind=i64) - 1_i64))
    Out_0001 = x + 3_i64
    return

  end subroutine f
```

which is then wrapped as follows:
```fortran
  subroutine bind_c_f(bound_x, bound_x_shape_1, bound_x_stride_1, &
        bound_Out_0001, Out_0001_shape_1) bind(c)

    implicit none

    type(c_ptr), intent(out) :: bound_Out_0001
    integer(i64), intent(out) :: Out_0001_shape_1
    type(c_ptr), value :: bound_x
    integer(i64), value :: bound_x_shape_1
    integer(i64), value :: bound_x_stride_1
    integer(i64), pointer :: x(:)
    integer(i64), allocatable :: Out_0001(:)
    integer(i64), pointer :: Out_0001_ptr(:)

    call C_F_Pointer(bound_x, x, [bound_x_shape_1 * bound_x_stride_1])
    call f(x = x(1_i64::bound_x_stride_1), Out_0001 = Out_0001)
    Out_0001_shape_1 = size(Out_0001, kind=i64)
    allocate(Out_0001_ptr(0:Out_0001_shape_1 - 1_i64))
    Out_0001_ptr = Out_0001
    bound_Out_0001 = c_loc(Out_0001_ptr)

  end subroutine bind_c_f
```

This function has the following protoype in C:
```c
int bind_c_f(void*, int64_t, int64_t, void*, int64_t*);
```

#### Example 3 : function with optional scalar arguments

The following function:
```python
def f(x : int = None):
    if x is None:
        return 2
    else:
        return x + 3
```

is translated to the following Fortran code:
```fortran
  function f(x) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    integer(i64), optional, value :: x

    if (.not. present(x)) then
      Out_0001 = 2_i64
      return
    else
      Out_0001 = x + 3_i64
      return
    end if

  end function f
```

which is then wrapped as follows:
```fortran
  function bind_c_f(bound_x) bind(c) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    type(c_ptr), value :: bound_x
    integer(i64), pointer :: x

    if (c_associated(bound_x)) then
      call C_F_Pointer(bound_x, x)
      Out_0001 = f(x = x)
    else
      Out_0001 = f()
    end if

  end function bind_c_f
```

This function has the following protoype in C:
```c
int64_t bind_c_f(void*);
```

#### Example 4 : function with optional array arguments

The following function:
```python
def f(x : 'float[:]' = None):
    import numpy as np
    if x is None:
        return np.ones(3)
    else:
        return x + 3
```

is translated to the following Fortran code:
```fortran
  subroutine f(x, Out_0001)

    implicit none

    real(f64), allocatable, intent(out) :: Out_0001(:)
    real(f64), optional, intent(in) :: x(0:)

    if (.not. present(x)) then
      allocate(Out_0001(0:2_i64))
      Out_0001 = 1.0_f64
      return
    else
      allocate(Out_0001(0:size(x, kind=i64) - 1_i64))
      Out_0001 = x + 3_i64
      return
    end if
    if (allocated(Out_0001)) then
      deallocate(Out_0001)
    end if

  end subroutine f
```

which is then wrapped as follows:
```fortran
  subroutine bind_c_f(bound_x, bound_x_shape_1, bound_x_stride_1, &
        bound_Out_0001, Out_0001_shape_1) bind(c)

    implicit none

    type(c_ptr), intent(out) :: bound_Out_0001
    integer(i64), intent(out) :: Out_0001_shape_1
    type(c_ptr), value :: bound_x
    integer(i64), value :: bound_x_shape_1
    integer(i64), value :: bound_x_stride_1
    real(f64), pointer :: x(:)
    real(f64), allocatable :: Out_0001(:)
    real(f64), pointer :: Out_0001_ptr(:)

    if (c_associated(bound_x)) then
      call C_F_Pointer(bound_x, x, [bound_x_shape_1 * bound_x_stride_1])
      call f(x = x(1_i64::bound_x_stride_1), Out_0001 = Out_0001)
    else
      call f(Out_0001 = Out_0001)
    end if
    Out_0001_shape_1 = size(Out_0001, kind=i64)
    allocate(Out_0001_ptr(0:Out_0001_shape_1 - 1_i64))
    Out_0001_ptr = Out_0001
    bound_Out_0001 = c_loc(Out_0001_ptr)

  end subroutine bind_c_f
```

This function has the following protoype in C:
```c
int bind_c_f(void*, int64_t, int64_t, void*, int64_t*);
```


### Class module variables

**Class module variables are not yet wrapped. This describes the future implementation**

Unlike Fortran, C does not have classes in the language. The wrapper therefore cannot pass the class to C via a description. Instead the wrapper should print a function which returns a pointer to the module variable.

Additionally the class method functions will be wrapped as described for functions. The attributes of the class will be exposed via wrapper functions.

## C To Python

The C to Python wrapper wraps C code to make it callable from Python. This module relies heavily on the [Python-C API](https://docs.python.org/3/c-api/index.html).

### Functions
