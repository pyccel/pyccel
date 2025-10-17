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
def _wrap_ClassName(self, stmt):
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
      if (allocated(x)) deallocate(x)
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

In order to create all the nodes necessary to describe the unpacking of the arguments we use functions named `_extract_X_FunctionDefArgument` where `X` is the type of the object being extracted from the `FunctionDefArgument`. This allows such functions to call each other recursively. This is notably useful for container types (tuples, lists, etc) whose elements may themselves be container types. The types of scalars are checked in the same way regardless of whether they are arguments or elements of a container so this also reduces code duplication.

Array arguments and results are handled similarly to array module variables, by adding additional variables for the shape and stride information. Strides are not used for results as pointers cannot be returned from functions.

Optional arguments are passed as C pointers. An if/else block then determines whether the pointer is assigned or not. This can be quite lengthy, however it is unavoidable for compilation with Intel, or NVIDIA. It is also unavoidable for arrays as it is important not to index an array (to access the strides) which is not present.

Finally the most complex cases such as functions as arguments are simply not printed. Instead these cases raise warnings or errors to alert the user that support is missing.

In order to create all the nodes necessary to describe the unpacking of the results we use functions named `_extract_X_FunctionDefResult` where `X` is the type of the object being extracted from the `FunctionDefResult`. This helps with the readability of the code.

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

This function has the following prototype in C:

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

This function has the following prototype in C:

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

This function has the following prototype in C:

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
    if (allocated(Out_0001)) deallocate(Out_0001)

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

This function has the following prototype in C:

```c
int bind_c_f(void*, int64_t, int64_t, void*, int64_t*);
```

### Class module variables

Unlike Fortran, C does not have classes in the language. The wrapper therefore cannot pass the class to C via a description. Instead the wrapper prints a function which returns a pointer to the module variable.

Additionally the class method functions are be wrapped as described for functions. The attributes of the class are be exposed via getter and setter wrapper functions.

## C To Python

The C to Python wrapper wraps C code to make it callable from Python. This module relies heavily on the [Python-C API](https://docs.python.org/3/c-api/index.html).

### Functions

A function that can be called from Python must have the following prototype:

```c
PyObject* func_name(PyObject* self, PyObject* args, PyObject* kwargs);
```

The arguments and keyword arguments are unpacked into individual `PyObject` pointers.
Each of these objects is checked to verify the type. If the type does not match the expected type then an error is raised as described in the [C-API documentation](https://docs.python.org/3/c-api/intro.html#exceptions).
If the type does match then the value is unpacked into a C object. This is done using custom functions defined in `pyccel/stdlib/cwrapper/` (see these files for more details) or using functions provided by `Python.h`.

In order to create all the nodes necessary to describe the unpacking of the arguments we use functions named `_extract_X_FunctionDefArgument` where `X` is the type of the object being extracted from the `FunctionDefArgument`. This allows such functions to call each other recursively. This is notably useful for container types (tuples, lists, etc) whose elements may themselves be container types. The types of scalars are checked in the same way regardless of whether they are arguments or elements of a container so this also reduces code duplication.

Similarly in order to create all the nodes necessary to describe the packing of the results we use functions named `_extract_X_FunctionDefResult` where `X` is the type of the object being packed from the `FunctionDefResult`. This allows such functions to call each other recursively. This is notably useful for container types (tuples, lists, etc) whose elements may themselves be container types.

Once C objects have been retrieved the function is called normally.

The wrapper is attached to the module via a `PyMethodDef` (see C-API [docs](https://docs.python.org/3/c-api/structures.html#c.PyMethodDef)).

#### Example 1

The following Python code:

```python
def f(x : 'float[:]', y : float = 3):
    return x + y
```

leads to C code with the following prototype:

```c
array_double_1d tmp__f(array_double_1d x, double y);
```

which is then wrapped as follows:

```c
static PyObject* tmp__f_wrapper(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* x_obj;
    PyObject* y_obj;
    void* x_data = NULL;
    int64_t x_shape[1];
    int64_t x_strides[1];
    array_double_1d x_0001 = {0};
    array_double_1d x = {0};
    double y;
    array_double_1d Out_0001 = {0};
    PyObject* Out_0001_obj;
    y_obj = Py_None;
    static char *kwlist[] = {
        (char*)"x",
        (char*)"y",
        NULL
    }; 
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &x_obj, &y_obj))
    {   
        return NULL;
    }
    if (!pyarray_check("x", x_obj, NPY_DOUBLE, INT64_C(1), NO_ORDER_CHECK))
    {
        return NULL;
    }
    x_data = PyArray_DATA((PyArrayObject*)(x_obj));
    get_strides_and_shape_from_numpy_array(x_obj, x_shape, x_strides);
    x_0001 = (array_double_1d)cspan_md_layout(c_ROWMAJOR, x_data, x_shape[INT64_C(0)] * x_strides[INT64_C(0)]);
    x = cspan_slice(array_double_1d, &x_0001, {INT64_C(0), c_END, x_strides[INT64_C(0)]});
    y = 3.0;
    if (y_obj != Py_None)
    {
        if (PyIs_NativeFloat(y_obj))
        {
            y = PyDouble_to_Double(y_obj);
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument y");
            return NULL;
        }
    }
    Out_0001 = tmp__f(x, y);
    Out_0001_obj = to_pyarray(INT64_C(1), NPY_DOUBLE, Out_0001.data, Out_0001.shape, 1, 0);
    return Out_0001_obj;
}
```

The function is linked to the module via a `PyMethodDef` as follows:

```c
static PyMethodDef tmp_methods[] = {
    {
        "f", // Function name
        (PyCFunction)tmp__f_wrapper, // Function implementation
        METH_VARARGS | METH_KEYWORDS, // Indicates that the function accepts args and kwargs
        "" // function docstring
    },
    { NULL, NULL, 0, NULL}
};
```

If the code was translated to Fortran the prototype is:

```c
int bind_c_f(void*, int64_t, int64_t, double, void*, int64_t*);
```

which is then wrapped as follows:

```c
static PyObject* bind_c_f_wrapper(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* x_obj;
    PyObject* y_obj;
    void* x_data = NULL;
    int64_t x_shape[1];
    int64_t x_strides[1];
    double y;
    void* Out_0001_data = NULL;
    int64_t Out_0001_shape[1];
    PyObject* Out_0001_obj;
    y_obj = Py_None;
    static char *kwlist[] = {
        (char*)"x",
        (char*)"y",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &x_obj, &y_obj))
    {
        return NULL;
    }
    if (!pyarray_check("x", x_obj, NPY_DOUBLE, INT64_C(1), NO_ORDER_CHECK))
    {
        return NULL;
    }
    x_data = PyArray_DATA((PyArrayObject*)(x_obj));
    get_strides_and_shape_from_numpy_array(x_obj, x_shape, x_strides);
    y = 3.0;
    if (y_obj != Py_None)
    {
        if (PyIs_NativeFloat(y_obj))
        {
            y = PyDouble_to_Double(y_obj);
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument y");
            return NULL;
        }
    }
    bind_c_f(x_data, x_shape[INT64_C(0)], x_strides[INT64_C(0)], y, &Out_0001_data, &Out_0001_shape[INT64_C(0)]);
    Out_0001_obj = to_pyarray(INT64_C(1), NPY_DOUBLE, Out_0001_data, Out_0001_shape, 1, 1);
    return Out_0001_obj;
}
```

#### Example 2 : function with tuple arguments

The following Python code:

```python
def get_first_element_of_tuple(a : 'tuple[int,...]'):
    return a[0]
```

leads to C code with the following prototype (as homogeneous tuples are treated like arrays):

```c
int64_t tmp__get_first_element_of_tuple(array_int64_1d a);
```

which is then wrapped as follows:

```c
static PyObject* tmp__get_first_element_of_tuple_wrapper(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* a_obj;
    int64_t a_size_0001;
    array_int64_1d a = {0};
    int Dummy_0000;
    PyObject* Dummy_0001;
    bool is_homog_tuple;
    int64_t size;
    int Dummy_0002;
    PyObject* Dummy_0003;
    int64_t Out_0001;
    int64_t* a_ptr;
    PyObject* Out_0001_obj;
    static char *kwlist[] = {
        (char*)"a",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &a_obj))
    {
        return NULL;
    }
    if (PyTuple_Check(a_obj))
    {
        size = PyTuple_Size(a_obj);
        is_homog_tuple = 1;
        for (Dummy_0002 = INT64_C(0); Dummy_0002 < size; Dummy_0002 += INT64_C(1))
        {
            Dummy_0003 = PyTuple_GetItem(a_obj, Dummy_0002);
            is_homog_tuple = is_homog_tuple && PyIs_NativeInt(Dummy_0003);
        }
    }
    else
    {
        is_homog_tuple = 0;
    }
    if (!is_homog_tuple)
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type tuple[int, ...] for argument a");
        return NULL;
    }
    a_size_0001 = PyTuple_Size(a_obj);
    a_ptr = malloc(sizeof(int64_t) * (a_size_0001));
    a = (array_int64_1d)cspan_md_layout(c_ROWMAJOR, a_ptr, a_size_0001);
    for (Dummy_0000 = INT64_C(0); Dummy_0000 < a_size_0001; Dummy_0000 += INT64_C(1))
    {
        Dummy_0001 = PyTuple_GetItem(a_obj, Dummy_0000);
        (*cspan_at(&a, Dummy_0000)) = PyInt64_to_Int64(Dummy_0001);
    }
    Out_0001 = tmp__get_first_element_of_tuple(a);
    Out_0001_obj = Int64_to_PyLong(&Out_0001);
    return Out_0001_obj;
}
```

### Interfaces

Interfaces are functions which accept more than one type.
These functions are handled via multiple functions in the wrapper:

1.  A function which can be called from Python with the prototype:

    ```c
    PyObject* func_name(PyObject* self, PyObject* args, PyObject* kwargs);
    ```

2.  A function which determines which combination of types were used in the call

3.  A function for each combination of types which calls the translated function

#### Example

The following Python code:

```python
def f(x : int | float):
    return x + 2
```

leads to C code with the following prototypes:

```c
double tmp__f_00(double x);
int64_t tmp__f_01(int64_t x);
```

which is then wrapped as follows:

```c
/*........................................*/
PyObject* tmp__f_00_wrapper(PyObject* x_obj)
{
    double x;
    double Out_0001;
    PyObject* Out_0001_obj;
    x = PyDouble_to_Double(x_obj);
    Out_0001 = tmp__f_00(x);
    Out_0001_obj = Double_to_PyDouble(&Out_0001);
    return Out_0001_obj;
}
/*........................................*/

/*........................................*/
PyObject* tmp__f_01_wrapper(PyObject* x_obj)
{
    int64_t x;
    int64_t Out_0001;
    PyObject* Out_0001_obj;
    x = PyInt64_to_Int64(x_obj);
    Out_0001 = tmp__f_01(x);
    Out_0001_obj = Int64_to_PyLong(&Out_0001);
    return Out_0001_obj;
}
/*........................................*/
/*_________________________________CommentBlock_________________________________*/
/*Assess the types. Raise an error for unexpected types and calculate an integer */
/*which indicates which function should be called.                               */
/*_______________________________________________________________________________*/
int64_t tmp__f_type_check(PyObject* x_obj)
{
    int64_t type_indicator;
    type_indicator = INT64_C(0);
    if (PyIs_NativeFloat(x_obj))
    {
        type_indicator += INT64_C(0);
    }
    else if (PyIs_NativeInt(x_obj))
    {
        type_indicator += INT64_C(1);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Unexpected type for argument x");
        return -INT64_C(1);
    }
    return type_indicator;
}
/*........................................*/
/*........................................*/
PyObject* tmp__f_wrapper(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* x_obj;
    int64_t type_indicator;
    static char *kwlist[] = {
        "x",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &x_obj))
    {
        return NULL;
    }
    type_indicator = f_type_check(x_obj);
    if (type_indicator == INT64_C(0))
    {
        return tmp__f_00_wrapper(x_obj);
    }
    else if (type_indicator == INT64_C(1))
    {
        return tmp__f_01_wrapper(x_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Unexpected type combination");
        return NULL;
    }
}
/*........................................*/
```

The function is linked to the module via a `PyMethodDef` as follows:

```c
static PyMethodDef tmp_methods[] = {
    {
        "f", // Function name
        (PyCFunction)tmp__f_wrapper, // Function implementation
        METH_VARARGS | METH_KEYWORDS, // Indicates that the function accepts args and kwargs
        "" // function docstring
    },
    { NULL, NULL, 0, NULL}
};
```

### Variables

In order to import variables from a module, they must be attached to the module using the function [`PyModule_AddObject`](https://docs.python.org/3/c-api/module.html#c.PyModule_AddObject).
In order to do this the value of the variable must be placed in a `PyObject`.
This is done in a function which is executed when the module is imported for the first time.

#### Example

The following Python code:

```python
x = 3
```

leads to the following module initialisation function:

```c
int32_t tmp_exec_func(PyObject* mod)
{
    PyObject* Dummy_0001;
    // Call the C function to initialise the module
    tmp__init();
    // Pack the module variable into a PyObject
    Dummy_0001 = Int64_to_PyLong(&x);
    // Add the PyObject to the Python module.
    if (PyModule_AddObject(mod, "x", Dummy_0001) < INT64_C(0))
    {
        // Raise an error in case of problems
        return -INT64_C(1);
    }
    return INT64_C(0);
}
```

### Classes

The wrapping of classes is described in detail at <https://docs.python.org/3/extending/newtypes_tutorial.html>.

The class methods are wrapped similarly to normal functions. The main difference is that the first argument (`self`) is used and represents the class instance instead of the module instance.

In order to wrap class attributes, a getter and a setter function are created. These are then linked to the class as described in the tutorial.

Classes have an additional difficulty which is not present for module functions and variables. The difficulty is that they define a new type which may be imported and used in other modules. This difficulty is managed using capsules. Capsules are objects in the C-Python API which are designed to expose the API to other compiled Python libraries. The use of these capsules is described at <https://docs.python.org/3/extending/extending.html#using-capsules>.

### Returning pointers

Pyccel allows the user to return a pointer to an object which is an argument of a function (including an attribute of a bound argument). Pointers are created when pointing at non-trivial objects (e.g. classes, NumPy arrays, lists). When doing this it is very important to ensure that the reference counter on the target is incremented and will be decremented when the pointer goes out of scope.

Python and Pyccel handle this slightly differently. Python directly returns the object passed as argument so we have `pointer is target`. The C to Python backend could theoretically do this, but they would have to compare the pointer for the results with the pointer for each of the target arguments to find the correct argument. This is unnecessarily complex and class attributes would have to be handled as a special case. Instead what is done is that a new object is created which operates on the same data.

For a NumPy array pointer, a `numpy.ndarray` is created which operates on the same data. NumPy uses the base property to track the object whose reference counter should be decremented when this object is destroyed. If there is only 1 possible target then the base is set to this target (i.e. to the class for a class property, or to the argument for a pointer to an argument) so the data is not deleted and the decremented object can take care of destroying it when necessary. If there are multiple possible NumPy targets then a warning is raised advising the user to avoid using this function. In reality it can be used as long as one takes care with scoping to ensure dangling pointers cannot occur.

For a class pointer, a new class instance is created. This class has a `referenced_objects` attribute which contains a Python list. All possible targets are added to this list. They are decremented by removing them from the Python list when the pointer class goes out of scope and is garbage collected.
