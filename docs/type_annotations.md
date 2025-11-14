# Type Annotations

Type annotations are an integral part of Pyccel. When parsing a function, Pyccel needs to know the type of the input arguments in order to perform type inference, and ultimately compute the type of the output result. Elsewhere it is also possible, but not obligatory, to provide type information.

Where possible we try to support Python-style annotations which are compatible with mypy.

You can leave annotations on function arguments, function results or variables:

```python
def fun(arg1: type1, arg2: type2, ..., argN: typeN) -> return_type:
    a : var_type
```

The only compulsory annotations are argument annotations, and result annotations in the case of recursive functions.

## Scalar Values

For scalar variables Pyccel supports the following data types:

-   built-in data types: `bool`, `int`, `float`, `complex`
-   NumPy integer types: `int8`, `int16`, `int32`, `int64`
-   NumPy floating point types: `float32`, `float64`, `double`
-   NumPy complex types: `complex64`, `complex128`

## NumPy Arrays

Unfortunately, NumPy does not yet provide mypy compatible type annotations which provide sufficient information for Pyccel. As a result we also support the use of string annotations:

```python
def fun(arg1: 'type1', arg2: 'type2', ..., argN: 'typeN') -> 'return_type':
    a : 'var_type'
```

To declare NumPy arrays with string syntax we write:

```python
def fun(arg1: 'type1[:]', arg2: 'type2[:,:]', ..., argN: 'typeN[dimensions]'):
```

The number of dimensions of an array is equal to the number of comma-separated colons in the square brackets.
So `arr[:]` means that the array `arr` has 1 dimension, `arr[:,:]` means that it has 2 dimensions and so on.
In general string type hints must be used to provide Pyccel with information about the number of dimensions.

:warning: When running Pyccel in interactive mode with `epyccel` anything imported before the function being translated is not copied into the file which will be translated. Using non-string types here is therefore likely to generate errors as the necessary imports for these objects will be missing.

## Tuples

Currently Pyccel supports tuples used locally in functions, as returned objects and in certain cases as arguments, however module variables are not yet handled. The implementation of the type annotations (including adding the missing support) is in progress.

Tuples can be homogeneous or inhomogeneous. A homogeneous tuple is a tuple whose elements all have the same type and shape. Pyccel translates homogeneous tuples in a similar way to NumPy arrays. When creating multiple dimensional tuples it is therefore important to ensure that all objects have compatible sizes otherwise they will be handled as inhomogeneous tuples. An inhomogeneous tuple describes all other types, but comes with extra restrictions. An inhomogeneous tuple is translated to multiple objects in the target language so it can only be used if the element can be identified during the translation. This means that expressions such as `a[i]` are not possible for inhomogeneous tuples while `a[0]` is valid.

Homogeneous tuple type annotations are supported for local variables and function arguments (if the tuples contain scalar objects).

To declare a homogeneous tuple the syntax is as follows:

```python
a : tuple[int,...] = (1,2,3,4)
```

Inhomogeneous tuple type annotations are supported for local variables.

To declare an inhomogeneous tuple the syntax is as follows:

```python
a : tuple[int,bool] = (1,False)
```

It is of course possible to create an inhomogeneous tuple in place of a homogeneous tuple to benefit from code optimisations that can arise from using multiple scalars in place of an array object. This will however imply the same restrictions as any other inhomogeneous tuple. E.g:

```python
a : tuple[int, int] = (1,2)
```

## Lists

Lists are in the process of being added to Pyccel. Homogeneous lists can be declared in Pyccel using the following syntax:

```python
a : list[int] = [1, 2]
b : list[bool] = [False, True]
c : list[float] = []
```

So far lists can be declared as local variables or as arguments or results of functions.

## Sets

Sets are in the process of being added to Pyccel. Homogeneous sets can be declared in Pyccel using the following syntax:

```python
a : set[int] = {1, 2}
b : set[bool] = {False, True}
c : set[float] = {}
```

Sets can be declared as local variables, arguments or results of functions, but not yet as class variables. An argument can be marked as constant using the `Final` qualifier:

```python
from typing import Final
def g(b : Final[set[bool]]):
    pass
```

## Dictionaries

Dictionaries are in the process of being added to Pyccel.
Homogeneous dictionaries can be declared in Pyccel using the following syntax:

```python
a : dict[int,float] = {1: 1.0, 2: 2.0}
b : dict[int,bool] = {1: False, 4: True}
c : dict[int,complex] = {}
```

Strings are not yet supported as keys in Fortran.
Dictionaries can be declared as local variables, or results of functions, but not yet as arguments or class variables.

## Strings

Pyccel contains very minimal support for strings. For example strings can be used as keys of dictionaries (only in C currently) or to represent flags in the user code with if statements checking their value. More complex string handling is not currently supported. See the documentation on [builtin functions](./builtin-functions.md) for an overview of the available string methods.

Strings can be declared in Pyccel using the following syntax:

```python
a : str = 'hello'
```

## Handling multiple types

The basic type annotations indicate only one type however it is common to need a function to be able to handle multiple types, e.g. integers and floats. In this case it is possible to provide a union type.
E.g.

```python
def f(a : int | float):
    pass

def g(a : 'int | float'):
    pass
```

Union types are useful for specifying multiple types however the function must handle all possible permutations of the argument combinations. For example the following type annotations:

```python
def f(a : int | float, b : int | float):
```

will lead to four functions being created, equivalent to:

```python
def f(a : int, b : int):
def f(a : int, b : float):
def f(a : float, b : int):
def f(a : float, b : float):
```

In order to keep the number of functions to what is necessary and thus reduce compilation times, users should use a `TypeVar` object from Python's `typing` module. This allows the type to be specified for all arguments at once.
E.g.

```python
from typing import TypeVar

T = TypeVar('T', int, float)

def f(a : T, b : T):
    pass
```

## Type Aliases

Python also provides type alias objects as described in the Python docs (<https://docs.python.org/3/library/typing.html#type-aliases>). This allows the user to more easily change between different types. Type parameter lists are not supported as they do not fully define the final type. Both the new Python 3.12 syntax and the old syntax are supported. Type aliases cannot be redefined. The type name will not appear in the underlying code.

E.g.

```python
from typing import TypeAlias

MyType : TypeAlias = float

def set_i(x : 'MyType[:]', i : 'int', val : MyType):
    x[i] = val
```

or:

```python
type MyType = float

def set_i(x : 'MyType[:]', i : 'int', val : MyType):
    x[i] = val
```

## Annotated types

Python provides the class `typing.Annotated` to allow variables to be annotated with context-specific metadata (<https://docs.python.org/3/library/typing.html#typing.Annotated>). Pyccel supports this class if it is encountered in user code. It also leverages this mechanism to describe types in more detail in stub files. In particular pointers and stack arrays can be declared with this notation.
E.g.

```python
from typing import Annotated
arr : 'Annotated[int[:], "pointer"]' # Equivalent to Fortran notation : integer(i64), pointer :: arr(:)
arr2 : 'Annotated[int[:], "stack"]' = np.ones(8) # Equivalent to Fortran notation : integer(i64) :: arr2(0:7)
```

This syntax can be combined with other type related concepts. E.g.

```python
from typing import TypeVar, Annotated
T = TypeVar('T', 'int[:]', 'float[:]')
arr : Annotated[T, 'pointer']
arr2 : Annotated[T, 'stack'] = np.ones(8)
```

:Note: It is currently not advised to use pointer annotations in files passed to pyccel-wrap. The annotation does not currently provide a way to indicate what the pointer is pointing at. This will lead to incorrect deallocation.
