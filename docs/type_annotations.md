# Type Annotations

Type annotations are an integral part of Pyccel. When parsing a function, Pyccel needs to know the type of the input arguments in order to perform type inference, and ultimately compute the type of the output result. Elsewhere it is also possible, but not obligatory, to provide type information.

Where possible we try to support Python-style annotations which are compatible with mypy.

You can leave annotations on function arguments, function results or variables:
```python
def fun(arg1: type1, arg2: type2, ..., argN: typeN) -> return_type:
    a : var_type
```

The only compulsory annotations are argument annotations, and result annotations in the case of recursive functions.

For scalar variables Pyccel supports the following data types:

-   built-in data types: `bool`, `int`, `float`, `complex`
-   NumPy integer types: `int8`, `int16`, `int32`, `int64`
-   NumPy real types: `float32`, `float64`, `double`
-   NumPy complex types: `complex64`, `complex128`

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

In order to keep the number of functions to what is necessary and thus reduce compilation times, Pyccel also provides a decorator `@template` which allows the type to be specified for all arguments at once.
E.g.
```python
from pyccel.decorators import template
@template(name='T', types=['int','float'])
def f(a : 'T', b : 'T'):
    pass
```

For more details, see the documentation for [templates](./templates.md).
