# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar


T = TypeVar(
    "T",
    "bool",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "float",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
NumType = TypeVar(
    "NumType",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "float",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
FArrays = TypeVar("FArrays", "float[:,:,:](order=F)", "float[:,:](order=F)")
CArrays = TypeVar("CArrays", "float[:,:,:](order=C)", "float[:,:](order=C)")

def return_array(a: "T", b: "T"):
    from numpy import array

    x = array([a, b], dtype=type(a))
    return x

def multi_returns(a: "T", b: "T"):
    from numpy import array

    x = array([a, b], dtype=type(a))
    y = array([a, b], dtype=type(a))
    return x, y

def return_array_array_op(a: "NumType", b: "NumType"):
    from numpy import array

    x = array([a, b], dtype=type(a))
    y = array([a, b], dtype=type(a))
    return x + y

def return_multi_array_array_op(a: "NumType", b: "NumType"):
    from numpy import array

    x = array([a, b], dtype=type(a))
    y = array([a, b], dtype=type(a))
    return x + y, x - y

def return_array_scalar_op(a: NumType):
    from numpy import ones

    x = ones(5, dtype=type(a))
    return x * a

def return_multi_array_scalar_op(a: NumType):
    from numpy import ones

    x = ones(5, dtype=type(a))
    y = ones(5, dtype=type(a))
    return x * a, y * a

def return_array_arg_array_op(a: "NumType[:]"):
    from numpy import ones

    x = ones(7)
    return x * a

def return_arrays_in_expression():
    def single_return():
        from numpy import array

        return array([1, 2, 3, 4])

    b = single_return() + 1

    return b

def return_arrays_in_expression2(n: int):
    def single_return(n: int):
        from numpy import ones

        return ones(n)

    b = single_return(n) + 1

    return b

def return_c_array(b: NumType):
    from numpy import array

    a = array([[1, 2, 3], [4, 5, 6]], dtype=type(b))
    return a

def return_f_array(b: NumType):
    from numpy import array

    a = array([[1, 2, 3], [4, 5, 6]], dtype=type(b), order="F")
    return a

def copy_f_to_f(b: FArrays):
    from numpy import array

    a = array(b, order="F")
    return a

def copy_f_to_c(b: FArrays):
    from numpy import array

    a = array(b, order="C")
    return a

def copy_c_to_c(b: CArrays):
    from numpy import array

    a = array(b, order="C")
    return a

def copy_c_to_f(b: CArrays):
    from numpy import array

    a = array(b, order="F")
    return a

def copy_c_to_default(b: CArrays):
    from numpy import array

    a = array(b)
    return a

def copy_f_to_default(b: FArrays):
    from numpy import array

    a = array(b)
    return a

def annotated_return(b: "float[:,:]", c: "float[:,:]") -> "float[:,:]":
    return b + c

def unknown_size(b: bool):
    from numpy import ones, zeros

    if b:
        a = ones(3)
    else:
        a = zeros(4)
    return a
