# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import inline, private
from typing import TypeVar, Final
import numpy as np

@private
def hidden():
    print("hidden")

def inline_1_out():
    @inline
    def cube(s: int):
        return s * s * s

    a = cube(3)
    b = cube(8 + 3)
    c = cube((b - a) // 20)
    d = cube(a)
    return a, b, c, d

def inline_0_out(x: "int[:]"):
    @inline
    def set_3(s: "int[:]", i: int):
        s[i] = 3

    set_3(x, 0)
    set_3(x, 1)

def inline_local():
    @inline
    def power_4(s: int):
        x = s * s
        return x * x

    a = power_4(3)
    b = power_4(8 + 3)
    c = power_4((b - a) // 2000)
    g = 4
    d = power_4(g)
    return a, b, c, d

def inline_local_name_clash():
    @inline
    def power_4(s: int):
        x = s * s
        return x * x

    a = power_4(3)
    b = power_4(8 + 3)
    c = power_4((b - a) // 2000)
    x = 2
    d = power_4(x)
    return a, b, c, d, x

def inline_optional():
    @inline
    def get_val(x: int = None, y: int = None):
        if x is None:
            a = 3
        else:
            a = x
        if y is not None:
            b = 4
        else:
            b = 5
        return a + b

    a = get_val(2, 7)
    b = get_val()
    c = get_val(6)
    d = get_val(y=0)
    return a, b, c, d

def inline_array():
    from numpy import empty

    @inline
    def fill_array(a: "float[:]"):
        for i in range(a.shape[0]):
            a[i] = 3.14

    arr = empty(4)
    fill_array(arr)
    return arr[0], arr[-1]

def nested_inline_call():
    @inline
    def get_val(x: int = None, y: int = None):
        if x is None:
            a = 3
        else:
            a = x
        if y is not None:
            b = 4
        else:
            b = 5
        return a + b

    a = get_val(get_val(2) + 3, 7)
    return a

def inline_return():
    @inline
    def tmp():
        a = 1
        return a

    b = tmp()
    c = tmp()
    d = tmp() + 3
    e = tmp() * 4
    return b, c, d, e

def inline_multiple_results():
    @inline
    def get_2_vals(a: int):
        return a * 2, a - 5

    get_2_vals(5)
    x = get_2_vals(7)
    y0, y1 = get_2_vals(3)
    return x, y0, y1

def inline_literal_return():
    @inline
    def tmp():
        return 2

    b = tmp()
    c = tmp()
    d = tmp() + 3
    e = tmp() * 4
    return b, c, d, e

def inline_array_return():
    @inline
    def tmp():
        return np.ones(2, dtype=int)

    b = tmp()
    c = np.sum(tmp())
    return b, c

def inline_multiple_return():
    @inline
    def tmp():
        a = 1
        b = 4
        return a, b

    b, c = tmp()
    d, e = tmp()
    return b, c, d, e

def inline_homogeneous_tuple_result():
    @inline
    def get_2_vals(a: int):
        b = (a * 2, a - 5)
        return b

    get_2_vals(5)
    x = get_2_vals(7)
    y0, y1 = get_2_vals(3)
    return x, y0, y1

def inline_inhomogeneous_tuple_result():
    @inline
    def get_2_vals(a: int):
        b: tuple[int, int] = (a * 2, a - 5)
        return b

    get_2_vals(5)
    x = get_2_vals(7)
    y0, y1 = get_2_vals(3)
    return x, y0, y1

def inhomogeneous_tuple_in_inline():
    @inline
    def tmp():
        a = (1, False)
        return a[0] + 2

    b = tmp()
    return b

def multi_level():
    @inline
    def tmp():
        a = ((1, False), 3.0)
        return a[0][0] + 2

    b = tmp()
    return b

T_my_sum = TypeVar("T_my_sum", "float[:]", "complex[:]")

def my_sum(v: Final[T_my_sum]):
    return v.sum()

