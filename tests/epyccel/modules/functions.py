# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import Final, TypeVar

T = TypeVar("T", int, float)
T_f = TypeVar("T_f", "int[:]", list[int], set[int])


def p_lose():
    lose = -10
    return lose


def func_no_return_1(x: int):
    x *= 2


def func_no_return_2():
    x = 2
    x *= 2


def func_no_args_f1():
    from numpy import pi

    value = (2 * pi) ** (3 / 2)
    return value


def func_return_constant():
    from numpy import pi

    return pi


def decorator_f1(x: "int"):
    y = x - 1
    return y


def decorator_f2(x: "int [:]"):
    y = x[0] - 1
    return y


def decorator_f3(x: "int [:]"):
    from numpy import empty_like

    y = empty_like(x)
    y[:] = x - 1
    return y


def decorator_f4(x: "float [:,:]"):
    from numpy import empty_like

    y = empty_like(x)
    y[:] = x - 1.0
    return y


def decorator_f5(m1: "int", x: "float [:]"):
    x[:] = 0.0
    for i in range(0, m1):
        x[i] = i * 1.0


def f6_1(m1: "int", m2: "int", x: "float [:,:]"):
    x[:, :] = 0.0
    for i in range(0, m1):
        for j in range(0, m2):
            x[i, j] = (2 * i + j) * 1.0


def decorator_f7(m1: "int", m2: "int", x: "float [:,:](order=F)"):
    x[:, :] = 0.0
    for i in range(0, m1):
        for j in range(0, m2):
            x[i, j] = (2 * i + j) * 1.0


def decorator_f8(x: "int", b: "bool"):
    a = x if b else 2
    return a


def arguments_f9(x: "int64[:]"):
    x += 1


def arguments_f10(x: "int64[:]"):
    x[:] += 1


def ackermann(m: "int", n: "int") -> int:
    if m == 0:
        return n + 1
    elif n == 0:
        return ackermann(m - 1, 1)
    else:
        return ackermann(m - 1, ackermann(m, n - 1))


def non_negative(i: "int"):
    if i < 0:
        return False
    else:
        return True


def get_min(a: "int", b: "int"):
    if a < b:
        return a
    else:
        return b


def multiple_returns_f14(x: "int", y: "int"):
    return x, y, y, y, x


def decorator_f15(a: "bool", b: "int8", c: "int16", d: "int32", e: "int64"):
    from numpy import int64

    if a:
        return int64(b + c)
    else:
        return d + e


def decorator_f16(a: "int16"):
    b = a
    return b


def decorator_f17(a: "int8"):
    b = a
    return b


def decorator_f18(a: "int32"):
    b = a
    return b


def decorator_f19(a: "int64"):
    b = a
    return b


def decorator_f20(a: "complex"):
    b = a
    return b


def decorator_f21(a: "complex64"):
    b = a
    return b


def decorator_f22(a: "complex128"):
    b = a
    return b


def union_type(a: int | float):
    return a * a


def return_annotation() -> int:
    my_var: int = 2
    return my_var


def wrong_argument_type(integer_arg: int):
    return integer_arg + 1


def wrong_known_argument_type_in_interface(templated_arg: T, integer_arg: int):
    return templated_arg + 1


def wrong_known_argument_type_in_interface_with_default(a: T, integer_arg: int = 5):
    return a + 1


def wrong_argument_combination_in_interface(a: T, b: T):
    return a + 1


def container_interface(a: Final[T_f]):
    return len(a)


def lambda_1(a: int):
    f1 = lambda x: x**2 + 1  # pylint: disable=unnecessary-lambda-assignment
    g1 = lambda x: f1(x) ** 2 + 1  # pylint: disable=unnecessary-lambda-assignment
    return g1(a)


def lambda_2(a: int):
    f2 = lambda x, y: x**2 + y**2 + 1  # pylint: disable=unnecessary-lambda-assignment
    return f2(a, 3 * a)


def add_2(a: float):
    return a + 2


def times_3(a: "float|complex"):
    b = 1.0
    b = add_2(b)
    a *= b
    return a
