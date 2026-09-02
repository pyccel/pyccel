# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

T = TypeVar("T", int, float)


def f_call(x: int, y: int):
    return x**y


def pow_r_r(x: "float", y: "float"):
    return x**y


def pow_r_i(x: "float", y: "int"):
    return x**y


def pow_i_r(x: "int", y: "float"):
    return x**y


def pow_c_c(x: "complex", y: "complex"):
    return x**y


def pow_c_i(x: "complex", y: "int"):
    return x**y


def pow_c_r(x: "complex", y: "float"):
    return x**y


def pow_r_c(x: "float", y: "complex"):
    return x**y


def square(x: T):
    return x**2


def sqrt(x: T):
    return x**0.5


def fabs(x: T):
    return (x * x) ** 0.5


def norm(x: "complex"):
    return (x * x.conjugate()) ** 0.5


def complicated_abs(x: "complex"):
    return ((x * x.conjugate()).real ** 2) ** 0.5


def fcomplex_sqrt(x: complex, y: complex) -> complex:
    z = (x + y) ** 0.5
    return z


def chain_pow1(x: float, y: float, z: float):
    return x**y**z


def chain_pow2(x: float, y: float, z: float):
    return (x**y) ** z


def chain_pow3(x: float, y: float, z: float):
    return x ** (y**z)
