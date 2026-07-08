from typing import Final, TypeVar

I = TypeVar("I", "int8", "int16", "int32", "int64")
F = TypeVar("F", "float32", "float")
C = TypeVar("T", "complex64", "complex128")

def array_int_1d_scalar_add(x: "I[:]", a: I, x_len: int):
    for i in range(x_len):
        x[i] += a


def array_float_1d_scalar_add(x: "F[:]", a: F, x_len: int):
    for i in range(x_len):
        x[i] += a

def array_complex_1d_scalar_add(x: "C[:]", a: C, x_len: int):
    for i in range(x_len):
        x[i] += a

def array_int_2d_scalar_add(x: "I[:,:]", a: I, d1: int, d2: int):
    for i in range(d1):
        for j in range(d2):
            x[i, j] += a

def array_float_2d_scalar_add(x: "F[:,:]", a: F, d1: int, d2: int):
    for i in range(d1):
        for j in range(d2):
            x[i, j] += a

def array_complex_2d_scalar_add(x: "C[:,:]", a: C, d1: int, d2: int):
    for i in range(d1):
        for j in range(d2):
            x[i, j] += a

def array_final(x: "Final[float[:]]"):
    return x[0]

