# pylint: disable=missing-function-docstring, missing-module-docstring


def div_i_i(x: int, y: int):
    return x / y


def div_i_r(x: int, y: "float"):
    return x / y


def div_r_i(x: "float", y: int):
    return x / y


def div_r_r(x: "float", y: "float"):
    return x / y


def div_c_c(x: "complex", y: "complex"):
    return x / y


def div_i_c(x: int, y: "complex"):
    return x / y


def div_c_i(x: "complex", y: int):
    return x / y


def div_r_c(x: "float", y: "complex"):
    return x / y


def div_c_r(x: "complex", y: "float"):
    return x / y
