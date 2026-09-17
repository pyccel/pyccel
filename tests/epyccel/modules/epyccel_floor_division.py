# pylint: disable=missing-function-docstring, missing-module-docstring


def fdiv_i_i_8(x: "int8", y: "int8"):
    return x // y


def fdiv_i_i_16(x: "int16", y: "int16"):
    return x // y


def fdiv_i_i_32(x: "int32", y: "int32"):
    return x // y


def fdiv_i_i_i(x: int, y: int, z: int):
    return x // y // z


def fdiv_b_b(x: "bool", y: "bool"):
    return x // y


def fdiv_i_r(x: int, y: "float"):
    return x // y


def fdiv_r_i(x: "float", y: int):
    return x // y


def fdiv_r_r(x: "float", y: "float"):
    return x // y
