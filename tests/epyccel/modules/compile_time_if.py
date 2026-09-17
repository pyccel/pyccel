# pylint: disable=missing-function-docstring, missing-module-docstring


def rank_differentiation_1(a: "int[:] | int[:,:]"):
    if len(a.shape) == 1:
        return a[0]
    else:
        return a[0, 0]


def rank_differentiation_2(a: "int[:] | int[:,:]"):
    if len(a.shape) != 2:
        return a[0]
    else:
        return a[0, 0]


def type_differentiation(a: "int | float"):
    if isinstance(a, int):
        return a * 2
    else:
        return -a
