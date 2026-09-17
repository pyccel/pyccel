# pylint: disable=missing-function-docstring, missing-module-docstring


def mod_eq_pow(a: int, m: int, n: int):
    return a % m == n**2


def mod_neq_pow(a: int, m: int, n: int):
    return a % m != n**2


def idiv_gt_add(a: int, m: int, n: int):
    return a // m > n + 1


def in_interval(a: float, b: float, c: float):
    return a <= b < c
