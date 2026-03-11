# pylint: disable=missing-function-docstring, missing-module-docstring

__all__ = ("f", "g", "h")


def f(x: "float [:]"):
    x[0] = 2.0


def g(x: "float [:]"):
    x[1] = 4.0


def h(x: "float [:]"):
    x[2] = 8.0
