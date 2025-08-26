# The type of the result of a function definition cannot be a union of multiple types.
# pylint: disable=missing-function-docstring, missing-module-docstring


def f(a : int | float, b : int | float) -> int | float: #pylint: disable=unsupported-binary-operation
    return a+b
