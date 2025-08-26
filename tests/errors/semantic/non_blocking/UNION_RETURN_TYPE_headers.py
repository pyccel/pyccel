# The type of the result of a function definition cannot be a union of multiple types.
# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeAlias

T : TypeAlias = 'int | float'

def f(a : T, b : T) -> T:
    return a+b
