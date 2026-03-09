# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar
import math

T = TypeVar('T', int, float)

def inv_factorial(x : T) -> float:
    return 1 / math.factorial(int(x))
