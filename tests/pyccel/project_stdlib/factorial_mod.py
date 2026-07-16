# pylint: disable=missing-function-docstring, missing-module-docstring
import math
from typing import TypeVar

T = TypeVar("T", int, float)


def inv_factorial(x: T) -> float:
    return 1 / math.factorial(int(x))
