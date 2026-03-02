# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

T = TypeVar('T', int, float)

def sign(x : T) -> int:
    import numpy as np
    return int(np.sign(x))
