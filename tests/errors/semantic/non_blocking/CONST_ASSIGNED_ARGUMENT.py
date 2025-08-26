# Cannot modify variable marked as Final
# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import Final

def funct_c(x : Final[int], a  : int):
    x += a
    return x
