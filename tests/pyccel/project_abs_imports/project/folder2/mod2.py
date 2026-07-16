# pylint: disable=missing-function-docstring, missing-module-docstring
from project.folder1.mod1 import sum_to_n

from pyccel.decorators import pure


@pure
def sum_to_n_squared(n: "int"):
    return sum_to_n(n) ** 2
