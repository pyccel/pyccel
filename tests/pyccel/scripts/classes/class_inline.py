# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from typing import Final
import numpy as np
from importable import calculate_sum
from pyccel.decorators import pure, inline

class MyClass:
    def __init__(self, n : int):
        self._x = np.ones(n)
        self._y = np.full(n, 22.5)

    @inline
    @property
    def x(self):
        return self._x

    @inline
    @property
    def y(self):
        return self._y

    @pure
    def calculate_sum(self : 'Final[MyClass]'):
        return calculate_sum(self.x) + calculate_sum(self.y)

if __name__ == '__main__':
    c = MyClass(10)
    print(c.calculate_sum())
