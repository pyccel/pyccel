# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from pyccel.decorators import inline

class A:
    def __init__(self, x : float):
        self._x = x
        self._y = self._calculate_y(2)

    @property
    def x(self):
        return self._x

    @inline
    def _calculate_y(self, n : int):
        return self.x + n

    def get_A_contents(self):
        return self.x, self.y

    @inline
    @property
    def y(self):
        return self._y
