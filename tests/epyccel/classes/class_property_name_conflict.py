# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self, x : float):
        self._x = x

    @property
    def x(self):
        return self._x

    def translate(self, y : float):
        x = self.x
        return x + y
