# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self, x : float):
        self._x = x
        self._y : float
        self._construct_y()
        self._construct_y_from_z(self.y)
        self._construct_y_from_z(3)

    def _construct_y(self):
        self._y = self.x + 3

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def _construct_y_from_z(self, z : 'int | float'):
        self._y = self._x + z
