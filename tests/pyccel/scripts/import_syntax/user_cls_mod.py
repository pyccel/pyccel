# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self, n : int):
        self._n = n

    @property
    def my_val(self):
        return self._n * 10

