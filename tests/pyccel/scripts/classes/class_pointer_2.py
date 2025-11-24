# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
import numpy as np

class A:
    def __init__(self):
        self.x = 4
        self.y = np.ones(5)

class B:
    def __init__(self, a : A):
        self._a = a

    @property
    def a(self):
        return self._a

if __name__ == '__main__':
    a = A()
    b = B(a)

    a_2 = b.a

    print(a_2.x)

