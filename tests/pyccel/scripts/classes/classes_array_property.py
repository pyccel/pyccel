import numpy as np

class A:
    def __init__(self, n : int):
        self._x = np.ones(n)

    @property
    def x(self):
        return self._x
