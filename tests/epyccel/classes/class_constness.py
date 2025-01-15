# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
import numpy as np

class ArrProperties:
    def __init__(self, n : int):
        self._n_pts = n
        self._default = 0

    @property
    def n_points(self):
        return self._n_pts

    @property
    def default_value(self):
        return self._default

def f(a : 'ArrProperties'):
    x = np.full(a.n_points, a.default_value)
    return x
