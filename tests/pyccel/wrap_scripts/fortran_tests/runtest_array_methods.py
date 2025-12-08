# pylint: disable=missing-module-docstring
from array_methods import ArrayOps
import numpy as np

if __name__ == "__main__":
    a = ArrayOps()
    a.set_data(np.array([1.0, 2.0, 3.0]), np.int32(3))
    print(a.sum())   # expected 6.0
    a *= 2.0
    print(a.sum())   # expected 12.0


