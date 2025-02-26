# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

if __name__ == '__main__':
    a = 1
    b = 0
    c = np.array([1,2,3,4])

    a, b, c[a+b] = c[a+b], a, b
