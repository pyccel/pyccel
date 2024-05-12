# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from endif import matmul

if __name__ == '__main__':
    a = np.ones((3,4))
    b = np.ones((4,3), order='F')
    c = np.empty((3,3))
    matmul(a, b, c)
    print(c)
