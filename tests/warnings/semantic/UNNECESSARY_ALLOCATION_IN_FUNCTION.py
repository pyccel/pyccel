# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

def Mi(x: float, M0: 'float[:, :]') -> 'float[:, :]':
    M1 = x * M0 + 1.
    return M1

def M_GEN(x: float) -> 'float[:, :, :]':
    M = np.ones((4, 4), dtype = float)
    M2 = np.empty((3, 4, 4), dtype = float)
    for i in range(3):
        M2[i] = Mi(x * i, M)
    return M2
