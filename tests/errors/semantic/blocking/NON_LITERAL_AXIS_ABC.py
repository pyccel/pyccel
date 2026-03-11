# axis argument must be a literal
# pylint: disable=missing-module-docstring
import numpy as np

if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[6, 5, 4], [3, 2, 1]])
    axis = 0
    c = np.cross(a, b, axisa=axis, axisb=axis, axisc=axis)
