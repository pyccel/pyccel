# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

def g():
    return (2,3)

if __name__ == '__main__':
    shape = (1, 2)
    a = np.zeros(g())
    b = np.zeros((4, 1))
    c = np.zeros(shape)
    print(a.shape)
    print(b.shape)
    print(c.shape)
