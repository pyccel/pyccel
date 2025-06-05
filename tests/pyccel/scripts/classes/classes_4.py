# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring

import numpy as np

y = np.ones(4)

class A:
    def __init__(self : 'A', n : int):
        self.x = np.ones(n)

    def __del__(self : 'A'):
        print(self.x)
        del self.x

if __name__ == '__main__':
    my_a = A(5)
    del my_a
    print(y[0])
