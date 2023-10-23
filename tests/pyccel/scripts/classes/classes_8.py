# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

import numpy as np

class A:
    def __init__(self : 'A', x : 'float[:]', y : float):
        self.x = x
        self.y = y
    
    def set_attr(self : 'A', x : 'float[:]', y : float):
        self.x = x
        self.y = y
    
    def get_x(self : 'A'):
        print(self.x)

    def get_y(self : 'A'):
        print(self.y)


class B:
    def __init__(self: 'B', a : 'A'):
        self.A = a

    def set_A(self : 'B' , a : 'A'):
        self.A = a

def initiat_A(x : 'float[:]', y : float):
    return A(x, y)

if __name__ == "__main__":
    x = B(initiat_A(np.ones(4), 1.))
    x.A.get_x()
    x.A.get_y()
    x.A.set_attr(np.array([2., 3., 4., 5.]), 6.)
    x.A.get_x()
    x.A.get_y()
    x.set_A(initiat_A(np.array([7., 8., 9., 10.]), 11.))
    # print(x.A.y)
