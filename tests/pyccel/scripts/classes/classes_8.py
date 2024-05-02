# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

import numpy as np

class MyClass:
    def __init__(self : 'MyClass', x : 'float[:]', y : float):
        self.x = x
        self.y = y

    def set_attr(self : 'MyClass', x : 'float[:]', y : float):
        self.x = x
        self.y = y

    def get_x(self : 'MyClass'):
        print(self.x)

    def get_y(self : 'MyClass'):
        print(self.y)

class B:
    def __init__(self: 'B', param : 'MyClass'):
        self.A = param

    def set_A(self : 'B' , param : 'MyClass'):
        self.A = param

def initiat_MyClass(x : 'float[:]', y : float):
    return MyClass(x, y)

if __name__ == "__main__":
    x = B(initiat_MyClass(np.array([1., 1., 1., 1.]), 1.))
    x.A.get_x()
    x.A.get_y()
    x.A.set_attr(np.array([2., 3., 4., 5.]), 6.)
    x.A.get_x()
    x.A.get_y()
    x.set_A(initiat_MyClass(np.array([7., 8., 9., 10.]), 11.))
    print(x.A.y)
