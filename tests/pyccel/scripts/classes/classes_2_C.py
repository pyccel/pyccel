# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring
import numpy as np

class Point:
    def __init__(self : Point):
        pass

    def addition(self : Point, a : float, b : float):
        return a + b

    def subtraction(self : Point, a : 'float[:]', b : 'float[:]'):
        return a - b

if __name__ == '__main__':
    p = Point()

    x = np.ones(4)
    y = np.full(4, 3.0)

    a = p.addition(1.1, 2.0)
    b = p.subtraction(y, x)

    print(a)
    print(b)
    print(p.addition(10.0, 11.0))
    print(p.subtraction(x, y))
    print(p.addition(10.0, 11.0) + 3.4)
