# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring
import numpy as np

#$ header class Point(public)
#$ header method __init__(Point)
#$ header method addition(Point, float, float)
#$ header method subtraction(Point, float[:], float[:])

class Point:
    def __init__(self):
        pass

    def addition(self, a, b):
        return a + b

    def subtraction(self, a, b):
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
