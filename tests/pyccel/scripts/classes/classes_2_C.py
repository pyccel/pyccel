# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring
import numpy as np

#$ header class Point(public)
#$ header method __init__(Point, float, float)
#$ header method addition(Point, float, float)
#$ header method subtraction(Point, float[:], float[:])

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def addition(self, a, b):
        return a + b

    def subtraction(self, a, b):
        return a - b

if __name__ == '__main__':
    p = Point(10.0, 11.0)

    x = np.ones(4)
    y = np.full(4, 3.0)

    a = p.addition(1.1, 2.0)
    b = p.subtraction(y, x)

    print(a)
    print(b)
    print(p.addition(p.x, p.y))
    print(p.subtraction(x, y))
    print(p.addition(p.x, p.y) + 3.4)
