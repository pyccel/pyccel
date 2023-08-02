# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring
#$ header class Point(public)
#$ header method __init__(Point, double[:])
#$ header method translate(Point, double[:])
import numpy as np

class Point:
    def __init__(self, x):
        self.x = x

    def translate(self, a):
        self.x[:]   =  self.x + a

if __name__ == '__main__':
    x = np.array([0.,0.,0.])
    p = Point(x)

    a = np.array([1.,1.,1.])

    p.translate(a)
    print(p.x)
