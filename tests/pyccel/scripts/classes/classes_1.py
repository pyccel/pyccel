# pylint: disable=missing-class-docstring,  disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

#$ header class Point(public)
#$ header method __init__(Point, double[:])
#$ header method __del__(Point)
#$ header method translate(Point, double[:])
import numpy as np
class Point(object):
    def __init__(self, x):
        self.x = x

    def __del__(self):
        pass

    def translate(self, a):
        self.x[:]   =  self.x + a

if __name__ == '__main__':
    x = np.array([0.,0.,0.])
    p = Point(x)

    a = np.array([1.,1.,1.])

    p.translate(a)
    print(p.x)

