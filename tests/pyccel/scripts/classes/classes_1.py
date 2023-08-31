# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
# coding: utf-8

#$ header class Point(public)
#$ header method __init__(Point, double[:])
#$ header method __del__(Point)
#$ header method translate(Point, double[:])
#$ header method print_x(Point)
#$ header class Line(public)
#$ header method __init__(Line, Point)

import numpy as np
class Point(object):
    def __init__(self, x):
        self._X = 10
        self._x = x

    def __del__(self):
        pass

    def translate(self, a):
        self._x[:]   =  self._x + a

    def print_x(self):
        print(self._x)

class Line(object):
    def __init__(self, l):
        self.l = l
        self.l._X = 11
        print(self.l._X)

        l.print_x()

if __name__ == '__main__':
    x = np.array([0.,0.,0.])
    p = Point(x)

    a = np.array([1.,1.,1.])

    p.translate(a)
    p.print_x()
    k = Line(p)
