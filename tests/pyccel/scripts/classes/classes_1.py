# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
# coding: utf-8
import numpy as np

class Point(object):
    def __init__(self : 'Point', x : 'float[:]'):
        self._X = 10
        self._x = x

    def __del__(self : 'Point'):
        pass

    def translate(self : 'Point', a : 'float[:]'):
        self._x[:]   =  self._x + a

    def print_x(self : 'Point'):
        print(self._x)

    def print_X(self : 'Point'):
        print(self._X)

class Line(object):
    def __init__(self : 'Line', l : Point):
        self.l = l
        self.l._X = 11
        print(self.l._X)

        self.l.print_x()

if __name__ == '__main__':
    x = np.array([0.,0.,0.])
    p = Point(x)

    a = np.array([1.,1.,1.])

    p.translate(a)
    p.print_x()
    p.print_X()

    p = Point(np.array([6.,6.,6.]))
    p.translate(a)
    p.print_x()
