# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
# coding: utf-8

class Point(object):
    def __init__(self, x : 'float[:]'):
        self._X = 10
        self._x = x

    #--------------------------------------
    def __del__(self):
        pass

    #--------------------------------------
    def translate(self, a : 'float[:]'):
        self._x[:]   =  self._x + a

    #--------------------------------------
    def get_x(self):
        return self._x

    #--------------------------------------
    #--------------------------------------
    def get_X(self):
        return self._X

class Line(object):
    def __init__(self, l : Point):
        self.l = l
        self.l._X = 11

        self._x = self.l.get_x()

    def get_x(self):
        return self._x
