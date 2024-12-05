# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
# coding: utf-8

class Point(object):
    def __init__(self, x : 'float[:]'):
        self._X = 10
        self._x = x

    def __del__(self):
        pass

    def translate(self, a : 'float[:]'):
        self._x[:]   =  self._x + a

    @property
    def x(self):
        return self._x

    @property
    def X(self):
        return self._X
