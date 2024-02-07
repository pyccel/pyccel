# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
# coding: utf-8

class Point(object):
    def __init__(self : 'Point', x : 'float[:]' = None):
        if x is not None:
            self._x = x

    def translate(self : 'Point', a : 'float[:]'):
        self._x[:]   =  self._x + a

    def get_x(self : 'Point'):
        return self._x[0]

