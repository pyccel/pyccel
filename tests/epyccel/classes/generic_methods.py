# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from pyccel.decorators import template

class Point(object):
    def __init__(self : 'Point', x : float, y : float):
        self.x = x
        self.y = y

    def __del__(self : 'Point'):
        pass

    @template('T', [int, float])
    def translate(self : 'Point', a : 'T', b : 'T'):
        self.x = self.x + a
        self.y = self.y + b

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
