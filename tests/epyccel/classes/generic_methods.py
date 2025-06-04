# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from typing import TypeVar

T = TypeVar('T', int, float)

class Point(object):
    def __init__(self : 'Point', x : float, y : float):
        self.x = x
        self.y = y

    def __del__(self : 'Point'):
        pass

    def translate(self : 'Point', a : T, b : T):
        self.x = self.x + a
        self.y = self.y + b

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
