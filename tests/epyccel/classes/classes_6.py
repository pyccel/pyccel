# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

from pyccel.decorators import inline

class Point(object):
    def __init__(self : 'Point', x : float, y : float):
        self.x = x
        self.y = y

    def __del__(self : 'Point'):
        pass

    def translate(self : 'Point', a : float, b : float):
        self.x = self.x + a
        self.y = self.y + b

    @inline
    def get_inline_attributes(self : 'Point', a : 'int | float'):
        return self.x, self.y, a

    def get_attributes(self : 'Point', a : 'int | float'):
        x, y, b = self.get_inline_attributes(a)
        return x, y, b
