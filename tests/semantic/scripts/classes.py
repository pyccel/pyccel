# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class Point(object):
    def __init__(self : 'Point', x : float, y : float):
        self.x = x
        self.y = y

    def __del__(self : 'Point'):
        pass

    def translate(self : 'Point', a : float, b : float):
        self.x = self.x + a
        self.y = self.y + b

p = Point(0.0, 0.0)
#x=p.x
p.x=5
