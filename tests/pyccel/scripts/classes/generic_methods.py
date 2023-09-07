# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from pyccel.decorators import template

class Point(object):
    def __init__(self : Point, x : float, y : float):
        self.x = x
        self.y = y

    def __del__(self : Point):
        pass

    @template('T', [int, float])
    def translate(self : Point, a : 'T', b : 'T'):
        self.x = self.x + a
        self.y = self.y + b

if __name__ == '__main__':
    p = Point(0.0, 0.0)
    x=p.x
    p.x=x
    a = p.x
    a = p.x - 2
    a = 2 * p.x - 2
    a = 2 * (p.x + 6) - 2

    p.y = a + 5
    p.y = p.x + 5

    p.translate(1.0, 2.0)

    print(p.x, p.y)
    print(a)
