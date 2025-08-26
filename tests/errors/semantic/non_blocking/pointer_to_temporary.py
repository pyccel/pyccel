# Pointer cannot point at a temporary object
# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from numpy import ones

class Point(object):
    def __init__(self, x : 'float[:]'):
        self.x = x

    def __del__(self):
        pass

    def translate(self, a : 'float[:]'):
        self.x = self.x + a


class Points(object):
    def __init__(self, x : Point):
        self.x = x

    def __del__(self):
        pass

x = ones(4)
P1 = Point(x)
P2 = Points(P1)
P3 = P2.x
P4 = P2
P5 = P2.x.x
print(x,P5)
