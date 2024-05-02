# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from numpy import ones
#$ header class Point(public)
#$ header method __init__(Point, [double])
#$ header method __del__(Point)
#$ header method translate(Point, [double])

#$ header class Points(public)
#$ header method __init__(Points, Point)
#$ header method __del__(Points)

class Point(object):
    def __init__(self, x):
        self.x = x

    def __del__(self):
        pass

    def translate(self, a):
        self.x = self.x + a


class Points(object):
    def __init__(self, x):
        self.x = x

    def __del__(self):
        pass

x = [1., 1., 1.]
P1 = Point(x)
P2 = Points(P1)
P3 = P2.x
P4 = P2
P5 = P2.x.x
print(x,P5)


