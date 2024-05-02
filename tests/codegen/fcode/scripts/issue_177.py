# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# coding: utf-8
# pylint: disable=unused-variable

from numpy import zeros
from numpy import ones

def test_1():
    x = zeros(5)
    xp = x

    y = zeros((5,7))
    yp = y


#$ header class Coords(public)
#$ header method __init__(Coords, double [:])
class Coords:
    def __init__(self, m):
        self.major_radius = m

m = ones(5)
coor = Coords(m)
print(coor.major_radius)

test_1()
