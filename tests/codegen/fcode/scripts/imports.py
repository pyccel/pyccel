# pylint: disable=missing-function-docstring, missing-module-docstring
#from expressions import ai
#xi = ai + 1
#
#from expressions import ad, bd
#xd = ad * bd
from numpy import ones
xi = 2

from decorators_types import decr
yi = decr(xi)

from classes_2 import Point

p = Point(ones(4))
