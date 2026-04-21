# Memory allocation should not be used in an expression
# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy.random import rand

a = rand(5) * 5
