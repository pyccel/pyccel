# Memory allocation should not be used in an expression
# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy.random import randint

a = randint(10, size=5)
