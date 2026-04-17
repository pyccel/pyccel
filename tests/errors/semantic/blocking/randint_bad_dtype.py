# Unsupported dtype for randint
# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy.random import randint

a = randint(0, 100, size=5, dtype="float64")
