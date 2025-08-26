# Expecting an argument of the same type as the elements of the set
# pylint: disable=missing-function-docstring, missing-module-docstring

a = {7, 9, 1}
b = 2j
a.discard(b)
