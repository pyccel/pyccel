# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

from numpy import zeros

x1 = zeros(4)
x2 = zeros((4,3))

## bad
x1 = 1.
x2 = 1.

# good
x1[:] = 1.
x2[:,:] = 1.
