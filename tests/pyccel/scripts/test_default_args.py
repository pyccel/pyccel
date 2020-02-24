# coding: utf-8
import numpy as np

from pyccel.decorators import types

#------------------------------------------------------------------------------
@types('int')
def f1(x = 1):
    y = x - 1
    return y

print(f1(2))
print(f1())

@types('real [:]','int')
def f5(x, m1 = 2):
    x[:] = 0.
    for i in range(0, m1):
        x[i] = i * 1.

# ...
m1 = 3

x_expected = np.zeros(m1)
f5(x_expected)
print(x_expected)
f5(x_expected, m1)


#------------------------------------------------------------------------------
@types('real','real')
def f3(x = 1.5, y = 2.5):
    return x+y

print(f3(19.2,6.7))
print(f3(4.5))
print(f3(y = 8.2))
print(f3())
