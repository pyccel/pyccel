# coding: utf-8
from pyccel.decorators import types
import numpy as np

#------------------------------------------------------------------------------
@types('int')
def f1(x = 1):
    y = x - 1
    return y

@types('real [:]','int')
def f5(x, m1 = 2):
    x[:] = 0.
    for i in range(0, m1):
        x[i] = i * 1.

#------------------------------------------------------------------------------
@types('real','real')
def f3(x = 1.5, y = 2.5):
    return x+y

@types('bool')
def is_nil_default_arg(a = None):
    c = False
    if a is None:
        c = True
    return c

@types('real','real','bool')
def recursivity(x, y = 0.0, z = None):
    @types('bool')
    def is_not_nil(z = None):
        c = False
        if z is not None:
            c = True
        return c

    tmp = is_not_nil(z)
    if (tmp):
        y = 2.5
    return x + y

#------------------------------------------------------------------------------

print(f1(2))
print(f1())

# ...
m1 = 3

x_expected = np.zeros(m1)
f5(x_expected)
print(x_expected)
f5(x_expected, m1)


print(f3(19.2,6.7))
print(f3(4.5))
print(f3(y = 8.2))
print(f3())

print(is_nil_default_arg())
print(is_nil_default_arg(None))
print(is_nil_default_arg(False))


print(recursivity(19.2,6.7))
print(recursivity(4.5))
print(recursivity(19.2,6.7,True))
print(recursivity(4.5,z = False))
