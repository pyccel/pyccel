# TODO
#g = lambda xs,ys,z: [[x + y*z for x in xs] for y in ys]
#g = lambda xs,y,z: [x + y*z for x in xs]

import numpy as np
import time

from pyccel.decorators import types, pure
from pyccel.epyccel import epyccel
from pyccel.epyccel import lambdify
from pyccel.functional import add, mul

#=========================================================
#VERBOSE = True
VERBOSE = False

ACCEL = 'openmp'
#ACCEL = None

settings = {'accelerator': ACCEL, 'verbose': VERBOSE}
#=========================================================

#=========================================================
@pure
@types('double')
def f1(x):
    r = x**2
    return r

@pure
@types('double', 'double')
def f2(x,y):
    r = x*y
    return r

#=========================================================
def test_1():
    L = lambda xs: map(f1, xs)

    L = lambdify( L, namespace = {'f1': f1}, **settings )

#=========================================================
def test_2():
    L = lambda xs,ys:  map(f2, product(xs,ys))

    L = lambdify( L, namespace = {'f2': f2}, **settings )

#########################################
if __name__ == '__main__':
#    test_1()
    test_2()
