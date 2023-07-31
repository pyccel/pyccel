# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

def fill_a( r: float, a: 'int[:]' ):

from pyccel.decorators import types
    if ( r == 0.0 ):
        return 0

    for i in range(len(a)):#pylint: disable=consider-using-enumerate
        a[i] = 1.0 / r

    return 1

def get_sum():
    r = 4.0
    a = np.empty(10,dtype=int)
    fill_a ( r, a )
    result = 0
    for ai in a:
        result += ai
    return result

def fill_b( r: float, a: 'int[:]' ):

    if ( r == 0.0 ):
        return 0, 1.0

    for i in range(len(a)):#pylint: disable=consider-using-enumerate
        a[i] = 1.0 / r

    return 1, 1.0/r

def get_sum2():
    r = 4.0
    a = np.empty(10,dtype=int)
    fill_b ( r, a )
    result = 0
    for ai in a:
        result += ai
    return result
