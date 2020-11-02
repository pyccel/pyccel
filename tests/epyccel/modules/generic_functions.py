# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

#$ header function decr(real, int) 
#$ header function decr(int, real) 

def decr(y, x): 
    z = y + x 
    return z

@types('int')
@types('real')
def f1(a):
    return a / 10

def decr(y, x): 
    z = y + x 
    return z


