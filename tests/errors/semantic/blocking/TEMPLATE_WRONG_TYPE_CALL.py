# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types
from pyccel.decorators import template

@template('z', types=['int', 'real'])
@template('y', types=['int', 'real'])
@types('z', 'z', 'y')
def multi_tmplt_1(x, y, z):
    return x + y + z

@template('z', types=['int'])
@template('y', types=['int', 'real'])
@types('z', 'y')
def multi_tmplt_2(y, z):
    return y + z

def tst_multi_tmplt_2():
    x = multi_tmplt_2(5.4, 5.4)
    return x
