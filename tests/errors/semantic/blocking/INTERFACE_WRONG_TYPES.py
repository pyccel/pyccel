# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types
from pyccel.decorators import template

@template('z', types=['int', 'real'])
@template('y', types=['int', 'real'])
@types('z', 'z', 'y')
def multi_tmplt_1(x, y, z):
    """Tests Interfaces"""
    return x + y + z

def tst_multi_tmplt_1():
    """Tests call of the above function"""
    x = multi_tmplt_1(5, 5.3, 7)
    return x
