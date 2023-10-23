# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import template

@template('a', types=['int', 'real'])
@template('b', types=['int', 'real'])
def multi_tmplt_1(x : 'a', y : 'a', z : 'b'):
    """Tests Interfaces"""
    return x + y + z

def tst_multi_tmplt_1():
    """Tests call of the above function"""
    x = multi_tmplt_1(5, 5.3, 7)
    return x
