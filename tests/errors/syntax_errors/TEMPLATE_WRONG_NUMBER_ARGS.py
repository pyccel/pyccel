# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types
from pyccel.decorators import template

@template(name='O', types=['int', 'real'], test='int') # pylint: disable=unexpected-keyword-arg
@types('O', 'O')
def tmplt_1(x, y):
    return x + y

def tst_tmplt_1():
    x = tmplt_1(5, 4)
    y = tmplt_1(6.56, 3.3)
    return x * y
