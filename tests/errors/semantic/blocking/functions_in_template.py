# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import template

@template('S', [int, float])
@template('T', ['(int)(int)', '(real)(real)'])
def f(g : 'T', a : 'S'):
    return g(a)

