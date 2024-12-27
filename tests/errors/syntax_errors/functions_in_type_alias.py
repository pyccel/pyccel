# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeAlias

S = int | float
T = '(int)(int)' | '(real)(real)'

def f(g : 'T', a : 'S'):
    return g(a)

