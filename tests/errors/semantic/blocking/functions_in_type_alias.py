# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeAlias

S : TypeAlias = 'int | float'
T : TypeAlias = '(int)(int) | (float)(float)'

def f(g : 'T', a : 'S'):
    return g(a)

