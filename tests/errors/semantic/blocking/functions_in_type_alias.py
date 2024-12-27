# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeAlias

S : TypeAlias = int | float
T : TypeAlias = '(int)(int)' | '(real)(real)'

def f(g : 'T', a : 'S'):
    return g(a)

