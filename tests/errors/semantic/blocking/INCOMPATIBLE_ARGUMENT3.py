# pylint: disable=missing-function-docstring, missing-module-docstring

from pyccel.decorators import inline
@inline
def add(a:int,b:int):
    return a+b

z = add(3.,2.)
