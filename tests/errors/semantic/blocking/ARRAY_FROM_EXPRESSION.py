from pyccel.decorators import types

@types('int[:,:]','int[:,:]')
def f(x,y):
    z = x + y # pylint: disable=unused-variable
