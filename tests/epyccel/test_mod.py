from pyccel.decorators import types

@types(int)
def g1(x):
    y = x+1
    return y

@types('int [:]')
def g2(x):
    y = x + 1
    return y
