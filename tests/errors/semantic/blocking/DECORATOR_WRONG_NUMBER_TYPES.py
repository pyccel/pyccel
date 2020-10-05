from pyccel.decorators import types


@types('int')
def func(n,m):
    return n + m
