from pyccel.decorators import types

@types('int', 'int')
def func(n):
    return n
