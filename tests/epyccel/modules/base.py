from pyccel.decorators import types

@types('bool', 'bool')
def cmp_bool(a, b):
    c = False
    if a == b:
        c = True
    return c
