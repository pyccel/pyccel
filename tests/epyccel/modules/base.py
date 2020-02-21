from pyccel.decorators import types

@types('bool', 'bool')
def cmp_bool(a, b):
    c = False
    ar = 1.0
    br = 1.0
    cr = 1.0
    if ar == br:
        cr = 2.0
    if a == b:
        c = True
    return c
