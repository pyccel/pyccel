from pyccel.decorators import types

# This has been done with c as an integer, until it works in the 
# fully Boolean version e.g.
#@types('bool', 'bool')
#def bool_compare_is(a, b):
#    c = False
#    if a is b:
#        c = True
#    return c

@types('bool')
def is_false(a):
    c = 0
    if a is False:
        c = 1
    return c

@types('bool')
def is_true(a):
    c = 0
    if a is True:
        c = 1
    return c

@types('bool', 'bool')
def compare_is(a, b):
    c = 1
    if a is b:
        c = 0
    return c

@types('bool')
def not_false(a):
    c = 0
    if a is not False:
        c = 1
    return c

@types('bool')
def not_true(a):
    c = 0
    if a is not True:
        c = 1
    return c

@types('bool')
def eq_false(a):
    c = 0
    if a == False:
        c = 1
    return c

@types('bool')
def eq_true(a):
    c = 0
    if a == True:
        c = 1
    return c

@types('bool')
def neq_false(a):
    c = 0
    if a != False:
        c = 1
    return c

@types('bool')
def neq_true(a):
    c = 0
    if a != True:
        c = 1
    return c
