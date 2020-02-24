from pyccel.decorators import types

@types('bool')
def is_false(a):
    c = False
    if a is False:
        c = True
    return c

@types('bool')
def is_true(a):
    c = False
    if a is True:
        c = True
    return c

@types('bool', 'bool')
def compare_is(a, b):
    c = False
    if a is b:
        c = True
    return c

@types('bool', 'bool')
def compare_is_not(a, b):
    c = False
    if a is not b:
        c = True
    return c

@types('bool')
def not_false(a):
    c = False
    if a is not False:
        c = True
    return c

@types('bool')
def not_true(a):
    c = False
    if a is not True:
        c = True
    return c

@types('bool')
def eq_false(a):
    c = False
    if a == False:
        c = True
    return c

@types('bool')
def eq_true(a):
    c = False
    if a == True:
        c = True
    return c

@types('bool')
def neq_false(a):
    c = False
    if a != False:
        c = True
    return c

@types('bool')
def neq_true(a):
    c = False
    if a != True:
        c = True
    return c

@types('bool')
def not_val(a):
    c = False
    if not a:
        c = True
    return c

@types('bool')
def is_nil(a):
    c = False
    if a is None:
        c = True
    return c

@types('bool')
def is_not_nil(a):
    c = False
    if a is not None:
        c = True
    return c
