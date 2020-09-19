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

@types('bool', 'int')
def compare_is_int(a, b):
    c = False
    if a is bool(b):
        c = True
    return c

@types('bool', 'int')
def compare_is_not_int(a, b):
    c = False
    if a is not bool(b):
        c = True
    return c

@types('int', 'int')
def compare_is_int_int(a, b):
    return a is b

@types('int', 'int')
def compare_isnot_int_int(a, b):
    return a is not b

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

@types('int')
def cast_int(a):
    b = int(a)
    return b

@types('bool')
def cast_bool(a):
    b = bool(a)
    return b

@types('float')
def cast_float(a):
    b = float(a)
    return b

@types('float')
def cast_float_to_int(a):
    b = int(a)
    return b

@types('int')
def cast_int_to_float(a):
    b = float(a)
    return b

@types('int')
def if_0_int(a):
    if a:
        return True
    else:
        return False

@types('real')
def if_0_real(a):
    if a:
        return True
    else:
        return False

@types('int','float')
def is_types(x,y):
    return x is y

@types('int','float')
def isnot_types(x,y):
    return x is not y

def none_is_none():
    return None is None

def none_isnot_none():
    return None is not None
