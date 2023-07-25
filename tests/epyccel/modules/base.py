# pylint: disable=missing-function-docstring, missing-module-docstring
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

@types('int')
def not_int(a):
    c = False
    if not a:
        c = True
    return c

@types('bool')
def is_nil(a = None):
    c = False
    if a is None:
        c = True
    return c

@types('bool')
def is_not_nil(a = None):
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

@types('int')
def is_same_int(x):
    return x is x

@types('int')
def isnot_same_int(x):
    return x is not x

@types('float')
def is_same_float(x):
    return x is x

@types('float')
def isnot_same_float(x):
    return x is not x

@types('complex')
def is_same_complex(x):
    return x is x

@types('complex')
def isnot_same_complex(x):
    return x is not x

def is_same_string():
    x = 'hello world'
    return x is x

def isnot_same_string():
    x = 'hello world'
    return x is not x

def none_is_none():
    return None is None

def none_isnot_none():
    return None is not None

@types('int')
def pass_if(x):
    if x > 0:
        pass
    x = x + 1
    return x

@types('real')
def pass2_if(b):
    c = 1
    if b:
        pass
    else:
        c = 2
    return c

def use_optional(a : int = None):
    b = 3
    if a:
        b += a
    return b

def none_equality(a : int = None):
    return a == None, a != None #pylint: disable=singleton-comparison

def none_none_equality():
    return None == None, None != None #pylint: disable=singleton-comparison, comparison-with-itself

def none_literal_equality():
    return None == 1, 3.5 != None #pylint: disable=singleton-comparison
