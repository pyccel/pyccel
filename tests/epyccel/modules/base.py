# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types

def is_false(a : 'bool'):
    c = False
    if a is False:
        c = True
    return c

def is_true(a : 'bool'):
    c = False
    if a is True:
        c = True
    return c

def compare_is(a : 'bool', b : 'bool'):
    c = False
    if a is b:
        c = True
    return c

def compare_is_not(a : 'bool', b : 'bool'):
    c = False
    if a is not b:
        c = True
    return c

def compare_is_int(a : 'bool', b : 'int'):
    c = False
    if a is bool(b):
        c = True
    return c

def compare_is_not_int(a : 'bool', b : 'int'):
    c = False
    if a is not bool(b):
        c = True
    return c

def not_false(a : 'bool'):
    c = False
    if a is not False:
        c = True
    return c

def not_true(a : 'bool'):
    c = False
    if a is not True:
        c = True
    return c

def eq_false(a : 'bool'):
    c = False
    if a == False:
        c = True
    return c

def eq_true(a : 'bool'):
    c = False
    if a == True:
        c = True
    return c

def neq_false(a : 'bool'):
    c = False
    if a != False:
        c = True
    return c

def neq_true(a : 'bool'):
    c = False
    if a != True:
        c = True
    return c

def not_val(a : 'bool'):
    c = False
    if not a:
        c = True
    return c

def not_int(a : 'int'):
    c = False
    if not a:
        c = True
    return c

def is_nil(a  : 'bool' =  None):
    c = False
    if a is None:
        c = True
    return c

def is_not_nil(a  : 'bool' =  None):
    c = False
    if a is not None:
        c = True
    return c

def cast_int(a : 'int'):
    b = int(a)
    return b

def cast_bool(a : 'bool'):
    b = bool(a)
    return b

def cast_float(a : 'float'):
    b = float(a)
    return b

def cast_float_to_int(a : 'float'):
    b = int(a)
    return b

def cast_int_to_float(a : 'int'):
    b = float(a)
    return b

def if_0_int(a : 'int'):
    if a:
        return True
    else:
        return False

def if_0_real(a : 'float'):
    if a:
        return True
    else:
        return False

def is_types(x : 'int', y : 'float'):
    return x is y

def isnot_types(x : 'int', y : 'float'):
    return x is not y

def is_same_int(x : 'int'):
    return x is x

def isnot_same_int(x : 'int'):
    return x is not x

def is_same_float(x : 'float'):
    return x is x

def isnot_same_float(x : 'float'):
    return x is not x

def is_same_complex(x : 'complex'):
    return x is x

def isnot_same_complex(x : 'complex'):
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

def pass_if(x : 'int'):
    if x > 0:
        pass
    x = x + 1
    return x

def pass2_if(b : 'float'):
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
