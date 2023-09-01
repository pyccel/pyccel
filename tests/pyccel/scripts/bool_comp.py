# pylint: disable=missing-function-docstring, missing-module-docstring

def is_false(a : 'bool'):
    c = False
    if a is False:
        c = True
    return c

def compare_is(a : 'bool', b : 'bool'):
    c = False
    if a is b:
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

def is_nil_default_arg(a  : 'bool' =  None):
    c = False
    if a is None:
        c = True
    return c

if __name__ == '__main__':
    print(is_false(False))
    print(compare_is(True,False))
    print(not_true(True))
    print(eq_false(True))
    print(neq_true(False))
    print(not_val(False))
    print(is_nil(True))
    print(is_nil(None))
    print(is_not_nil(True))
    print(is_not_nil(None))
    print(is_nil_default_arg(True))
    print(is_nil_default_arg(None))
    print(is_nil_default_arg())
