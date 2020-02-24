from pyccel.decorators import types

@types('bool')
def is_false(a):
    c = False
    if a is False:
        c = True
    return c

@types('bool', 'bool')
def compare_is(a, b):
    c = False
    if a is b:
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

print(is_false(False))
print(compare_is(True,False))
print(not_true(True))
print(eq_false(True))
print(neq_true(False))
print(not_val(False))
