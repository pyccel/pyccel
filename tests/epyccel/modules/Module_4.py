# pylint: disable=missing-function-docstring, missing-module-docstring

def basic_optional(a  : 'int' =  None):
    if a is  None :
        return 2
    return a

def call_optional_1():
    return basic_optional()

def call_optional_2(b  : 'int' =  None):
    return basic_optional(b)

def change_optional(a : int = None):
    if a is None:
        a = 4
    else:
        a += 3
    return 5+a

def optional_func_call():
    x = 3
    y = change_optional(x)
    return x,y
