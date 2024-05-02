# pylint: disable=missing-function-docstring, missing-module-docstring

def basic_optional(a  : 'int' =  None):
    if a is  None :
        return 2
    return a

def call_optional_1():
    return basic_optional()

def call_optional_2(b  : 'int' =  None):
    return basic_optional(b)
