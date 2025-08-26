# Too few arguments passed in function call
# pylint: disable=missing-function-docstring, missing-module-docstring, no-value-for-parameter

def f(b:'int'):
    print(b)

f()
