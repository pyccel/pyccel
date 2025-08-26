# Too many arguments passed in function call
# pylint: disable=missing-function-docstring, missing-module-docstring

def f(b:'int'):
    print(b)

f(1, 2)
