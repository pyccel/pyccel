# pylint: disable=missing-function-docstring, missing-module-docstring

@types('int', 'int', results=['int'])
@types('float', 'float', results=['float'])
def f(a, b):
    return a+b
