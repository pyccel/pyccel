# pylint: disable=missing-function-docstring, missing-module-docstring
def f():
    return g() + do()

def g():
    return 2

def do():
    return 3

def do_0001():
    return 4


a = f()

