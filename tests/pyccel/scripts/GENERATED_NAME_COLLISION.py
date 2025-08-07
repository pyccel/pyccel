# pylint: disable=missing-function-docstring, missing-module-docstring

def f():
    do_0001 = 5
    return g() + do() + do_0001

def g():
    return 2

def do():
    return 4

if __name__ == '__main__':
    a = f()
