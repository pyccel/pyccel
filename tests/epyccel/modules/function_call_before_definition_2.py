# pylint: disable=missing-function-docstring, missing-module-docstring
def f():
    print(1)
    return g()

def g():
    print(2)
    return 2

a = f()

