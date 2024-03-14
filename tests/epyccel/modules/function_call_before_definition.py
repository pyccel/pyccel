# pylint: disable=missing-function-docstring, missing-module-docstring

def f1():
    def g1():
        return x
    x = 1
    return g1()

x1 = f1()

def f2():
    return g2()

def g2():
    return 2

x2 = f2()

def f3():
    def fg1():
        return fg2()

    def fg2():
        return x
    x = 3
    return fg1()

x3 = f3()

