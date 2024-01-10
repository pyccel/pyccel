# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self : 'A'):
        pass

class B:
    def __init__(self : 'B', x : float):
        self.x = x

def get_A():
    return A()


def get_A_int():
    return A(), 3

def get_B(y : float):
    return B(y)
