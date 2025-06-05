# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self : 'A', x : int):
        self.data : int = x

    def update(self : 'A', x : int):
        self.data = x

def get_A():
    return A(4)

def get_A_int():
    return A(2), 2

def get_x_from_A(a : 'A' = None):
    if a is not None:
        return a.data
    else:
        return 5
