# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self : 'A'):
        pass

class B:
    def __init__(self : 'B', x : float):
        self.x = x

class C:
    def __init__(self):
        pass

    def get_3(self):
        return 3

def get_A():
    return A()


def get_A_int():
    return A(), 3

def get_B(y : float):
    return B(y)

def get_x_from_B(b_obj : 'B'):
    return b_obj.x

def get_an_x_from_B(b_obj : 'B' = None):
    if b_obj is not None:
        return b_obj.x
    else:
        return -2.0
