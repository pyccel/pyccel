# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self, x : int):
        self.x = x

def fn():
    lst = [A(1), A(2), A(3)]
    lst.append(A(4))
    lst.append(A(5))
    return lst
