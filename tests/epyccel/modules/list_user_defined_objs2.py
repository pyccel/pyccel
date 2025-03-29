# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self, x : int):
        self.x = x

def fn():
    a = A(1)
    b = A(2)
    c = A(3)
    d = A(4)
    e = A(5)
    lst = [a, b, c]
    lst2 = [d, e]
    lst.extend(lst2)
    return len(lst)
