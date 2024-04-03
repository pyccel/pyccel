# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self, n : int):
        self.ls = [n, n]

class B:
    def __init__(self, a : A):
        self.a = a

def fn():
    a = (3, 4, 5)
    my_a = A(2)
    my_b = B(my_a)
    my_b.a.ls.extend([3,4])
    my_b.a.ls.extend((3,4))
    my_b.a.ls.extend(range(0, 4))
    my_b.a.ls.extend(a)
    store = [1]
    store.extend(my_b.a.ls)
    return store

