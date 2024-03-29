# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self, a : int):
        self._a = a

def choose_A(a1 : A, a2 : A, b : bool):
    if b:
        return a1
    else:
        return a2
