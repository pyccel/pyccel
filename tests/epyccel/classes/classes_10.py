# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring
class A:
    def __init__(self, value: int):
        self.value = value

    def __getitem__(self, index: int):
        return B(self)

class B:
    def __init__(self, a: A):
        self.a = a
