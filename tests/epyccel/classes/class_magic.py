# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring

class A:
    def __init__(self, x : int):
        self.x = x

    def __add__(self, other : int):
        return A(self.x+other)

    def __sub__(self, other : int):
        return A(self.x-other)

    def __mul__(self, other : int):
        return A(self.x*other)

    def __truediv__(self, other : int):
        return A(int(self.x/other))

    def __lshift__(self, other : int):
        return A(self.x << other)

    def __rshift__(self, other : int):
        return A(self.x >> other)

    def __and__(self, other : int):
        return A(self.x & other)

    def __or__(self, other : int):
        return A(self.x | other)

    def __iadd__(self, other : int):
        self.x += other
        return self

    def __isub__(self, other : int):
        self.x -= other
        return self

    def __imul__(self, other : int):
        self.x *= other
        return self

    def __itruediv__(self, other : int):
        self.x = int(self.x / other)
        return self

    def __ilshift__(self, other : int):
        self.x <<= other
        return self

    def __irshift__(self, other : int):
        self.x >>= other
        return self

    def __iand__(self, other : int):
        self.x &= other
        return self

    def __ior__(self, other : int):
        self.x |= other
        return self

    def __len__(self):
        return 1
