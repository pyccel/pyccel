# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

class A:
    def __init__(self):
        self.x = 4

class B:
    def __init__(self, a : A):
        self._a = a

    @property
    def a(self):
        return self._a

if __name__ == '__main__':
    a = A()
    b = B(a)

    a_2 = b.a

    print(a_2.x)

