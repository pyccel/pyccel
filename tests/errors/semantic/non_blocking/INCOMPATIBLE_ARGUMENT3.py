# Argument 2 : 2.3
# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def f(self, x : int):
        y = x + 1
        return y

my_a = A()
z = my_a.f(2.3)
print(z)
