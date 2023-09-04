# pylint: disable=missing-function-docstring, missing-module-docstring
@pure
def square(x : float):
    s = x*x
    return s

a = 2.0
b = square(a)
print(b)
