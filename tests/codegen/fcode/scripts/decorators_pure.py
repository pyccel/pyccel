# pylint: disable=missing-function-docstring, missing-module-docstring/
@pure
@types(float)
def square(x):
    s = x*x
    return s

a = 2.0
b = square(a)
print(b)
