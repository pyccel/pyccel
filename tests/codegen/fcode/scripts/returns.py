# pylint: disable=missing-function-docstring, missing-module-docstring
def max_(x: float, y: float):
    if x > y:
        z = x
        return z
    else:
        z = y
        return z


x = 5.0
y = 6.0
print(max_(x, y))
