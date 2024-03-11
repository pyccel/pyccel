# pylint: disable=missing-function-docstring, missing-module-docstring
#$ header max_(double, double)
def max_(x, y):
    if x>y:
        z = x
        return z
    else:
        z = y
        return z



x = 5.
y= 6.
print(max_(x, y))
