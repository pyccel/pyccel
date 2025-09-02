# Variable b goes out of scope but may be the target of a pointer which is still required
# pylint: disable=missing-function-docstring, missing-module-docstring

def f():
    b = [[1,2,3], [4,5,6]]
    c = b[0]
    return c
