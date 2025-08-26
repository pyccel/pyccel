# Variable a goes out of scope but may be the target of a pointer which is still required
# pylint: disable=missing-function-docstring, missing-module-docstring

def f():
    a = [1, 2, 3]
    b = [[4,5,6]]
    b.append(a)
    return b
