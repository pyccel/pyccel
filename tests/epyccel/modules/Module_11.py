# pylint: disable=missing-function-docstring, missing-module-docstring

def update_multiple():
    a = {1, 2, 3}
    a.update({4, 5})
    return len(a)

def set_union():
    a = {1,2,3,4,5}
    b = {4,5,6}
    c = a.union(b)
    return len(c)
