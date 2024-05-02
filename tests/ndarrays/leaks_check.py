# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy import array, zeros

def create_array():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])# pylint: disable=unused-variable

def array_to_pointer():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = a# pylint: disable=unused-variable

def view_assign():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = a[1:]# pylint: disable=unused-variable

def pointer_to_pointer():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = a
    c = b# pylint: disable=unused-variable

def pointer_reassign():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    c = array([[1, 2, 3, 4, 5, 0], [0, 6, 7, 8, 9, 10]])
    b = a# pylint: disable=unused-variable
    b = a[1:]
    b = c

def conditional_alloc(b1 : bool, b2 : bool):
    if b1:
        if b2:
            x = zeros(3, dtype=int)
            n = x.shape[0]
        else:
            n = 0
    else:
        x = zeros(4, dtype=int)
        n = x.shape[0]
    return n

def return_array():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    return a

def arrays_in_multi_returns():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = zeros(4)
    return a, b, 4

# testing garbage collecting in a Function

if __name__ == '__main__':
    create_array()
    array_to_pointer()
    view_assign()
    pointer_to_pointer()
    pointer_reassign()
    conditional_alloc(True,True)
    conditional_alloc(True,False)
    conditional_alloc(False,False)

    # testing garbage collecting in a Program

    z = return_array()
    y,x,s = arrays_in_multi_returns()
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = a
    c = a[1:]
    b = c[1:]
    b = c
