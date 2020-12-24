# pylint: disable=missing-function-docstring, missing-module-docstring/
from numpy import array

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

# testing garbage collecting in a Function

create_array()
array_to_pointer()
view_assign()
pointer_to_pointer()
pointer_reassign()

# testing garbage collecting in a Program

a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
b = a
c = a[1:]
b = c[1:]
b = c
