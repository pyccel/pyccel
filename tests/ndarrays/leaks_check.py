# pylint: disable=missing-function-docstring
from numpy import array

def create_array():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

def array_to_pointer():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = a

def view_assign():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = a[1:]

def pointer_to_pointer():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = a
    c = b

def pointer_reassign():
    a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    c = array([[1, 2, 3, 4, 5, 0], [0, 6, 7, 8, 9, 10]])
    b = a
    b = a[1:]
    b = c

create_array()
array_to_pointer()
view_assign()
pointer_to_pointer()
pointer_reassign()
