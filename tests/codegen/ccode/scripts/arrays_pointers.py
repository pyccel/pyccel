# pylint: disable=missing-function-docstring, missing-module-docstring/
#==============================================================================

def allocatable_to_pointer():

    from numpy import array

    a = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    c = a #pylint:disable=unused-variable

def pointer_to_pointer():

    from numpy import array

    a = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a
    c = b #pylint:disable=unused-variable

def reassinging_pointers():

    from numpy import array

    a = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a
    b = a[1:] #pylint:disable=unused-variable
