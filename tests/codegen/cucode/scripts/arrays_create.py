# pylint: disable=missing-function-docstring, missing-module-docstring
#==============================================================================

def array_create_C_literal_value():

    from numpy import array

    a = array([0, 0, 0]) #pylint:disable=unused-variable

# this is a bug need to get fixed #546
#def array_create_C_variable():
#    from numpy import array
#    a = array([0, 0, 0])
#    b = array(a) #pylint:disable=unused-variable

def array_full_C():

    from numpy import full

    a = full((2, 2), 1, dtype=float) #pylint:disable=unused-variable

def array_empty_c():

    from numpy import empty

    a = empty((5, 2)) #pylint:disable=unused-variable

def array_ones_c():

    from numpy import ones

    a = ones((5, 2)) #pylint:disable=unused-variable

def array_zeros_c():

    from numpy import zeros

    a = zeros((5, 2)) #pylint:disable=unused-variable

def array_empty_like_c():

    from numpy import array, empty_like

    a = array([1, 2, 3, 3, 4, 5])
    b = empty_like(a, float) #pylint:disable=unused-variable

def array_full_like_c():

    from numpy import array, full_like, complex128

    a = array([1, 2, 3, 3, 4, 5])
    b = full_like(a, 0.3 + 0.3j, complex128) #pylint:disable=unused-variable

def array_ones_like_c():

    from numpy import array, ones_like

    a = array([1, 2, 3, 3, 4, 5])
    b = ones_like(a) #pylint:disable=unused-variable

def array_zeros_like_c():

    from numpy import array, zeros_like

    a = array([1, 2, 3, 3, 4, 5])
    b = zeros_like(a, float) #pylint:disable=unused-variable
