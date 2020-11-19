# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types
#==============================================================================

def array_create_C_literal_value():

    from numpy import array

    a = array([0,0,0])

# this is a bug need to get fixed #546
#def array_create_C_variable():
#    from numpy import array
#    a = array([0, 0, 0])
#    b = array(a)

def array_full_C():

    from numpy import full, float

    a = full((2, 2), 1,dtype=float)

def array_empty_c():

    from numpy import empty

    a = empty((5,2))

def array_ones_c():

    from numpy import ones

    a = ones((5,2))

def array_zeros_c():

    from numpy import zeros

    a = zeros((5,2))

def array_empty_like_c():

    from numpy import array, empty_like

    a = array([1,2,3,3,4,5])
    b = empty_like(a, float)

def array_full_like_c():

    from numpy import array, full_like, complex128

    a = array([1,2,3,3,4,5])
    b = full_like(a, 0.3+ 0.3j, complex128)

def array_ones_like_c():

    from numpy import array, ones_like

    a = array([1,2,3,3,4,5])
    b = ones_like(a)

def array_zeros_like_c():

    from numpy import array, zeros_like

    a = array([1,2,3,3,4,5])
    b = zeros_like(a, float)

