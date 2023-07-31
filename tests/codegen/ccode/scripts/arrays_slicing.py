# pylint: disable=missing-function-docstring, missing-module-docstring
#==============================================================================

def array_view_C_literals():

    from numpy import array
    z = array([[1, 2, 3], [4, 5, 6]])
    v = z[1:, 1:] #pylint:disable=unused-variable

def array_view_C_var():

    from numpy import array
    z = array([[1, 2, 3], [4, 5, 6]])
    a = 1
    v = z[a:, 0:2] #pylint:disable=unused-variable

def array_view_C_negative_indexes_literals():

    from numpy import full
    z = full((2, 5), 2)
    v = z[-1:, 0:-3] #pylint:disable=unused-variable

@allow_negative_index('y')
def array_view_C_negative_indexes_var():

    from numpy import full
    a = -1
    y = full((2, 5), 2)
    v = y[a:, 0:-3] #pylint:disable=unused-variable

def array_view_C_without_Slice_obj_literal():
    from numpy import ones

    z = ones((10, 10))
    v = z[2] #pylint:disable=unused-variable

def array_view_C_without_Slice_obj_var_pos():
    from numpy import zeros

    z = zeros((10, 10))
    i = 1
    v = z[i] #pylint:disable=unused-variable

@allow_negative_index('z')
def array_view_C_without_Slice_obj_negative():
    from numpy import zeros

    z = zeros((10, 10))
    i = -1
    v = z[i] #pylint:disable=unused-variable
