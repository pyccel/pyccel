# pylint: disable=missing-function-docstring, missing-module-docstring/
#==============================================================================

def array_augassign():

    from numpy import array

    a = array([0, 0, 0])
    b = array([1, 1, 1])
    a +=b
