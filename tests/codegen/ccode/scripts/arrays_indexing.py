# pylint: disable=missing-function-docstring, missing-module-docstring
#==============================================================================

def array_getitem_C():

    from numpy import array

    a = array([0, 0, 0])
    b = 0
    for i in range(3):
        b = a[i] + b

def array_setitem_C():

    from numpy import empty

    a = empty((5, 2))
    for i in range(5):
        for j in range(2):
            a[i][j] = 1

def array_negative_indexing_literal():

    from numpy import array

    a = array([1, 2, 3])
    a[-1] = a[-1] + 1

@allow_negative_index('a')
def array_negative_indexing():

    from numpy import array

    a = array([1, 2, 3])
    i = -1
    a[i] = a[i] + a[i - 1]
