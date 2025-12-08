# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import pure

__all__ = [
        'homogeneous_tuple_int',
        'homogeneous_tuple_bool',
        'homogeneous_tuple_float',
        'homogeneous_tuple_string',
        'homogeneous_tuple_math',
        'homogeneous_tuple_containing_var',
        'homogeneous_tuple_of_arrays',
        'inhomogeneous_tuple_1',
        'inhomogeneous_tuple_2',
        'inhomogeneous_tuple_3',
        'inhomogeneous_tuple_2_levels_1',
        'inhomogeneous_tuple_2_levels_2',
        'homogeneous_tuple_2_levels',
        'tuple_unpacking_1',
        'tuple_unpacking_2',
        'tuple_unpacking_3',
        'tuple_unpacking_4',
        'tuple_unpacking_5',
        'tuple_name_clash',
        'tuples_as_indexing_basic',
        'tuples_as_indexing_var',
        'tuple_multi_indexing_1',
        'tuple_multi_indexing_2',
        'tuple_inhomogeneous_return',
        'tuple_homogeneous_return',
        'tuple_arg_unpacking',
        'tuple_indexing_basic',
        'tuple_indexing_2d',
        'tuple_visitation_homogeneous',
        'tuple_visitation_inhomogeneous',
        'tuples_homogeneous_have_pointers',
        'tuples_inhomogeneous_have_pointers',
        'tuples_homogeneous_copies_have_pointers',
        'tuples_inhomogeneous_copies_have_pointers',
        'tuples_mul_homogeneous',
        'tuples_mul_homogeneous2',
        'tuples_mul_homogeneous3',
        'tuples_mul_homogeneous4',
        'tuples_mul_homogeneous5',
        'tuples_mul_inhomogeneous',
        'tuples_mul_inhomogeneous2',
        'tuples_mul_homogeneous_2d',
        'tuples_mul_mixed_homogeneous_2d',
        'tuples_mul_inhomogeneous_2d',
        'tuples_add_homogeneous',
        'tuples_add_homogeneous_variables',
        'tuples_add_homogeneous_with_variables',
        'tuples_add_homogeneous_with_variables2',
        'tuples_add_inhomogeneous',
        'tuples_add_inhomogeneous_variables',
        'tuples_add_inhomogeneous_with_variables',
        'tuples_add_inhomogeneous_with_variables2',
        'tuples_add_mixed_homogeneous',
        'tuples_add_mixed_homogeneous_variables',
        'tuples_add_mixed_homogeneous_with_variables',
        'tuples_2d_sum',
        'tuples_func',
        'tuple_slice',
        'tuple_variable_index',
        'tuple_variable_slice',
        'tuple_negative_slice',
        'inhomogeneous_tuple_negative_slice',
        'inhomogeneous_tuple_var_negative_slice',
        'tuple_index',
        'tuple_homogeneous_int',
        'tuple_homogeneous_bool',
        'tuple_homogeneous_float',
        'tuple_homogeneous_string',
        'tuple_homogeneous_math',
        'tuple_inhomogeneous_1',
        'tuple_inhomogeneous_2',
        'tuple_inhomogeneous_3',
        'tuple_homogeneous',
        'tuple_inhomogeneous',
        'tuple_multilevel_inhomogeneous',
        'annotated_tuple_inhomog_return',
        'annotated_tuple_homog_return',
        'tuple_return_unknown_length',
        'tuple_assignment',
        'return_1_elem_inhomog_tuple',
        'return_empty_tuple',
        'return_empty_int_tuple',
        'return_annotated_empty_tuple',
        ]

def homogeneous_tuple_int():
    ai = (1,4,5)
    i = 1
    return ai[0], ai[i], ai[2]

def homogeneous_tuple_bool():
    ai = (False, True)
    i = 1
    return ai[0], ai[i]

def homogeneous_tuple_float():
    ai = (1.5, 4.3, 5.2, 7.2, 9.999)
    i = 1
    return ai[0], ai[i], ai[2], ai[3], ai[4]

def homogeneous_tuple_string():
    ai = ('hello', 'tuple', 'world', '!!')
    i = 1
    return ai[0], ai[i], ai[2], ai[3]

def homogeneous_tuple_math():
    ai = (4+5,3*9, 2**3)
    i = 1
    return ai[0], ai[i], ai[2]

def homogeneous_tuple_containing_var():
    elem = 4
    ai = (1, elem, 5)
    i = 1
    return ai[0], ai[i], ai[2]

def inhomogeneous_tuple_1():
    ai = (0, False, 3+1j)
    return ai[0], ai[1], ai[2]

def inhomogeneous_tuple_2():
    ai = (0, False, 3)
    return ai[0], ai[1], ai[2]

def inhomogeneous_tuple_3():
    ai = (0, 1.0, 3)
    return ai[0], ai[1], ai[2]

def inhomogeneous_tuple_2_levels_1():
    # TODO [EB 15.06.21] Put back original test when strings are supported in C
    #ai = ((1,2), (4,False), (3.0, 'boo'))
    ai = ((1,2), (4,False), (3.0, True))
    return ai[0][0], ai[0][1], ai[1][0], ai[1][1], ai[2][0]

def inhomogeneous_tuple_2_levels_2():
    ai = ((0,1,2), (True,False,True))
    return ai[0][0], ai[0][1] ,ai[0][2], ai[1][0], ai[1][1], ai[1][2]

def homogeneous_tuple_2_levels():
    ai = ((0,1,2), (3,4,5))
    return ai[0][0], ai[0][1] ,ai[0][2], ai[1][0], ai[1][1], ai[1][2]

def tuple_unpacking_1():
    ai = (1,False,3.5)
    a,b,c = ai
    return a,b,c

def tuple_unpacking_2():
    a,b,c = 1,False,3.5
    return a,b,c

def tuple_unpacking_3(x : 'int[:,:]'):
    x[0,0], x[1,0] = 2, 2

def tuple_unpacking_4(x : 'int[:,:]'):
    x[:,0], x[0,:] = 2, 3

def tuple_unpacking_5():
    ai = (1,2,3)
    a,b,c = ai
    return a,b,c

def tuple_name_clash():
    ai = (1+2j, False, 10.4)
    ai_0 = 44
    return ai_0, ai[0], ai[1], ai[2]

def tuples_as_indexing_basic():
    from numpy import ones
    x = ones((2,3,2))
    for z in range(2):
        for y in range(3):
            for w in range(2):
                x[z,y,w] = w+y*2+z*6
    idx = (1,1,0)
    return x[idx]

def tuples_as_indexing_var():
    from numpy import ones
    x = ones((2,3,2))
    for z in range(2):
        for y in range(3):
            for w in range(2):
                x[z,y,w] = w+y*2+z*6
    idx_0 = 1
    idx = (1,idx_0,0)
    return x[idx]

def tuple_multi_indexing_1():
    from numpy import ones
    x = ones((2,3,2))
    for z in range(2):
        for y in range(3):
            for w in range(2):
                x[z,y,w] = w+y*2+z*6
    idx = (1,1,0)
    ai = x[idx,0,1]
    return ai[0], ai[1], ai[2]

def tuple_multi_indexing_2():
    from numpy import ones
    x = ones((2,3,2))
    for z in range(2):
        for y in range(3):
            for w in range(2):
                x[z,y,w] = w+y*2+z*6
    idx = (1,1,0)
    idx_2 = (0,1,2)
    ai = x[idx,idx_2,1]
    return ai[0], ai[1], ai[2]

def tuple_inhomogeneous_return():
    ai = (7.5, False, 8)
    return ai

def tuple_homogeneous_return():
    ai = (7.5, 4.2, 8)
    return ai

def tuple_arg_unpacking():
    @pure
    def add2(x : 'int', y : 'int'):
        return x+y

    args = (3,4)
    z = add2(*args)
    return z

def tuple_indexing_basic():
    ai = (1,2,3,4)
    z = 0
    for i in range(4):
        z += ai[i]
    return z

def tuple_indexing_2d():
    ai = ((0,1,2), (True,False,True))
    z = 0
    for i in range(3):
        if ai[1][i]:
            z += ai[0][i]
    return z

def tuple_visitation_inhomogeneous():
    ai = (1,3.5, False)
    for a in ai:
        print(a)

def tuple_visitation_homogeneous():
    ai = (1,5, 4)
    for a in ai:
        print(a)

def tuples_homogeneous_have_pointers():
    from numpy import zeros
    a = zeros(2)
    b = zeros(2)
    c = (a,b)
    a[1] = 4
    return c[0][0], c[0][1], c[1][0], c[1][1]

def tuples_inhomogeneous_have_pointers():
    from numpy import zeros
    a = zeros(2)
    b = zeros(3)
    c = (a,b)
    a[1] = 4
    return c[0][0], c[0][1], c[1][0], c[1][1], c[1][2]

def tuples_homogeneous_copies_have_pointers():
    from numpy import zeros
    a = zeros(2)
    b = zeros(2)
    c = (a,b)
    d = c
    a[1] = 4
    return d[0][0], d[0][1], d[1][0], d[1][1]

def tuples_inhomogeneous_copies_have_pointers():
    from numpy import zeros
    a = zeros(2)
    b = zeros(3)
    c = (a,b)
    d = c
    a[1] = 4
    return d[0][0], d[0][1], d[1][0], d[1][1], d[1][2]

def tuples_mul_homogeneous():
    a = (1,2,3)
    b = a*2
    i = 1
    return b[0], b[i], b[2], b[3], b[4], b[5]

def tuples_mul_homogeneous2():
    a = (1,2,3)
    b = 2*a
    i = 1
    return b[0], b[i], b[2], b[3], b[4], b[5]

def tuples_mul_homogeneous3():
    a = (1,2,3)
    s = 2
    b = a*s
    i = 1
    return b[0], b[i], b[2], b[3], b[4], b[5]

def tuples_mul_homogeneous4():
    s = 2
    b = (1,2,3)*s
    i = 1
    return b[0], b[i], b[2], b[3], b[4], b[5]

def tuples_mul_homogeneous5():
    import numpy as np
    s = 2
    a = np.ones(5)
    b = (1,2,3)*(len(a)*s)
    i = 1
    return b[0], b[i], b[2], b[3], b[4], b[5]

def tuples_mul_inhomogeneous():
    a = (1,False)
    b = a*3
    return b[0], b[1], b[2], b[3], b[4], b[5]

def tuples_mul_inhomogeneous2():
    a = (1,False)
    b = 3*a
    return b[0], b[1], b[2], b[3], b[4], b[5]

def tuples_mul_homogeneous_2d():
    a= ((1,2), (3,4), (5,6))
    b = a*2
    return b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], \
            b[3][0], b[3][1], b[4][0], b[4][1], b[5][0], b[5][1]

def tuples_mul_mixed_homogeneous_2d():
    a= ((1,2), (True,False), (5,6))
    b = a*2
    return b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], \
            b[3][0], b[3][1], b[4][0], b[4][1], b[5][0], b[5][1]

def tuples_mul_inhomogeneous_2d():
    a= ((1,False), (3.0,False), (True,6))
    b = a*2
    return b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], \
            b[3][0], b[3][1], b[4][0], b[4][1], b[5][0], b[5][1]

def tuples_add_homogeneous():
    a = (1,2,3) + (4,5,6)
    i = 1
    return a[0], a[i], a[2], a[3], a[4], a[5]

def tuples_add_homogeneous_variables():
    a = (1,2,3)
    b = (4,5,6)
    c = a + b
    i = 1
    return c[0], c[i], c[2], c[3], c[4], c[5]

def tuples_add_homogeneous_with_variables():
    a = (1,2,3)
    c = a + (4,5,6)
    i = 1
    return c[0], c[i], c[2], c[3], c[4], c[5]

def tuples_add_homogeneous_with_variables2():
    a = (1,2,3)
    c = (4,5,6) + a
    i = 1
    return c[0], c[i], c[2], c[3], c[4], c[5]

def tuples_add_inhomogeneous():
    a = (1,2,True) + (False,5,6)
    return a[0], a[1], a[2], a[3], a[4], a[5]

def tuples_add_inhomogeneous_variables():
    a = (1,2,False)
    b = (4,5,True)
    c = a + b
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_add_inhomogeneous_with_variables():
    a = (1,2,True)
    c = a + (4,False,6)
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_add_inhomogeneous_with_variables2():
    a = (1,2,True)
    c = (4,False,6) + a
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_add_mixed_homogeneous():
    a = (1,2,3) + (False,5,6)
    return a[0], a[1], a[2], a[3], a[4], a[5]

def tuples_add_mixed_homogeneous_variables():
    a = (1,2,3)
    b = (4,5,True)
    c = a + b
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_add_mixed_homogeneous_with_variables():
    a = (1,2,3)
    c = a + (4,False,6)
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_2d_sum():
    a = ((1,2), (3,4))
    b = a + ((5,6),)
    i = 1
    return b[0][0], b[0][i], b[1][0], b[i][i], b[2][0], b[2][1]

def tuples_func():
    def my_tup():
        return 1, 2
    c = my_tup()
    return c[0], c[1]

def tuple_slice():
    a,b = (1,2,3)[:2]
    return a,b

def tuple_variable_index():
    a = 1
    b = (1,2,3)[a]
    return b

def tuple_variable_slice():
    a = 1
    b = (1,2,3)[:a]
    return b[0]

def tuple_negative_slice():
    a,b = (1,2,3)[:-1]
    return a,b

def inhomogeneous_tuple_negative_slice():
    a,b = (1,False,3)[:-1]
    return a,b

def inhomogeneous_tuple_var_negative_slice():
    c = (1,False,3)
    a,b = c[:-1]
    return a,b

def tuple_index():
    a = (1,2,3,False)[2]
    return a

def tuple_homogeneous_int():
    a = tuple((1, 2, 3))
    i = 1
    return a[0], a[i], a[2], len(a)

def tuple_homogeneous_bool():
    a = tuple((False, True))
    i = 1
    return a[0], a[i], len(a)

def tuple_homogeneous_float():
    a = tuple((1.5, 4.3, 5.2, 7.2, 9.999))
    i = 1
    return a[0], a[i], a[2], a[3], a[4], len(a)

def tuple_homogeneous_string():
    a = tuple(('hello', 'tuple', 'world', '!!'))
    i = 1
    return a[0], a[i], a[2], a[3], len(a)

def tuple_homogeneous_math():
    a = tuple((4 + 5, 3 * 9, 2 ** 3))
    i = 1
    return a[0], a[i], a[2], len(a)

def tuple_inhomogeneous_1():
    a = tuple((0, False, 3 + 1j))
    return a[0], a[1], a[2], len(a)

def tuple_inhomogeneous_2():
    a = tuple((0, False, 3))
    return a[0], a[1], a[2], len(a)

def tuple_inhomogeneous_3():
    a = tuple((0, 1.0, 3))
    return a[0], a[1], a[2], len(a)

def tuple_homogeneous():
    b = (10, 20, 30, 40)
    a = tuple(b)
    return a[0], a[1], a[2], a[3], len(a)

def tuple_inhomogeneous():
    b = ( 42, True, 3.14)
    a = tuple(b)
    return a[0], a[1], a[2], len(a)

def tuple_multilevel_inhomogeneous():
    a = (1,(2,(3,4)))
    return a[0], a[1][0], a[1][1][0], a[1][1][1]

def homogeneous_tuple_of_arrays():
    from numpy import array, empty
    x = array(((1,2), (3,4)), order='F')
    y = array(((5,4), (7,8)), order='F')
    z = array(((9,10), (11,12)), order='F')
    a = (x, y, z)
    b = empty((3,2,2))
    for j in range(2):
        for k in range(2):
            b[0,j,k] = a[0][j,k]
            b[1,j,k] = a[1][j,k]
            b[2,j,k] = a[2][j,k]
    return b

def annotated_tuple_inhomog_return() -> 'tuple[int,int]':
    return 1,2

def annotated_tuple_homog_return() -> 'tuple[int,...]':
    return 1,2

def tuple_return_unknown_length():
    b = False
    a = 1
    if b:
        return (a,)
    else:
        c = 2
        return (a, c)

def tuple_assignment():
    a : 'tuple[int,int]' = (1,2)
    b : 'tuple[int,...]' = a
    return b[0], b[1]

def return_1_elem_inhomog_tuple():
    a : 'tuple[int]' = (1,)
    return a

def return_empty_tuple():
    return ()

def return_empty_int_tuple() -> tuple[int,...]:
    return ()

def return_annotated_empty_tuple() -> tuple[()]:
    return ()
