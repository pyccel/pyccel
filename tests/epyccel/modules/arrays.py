# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar
import numpy as np

from pyccel.decorators import stack_array, allow_negative_index

a_1d   = np.array([1 << i for i in range(21)], dtype=int)
a_1d_f = np.array([1 << i for i in range(21)], dtype=int, order="F")
a_2d_f = np.array([[1 << j for j in range(21)] for i in range(21)], dtype=int, order='F')
a_2d_c = np.array([[1 << j for j in range(21)] for i in range(21)], dtype=int)

T = TypeVar('T', 'int[:]', 'int[:,:]', 'int[:,:,:]', 'int[:,:](order=F)', 'int[:,:,:](order=F)')
T2D = TypeVar('T2D', 'bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]',
                     'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]',
                     'float64[:,:]', 'complex64[:,:]', 'complex128[:,:]')
S = TypeVar('S' , 'int', 'int8', 'int16', 'int32', 'int64', 'float',
                  'float32', 'float64', 'complex64', 'complex128')

def array_return_first_element(a : S, b : S):
    x = np.array([a,b])
    return x[0]

#==============================================================================
# 1D ARRAYS OF INT-32
#==============================================================================

def array_int32_1d_scalar_add(x : 'int32[:]', a : 'int32'):
    x[:] += a

def array_int32_1d_scalar_sub(x : 'int32[:]', a : 'int32'):
    x[:] -= a

def array_int32_1d_scalar_mul(x : 'int32[:]', a : 'int32'):
    x[:] *= a

def array_int32_1d_scalar_div(x : 'int32[:]', a : 'int32'):
    x[:] = x / a

def array_int32_1d_scalar_idiv(x : 'int32[:]', a : 'int32'):
    x[:] = x // a

def array_int32_1d_add(x : 'int32[:]', y : 'int32[:]'):
    x[:] += y

def array_int32_1d_sub(x : 'int32[:]', y : 'int32[:]'):
    x[:] -= y

def array_int32_1d_mul(x : 'int32[:]', y : 'int32[:]'):
    x[:] *= y

def array_int32_1d_idiv(x : 'int32[:]', y : 'int32[:]'):
    x[:] = x // y

def array_int32_1d_add_augassign(x : 'int32[:]', y : 'int32[:]'):
    x += y

def array_int32_1d_sub_augassign(x : 'int32[:]', y : 'int32[:]'):
    x -= y

def array_int_1d_initialization_1():
    a = np.array([1, 2, 4, 8, 16])
    b = np.array(a)
    return np.sum(b), b[0], b[-1]

def array_int_1d_initialization_2():
    a = [1, 2, 4, 8, 16]
    b = np.array(a)
    return np.sum(b), b[0], b[-1]

def array_int_1d_initialization_3():
    a = (1, 2, 4, 8, 16)
    b = np.array(a)
    return np.sum(b), b[0], b[-1]

def array_int_1d_initialization_4():
    b = np.array([i*2 for i in range(10)])
    return b

#==============================================================================
# 2D ARRAYS OF INT-32 WITH C ORDERING
#==============================================================================

def array_int32_2d_C_scalar_add(x : 'int32[:,:]', a : 'int32'):
    x[:,:] += a

def array_int32_2d_C_scalar_sub(x : 'int32[:,:]', a : 'int32'):
    x[:,:] -= a

def array_int32_2d_C_scalar_mul(x : 'int32[:,:]', a : 'int32'):
    x[:,:] *= a

def array_int32_2d_C_scalar_idiv(x : 'int32[:,:]', a : 'int32'):
    x[:,:] = x // a

def array_int32_2d_C_add(x : 'int32[:,:]', y : 'int32[:,:]'):
    x[:,:] += y

def array_int32_2d_C_sub(x : 'int32[:,:]', y : 'int32[:,:]'):
    x[:,:] -= y

def array_int32_2d_C_mul(x : 'int32[:,:]', y : 'int32[:,:]'):
    x[:,:] *= y

def array_int32_2d_C_idiv(x : 'int32[:,:]', y : 'int32[:,:]'):
    x[:,:] = x // y

#==============================================================================
# 2D ARRAYS OF INT-32 WITH F ORDERING
#==============================================================================

def array_int32_2d_F_scalar_add(x : 'int32[:,:](order=F)', a : 'int32'):
    x[:,:] += a

def array_int32_2d_F_scalar_sub(x : 'int32[:,:](order=F)', a : 'int32'):
    x[:,:] -= a

def array_int32_2d_F_scalar_mul(x : 'int32[:,:](order=F)', a : 'int32'):
    x[:,:] *= a

def array_int32_2d_F_scalar_idiv(x : 'int32[:,:](order=F)', a : 'int32'):
    x[:,:] = x // a

def array_int32_2d_F_add(x : 'int32[:,:](order=F)', y : 'int32[:,:](order=F)'):
    x[:,:] += y

def array_int32_2d_F_sub(x : 'int32[:,:](order=F)', y : 'int32[:,:](order=F)'):
    x[:,:] -= y

def array_int32_2d_F_mul(x : 'int32[:,:](order=F)', y : 'int32[:,:](order=F)'):
    x[:,:] *= y

def array_int32_2d_F_idiv(x : 'int32[:,:](order=F)', y : 'int32[:,:](order=F)'):
    x[:,:] = x // y


#==============================================================================
# 1D ARRAYS OF INT-64
#==============================================================================

def array_int_1d_scalar_add(x : 'int[:]', a : 'int'):
    x[:] += a

def array_int_1d_scalar_sub(x : 'int[:]', a : 'int'):
    x[:] -= a

def array_int_1d_scalar_mul(x : 'int[:]', a : 'int'):
    x[:] *= a

def array_int_1d_scalar_idiv(x : 'int[:]', a : 'int'):
    x[:] = x // a

def array_int_1d_add(x : 'int[:]', y : 'int[:]'):
    x[:] += y

def array_int_1d_sub(x : 'int[:]', y : 'int[:]'):
    x[:] -= y

def array_int_1d_mul(x : 'int[:]', y : 'int[:]'):
    x[:] *= y

def array_int_1d_idiv(x : 'int[:]', y : 'int[:]'):
    x[:] = x // y

#==============================================================================
# 2D ARRAYS OF INT-64 WITH C ORDERING
#==============================================================================

def array_int_2d_C_scalar_add(x : 'int[:,:]', a : 'int'):
    x[:,:] += a

def array_int_2d_C_scalar_sub(x : 'int[:,:]', a : 'int'):
    x[:,:] -= a

def array_int_2d_C_scalar_mul(x : 'int[:,:]', a : 'int'):
    x[:,:] *= a

def array_int_2d_C_scalar_idiv(x : 'int[:,:]', a : 'int'):
    x[:,:] = x // a

def array_int_2d_C_add(x : 'int[:,:]', y : 'int[:,:]'):
    x[:,:] += y

def array_int_2d_C_sub(x : 'int[:,:]', y : 'int[:,:]'):
    x[:,:] -= y

def array_int_2d_C_mul(x : 'int[:,:]', y : 'int[:,:]'):
    x[:,:] *= y

def array_int_2d_C_idiv(x : 'int[:,:]', y : 'int[:,:]'):
    x[:,:] = x // y

def array_int_2d_C_initialization(a : 'int[:,:]'):
    tmp = np.array([[1, 2, 3], [4, 5, 6]])
    a[:,:] = tmp[:,:]

#==============================================================================
# 2D ARRAYS OF INT-64 WITH F ORDERING
#==============================================================================

def array_int_2d_F_scalar_add(x : 'int[:,:](order=F)', a : 'int'):
    x[:,:] += a

def array_int_2d_F_scalar_sub(x : 'int[:,:](order=F)', a : 'int'):
    x[:,:] -= a

def array_int_2d_F_scalar_mul(x : 'int[:,:](order=F)', a : 'int'):
    x[:,:] *= a

def array_int_2d_F_scalar_idiv(x : 'int[:,:](order=F)', a : 'int'):
    x[:,:] = x // a

def array_int_2d_F_add(x : 'int[:,:](order=F)', y : 'int[:,:](order=F)'):
    x[:,:] += y

def array_int_2d_F_sub(x : 'int[:,:](order=F)', y : 'int[:,:](order=F)'):
    x[:,:] -= y

def array_int_2d_F_mul(x : 'int[:,:](order=F)', y : 'int[:,:](order=F)'):
    x[:,:] *= y

def array_int_2d_F_idiv(x : 'int[:,:](order=F)', y : 'int[:,:](order=F)'):
    x[:,:] = x // y

def array_int_2d_F_initialization(a : 'int[:,:](order=F)'):
    tmp = np.array([[1, 2, 3], [4, 5, 6]], dtype='int', order='F')
    a[:,:] = tmp[:,:]


#==============================================================================
# 1D ARRAYS OF REAL
#==============================================================================

def array_float_1d_scalar_add(x : 'float[:]', a : 'float'):
    x[:] += a

def array_float_1d_scalar_sub(x : 'float[:]', a : 'float'):
    x[:] -= a

def array_float_1d_scalar_mul(x : 'float[:]', a : 'float'):
    x[:] *= a

def array_float_1d_scalar_div(x : 'float[:]', a : 'float'):
    x[:] /= a

def array_float_1d_scalar_mod(x : 'float[:]', a : 'float'):
    x[:] %= a

def array_float_1d_scalar_idiv(x : 'float[:]', a : 'float'):
    x[:] = x // a

def array_float_1d_add(x : 'float[:]', y : 'float[:]'):
    x[:] += y

def array_float_1d_sub(x : 'float[:]', y : 'float[:]'):
    x[:] -= y

def array_float_1d_mul(x : 'float[:]', y : 'float[:]'):
    x[:] *= y

def array_float_1d_div(x : 'float[:]', y : 'float[:]'):
    x[:] /= y

def array_float_1d_mod(x : 'float[:]', y : 'float[:]'):
    x[:] %= y

def array_float_1d_idiv(x : 'float[:]', y : 'float[:]'):
    x[:] = x // y

#==============================================================================
# 2D ARRAYS OF REAL WITH C ORDERING
#==============================================================================

def array_float_2d_C_scalar_add(x : 'float[:,:]', a : 'float'):
    x[:,:] += a

def array_float_2d_C_scalar_sub(x : 'float[:,:]', a : 'float'):
    x[:,:] -= a

def array_float_2d_C_scalar_mul(x : 'float[:,:]', a : 'float'):
    x[:,:] *= a

def array_float_2d_C_scalar_div(x : 'float[:,:]', a : 'float'):
    x[:,:] /= a

def array_float_2d_C_scalar_mod(x : 'float[:,:]', a : 'float'):
    x[:,:] %= a

def array_float_2d_C_add(x : 'float[:,:]', y : 'float[:,:]'):
    x[:,:] += y

def array_float_2d_C_sub(x : 'float[:,:]', y : 'float[:,:]'):
    x[:,:] -= y

def array_float_2d_C_mul(x : 'float[:,:]', y : 'float[:,:]'):
    x[:,:] *= y

def array_float_2d_C_div(x : 'float[:,:]', y : 'float[:,:]'):
    x[:,:] /= y

def array_float_2d_C_mod(x : 'float[:,:]', y : 'float[:,:]'):
    x[:,:] %= y

def array_float_2d_C_array_initialization(a : 'float[:,:]'):
    tmp = np.array([[1, 2, 3], [4, 5, 6]], dtype='float')
    a[:,:] = tmp[:,:]

def array_float_3d_C_array_initialization_1(x : 'float[:,:]', y : 'float[:,:]', a : 'float[:,:,:]'):
    tmp      = np.array([x, y], dtype='float')
    a[:,:,:] = tmp[:,:,:]

def array_float_3d_C_array_initialization_2(a : 'float[:,:,:]'):
    x = np.array([[[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
              [[12., 13., 14., 15.], [16., 17., 18., 19.], [20., 21., 22., 23.]]], order='C')
    a[:,:,:] = x[:,:,:]

def array_float_4d_C_array_initialization(x : 'float[:,:,:]', y : 'float[:,:,:]', a : 'float[:,:,:,:]'):
    tmp      = np.array([x, y], dtype='float')
    a[:,:,:,:] = tmp[:,:,:,:]

##==============================================================================
## TEST NESTED ARRAYS INITIALIZATION WITH ORDER C
##==============================================================================

def array_float_nested_C_array_initialization(x : 'float[:,:,:]', y : 'float[:,:]', z : 'float[:,:]', a : 'float[:,:,:,:]'):
    tmp      = np.array((x, (y, z, z), x), dtype='float')
    a[:,:,:,:] = tmp[:,:,:,:]

def array_float_nested_C_array_initialization_2(a : 'float[:,:,:]', e : 'float[:,:]', f : 'float[:]', x : 'float[:,:,:,:]'):
    tmp      = np.array(((e, (f, f)), a, ((f, f), (f, f))), dtype='float')
    x[:,:,:,:] = tmp[:,:,:,:]

def array_float_nested_C_array_initialization_3(a : 'float[:,:,:]', e : 'float[:,:]', x : 'float[:,:,:,:]'):
    tmp      = np.array(((e, ((1., 2., 3.), (1., 2., 3.))),
                       a,
                       (((1., 2., 3.), (1., 2., 3.)),
                        ((1., 2., 3.), (1., 2., 3.)))), dtype='float')
    x[:,:,:,:] = tmp[:,:,:,:]

##==============================================================================
## TEST NESTED ARRAYS INITIALIZATION WITH ORDER F
##==============================================================================

def array_float_nested_F_array_initialization(x : 'float[:,:,:]', y : 'float[:,:]', z : 'float[:,:]', a : 'float[:,:,:,:](order=F)'):
    tmp      = np.array((x, (y, z, z), x), dtype='float', order="F")
    a[:,:,:,:] = tmp[:,:,:,:]

def array_float_nested_F_array_initialization_2(a : 'float[:,:,:]', e : 'float[:,:]', f : 'float[:]', x : 'float[:,:,:,:](order=F)'):
    tmp      = np.array(((e, (f, f)), a, ((f, f), (f, f))), dtype='float', order="F")
    x[:,:,:,:] = tmp[:,:,:,:]

def array_float_nested_F_array_initialization_3(a : 'float[:,:,:]', e : 'float[:,:]', x : 'float[:,:,:,:](order=F)'):
    tmp      = np.array(((e, ((1., 2., 3.), (1., 2., 3.))),
                       a,
                       (((1., 2., 3.), (1., 2., 3.)),
                        ((1., 2., 3.), (1., 2., 3.)))), dtype='float', order="F")
    x[:,:,:,:] = tmp[:,:,:,:]

def array_float_nested_F_array_initialization_mixed(x : 'float[:,:,:](order=F)', y : 'float[:,:](order=F)', z : 'float[:,:](order=F)', a : 'float[:,:,:,:](order=F)'):
    tmp      = np.array((x, (y, z, z), x), dtype='float', order="F")
    a[:,:,:,:] = tmp[:,:,:,:]

##==============================================================================
## TEST ARRAY VIEW STEPS ARRAY INITIALIZATION ORDER C 1D
##==============================================================================

def array_view_steps_C_1D_1(a : 'int[:]'):
    tmp = a[::2]
    b = np.array(tmp)
    return b

def array_view_steps_C_1D_2(a : 'int[:]'):
    tmp = a[1:10:2]
    b = np.array(tmp)
    return b

##==============================================================================
## TEST ARRAY VIEW STEPS ARRAY INITIALIZATION ORDER C 2D
##==============================================================================

def array_view_steps_C_2D_1(a : 'int[:,:]'):
    tmp = a[::2]
    b = np.array(tmp)
    return b

def array_view_steps_C_2D_2(a : 'int[:,:]'):
    tmp = a[1:10:2]
    b = np.array(tmp)
    return b

def array_view_steps_C_2D_3(a : 'int[:,:]'):
    tmp = a[1:10:2, 1::2]
    b = np.array(tmp)
    return b

##==============================================================================
## TEST ARRAY VIEW STEPS ARRAY INITIALIZATION ORDER F 1D
##==============================================================================

def array_view_steps_F_1D_1(a : 'int[:](order=F)'):
    tmp = a[::2]
    b = np.array(tmp, order="F")
    return b

def array_view_steps_F_1D_2(a : 'int[:](order=F)'):
    tmp = a[1:10:2]
    b = np.array(tmp, order="F")
    return b

##==============================================================================
## TEST ARRAY VIEW STEPS ARRAY INITIALIZATION ORDER F 2D
##==============================================================================

def array_view_steps_F_2D_1(a : 'int[:,:](order=F)'):
    tmp = a[::2]
    b = np.array(tmp, order="F")
    return b

def array_view_steps_F_2D_2(a : 'int[:,:](order=F)'):
    tmp = a[1:10:2]
    b = np.array(tmp, order="F")
    return b

def array_view_steps_F_2D_3(a : 'int[:,:](order=F)'):
    tmp = a[1:10:2, 1::2]
    b = np.array(tmp, order="F")
    return b

#==============================================================================
# 2D ARRAYS OF REAL WITH F ORDERING
#==============================================================================

def array_float_2d_F_scalar_add(x : 'float[:,:](order=F)', a : 'float'):
    x[:,:] += a

def array_float_2d_F_scalar_sub(x : 'float[:,:](order=F)', a : 'float'):
    x[:,:] -= a

def array_float_2d_F_scalar_mul(x : 'float[:,:](order=F)', a : 'float'):
    x[:,:] *= a

def array_float_2d_F_scalar_div(x : 'float[:,:](order=F)', a : 'float'):
    x[:,:] /= a

def array_float_2d_F_scalar_mod(x : 'float[:,:](order=F)', a : 'float'):
    x[:,:] %= a

def array_float_2d_F_add(x : 'float[:,:](order=F)', y : 'float[:,:](order=F)'):
    x[:,:] += y

def array_float_2d_F_sub(x : 'float[:,:](order=F)', y : 'float[:,:](order=F)'):
    x[:,:] -= y

def array_float_2d_F_mul(x : 'float[:,:](order=F)', y : 'float[:,:](order=F)'):
    x[:,:] *= y

def array_float_2d_F_div(x : 'float[:,:](order=F)', y : 'float[:,:](order=F)'):
    x[:,:] /= y

def array_float_2d_F_mod(x : 'float[:,:](order=F)', y : 'float[:,:](order=F)'):
    x[:,:] %= y

def array_float_2d_F_array_initialization(a : 'float[:,:](order=F)'):
    tmp = np.array([[1, 2, 3], [4, 5, 6]], dtype='float', order='F')
    a[:,:] = tmp[:,:]

def array_float_3d_F_array_initialization_1(x : 'float[:,:](order=F)', y : 'float[:,:](order=F)', a : 'float[:,:,:](order=F)'):
    tmp      = np.array([x, y], dtype='float', order='F')
    a[:,:,:] = tmp[:,:,:]

def array_float_3d_F_array_initialization_2(a : 'float[:,:,:](order=F)'):
    x = np.array([[[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
                 [[12., 13., 14., 15.], [16., 17., 18., 19.], [20., 21., 22., 23.]]], order='F')
    a[:,:,:] = x[:,:,:]

def array_float_4d_F_array_initialization(x : 'float[:,:,:](order=F)', y : 'float[:,:,:](order=F)', a : 'float[:,:,:,:](order=F)'):
    tmp      = np.array([x, y], dtype='float', order='F')
    a[:,:,:,:] = tmp[:,:,:,:]

def array_float_4d_F_array_initialization_mixed_ordering(x : 'float[:,:](order=F)', a : 'float[:,:,:,:](order=F)'):
    tmp      = np.array(((((0., 1.), (2., 3.)),
                          ((4., 5.), (6., 7.)),
                          ((8., 9.), (10., 11.))),
                          (((12., 13.), (14., 15.)),
                          x,
                          ((20., 21.), (22., 23.)))),
                          dtype='float', order='F')

    a[:,:,:,:] = tmp[:,:,:,:]

#==============================================================================
# COMPLEX EXPRESSIONS IN 3D : TEST CONSTANT AND UNKNOWN SHAPES
#==============================================================================


def array_int32_1d_complex_3d_expr(x : 'int32[:]', y : 'int32[:]'):
    z = np.full(3,5, dtype=np.int32)
    x[:] = (x // y) * x + z

def array_int32_2d_C_complex_3d_expr(x : 'int32[:,:]', y : 'int32[:,:]'):
    z = np.full((2,3),5, dtype=np.int32)
    x[:] = (x // y) * x + z

def array_int32_2d_F_complex_3d_expr(x : 'int32[:,:](order=F)', y : 'int32[:,:](order=F)'):
    z = np.full((2,3),5,order='F', dtype=np.int32)
    x[:] = (x // y) * x + z

def array_float_1d_complex_3d_expr(x : 'float[:]', y : 'float[:]'):
    z = np.full(3,5)
    x[:] = (x // y) * x + z

def array_float_2d_C_complex_3d_expr(x : 'float[:,:]', y : 'float[:,:]'):
    z = np.full((2,3),5)
    x[:] = (x // y) * x + z

def array_float_2d_F_complex_3d_expr(x : 'float[:,:](order=F)', y : 'float[:,:](order=F)'):
    z = np.full((2,3),5,order='F')
    x[:] = (x // y) * x + z

def array_int32_in_bool_out_1d_complex_3d_expr(x : 'int32[:]', y : 'int32[:]', ri : 'bool[:]'):
    z = np.full(3,5, dtype=np.int32)
    ri[:] = (x // y) * x > z

def array_int32_in_bool_out_2d_C_complex_3d_expr(x : 'int32[:,:]', y : 'int32[:,:]', ri : 'bool[:,:]'):
    z = np.full((2,3),5, dtype=np.int32)
    ri[:] = (x // y) * x > z

def array_int32_in_bool_out_2d_F_complex_3d_expr(x : 'int32[:,:](order=F)', y : 'int32[:,:](order=F)', ri : 'bool[:,:](order=F)'):
    z = np.full((2,3),5,order='F', dtype=np.int32)
    ri[:] = (x // y) * x > z

#==============================================================================
# 1D STACK ARRAYS OF REAL
#==============================================================================

@stack_array('a')
def array_float_1d_sum_stack_array():
    a = np.zeros(10)
    s = 0.
    for i in range(10):
        s += a[i]
    return s

@stack_array('a')
def array_float_1d_div_stack_array():
    a = np.ones(10)
    s = 0.
    for i in range(10):
        s += 1.0 / a[i]
    return s

@stack_array('a')
@stack_array('b')
def multiple_stack_array_1():
    a = np.ones(5)
    b = np.array([1, 3, 5, 7, 9])
    s = 0.0
    for i in range(5):
        s += a[i] / b[i]
    return s

@stack_array('a')
@stack_array('b', 'c')
def multiple_stack_array_2():
    a = np.ones(5)
    b = np.array([2, 4, 6, 8, 10])
    c = np.array([1, 3, 5, 7, 9])
    s = 0.0
    for i in range(5):
        s = s + b[i] - a[i] / c[i]
    return s

#==============================================================================
# 2D STACK ARRAYS OF REAL
#==============================================================================

@stack_array('a')
def array_float_2d_sum_stack_array():
    a = np.zeros((10, 10))
    s = 0.
    for i in range(10):
        for j in range(10):
            s += a[i][j]
    return s

@stack_array('a')
def array_float_2d_div_stack_array():
    a = np.full((10, 10), 2)
    s = 1.
    for i in range(10):
        for j in range(10):
            s /= a[i][j]
    return s

@stack_array('a')
@stack_array('b')
def multiple_2d_stack_array_1():
    a = np.ones((2, 5))
    b = np.array([[1, 3, 5, 7, 9], [11, 13, 17, 19, 23]])
    s = 0.0
    j = 0
    for i in range(2):
        for j in range(5):
            s += a[i][j] / b[i][j]
    return s

@stack_array('a')
@stack_array('b', 'c')
def multiple_2d_stack_array_2():
    a = np.ones(5)
    b = np.array([[2, 4, 6, 8, 10], [1, 3, 5, 7, 9]])
    c = np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    s = 0.0
    for i in range(2):
        for j in range(5):
            s = s + b[i][j] - a[j] / c[i][j]
    return s

#==============================================================================
# TEST: Array with ndmin argument
#==============================================================================
def array_ndmin_1(x : T):
    y = np.array(x, ndmin=1)
    return y

def array_ndmin_2(x : T):
    y = np.array(x, ndmin=2)
    return y

def array_ndmin_4(x : T):
    y = np.array(x, ndmin=4)
    return y

def array_ndmin_2_order(x : T):
    y = np.array(x, ndmin=2, order='F')
    return y

#==============================================================================
# TEST: Product and matrix multiplication
#==============================================================================

def array_float_1d_1d_prod(x : 'float[:]', out : 'float[:]'):
    out[:] = np.prod(x)

def array_float_2d_1d_matmul(A : 'float[:,:]', x : 'float[:]', out : 'float[:]'):
    out[:] = np.matmul(A, x)

def array_float_2d_1d_matmul_creation(A : 'float[:,:]', x : 'float[:]'):
    out = np.matmul(A, x)
    return out.sum()

def array_float_2d_1d_matmul_order_F(A : 'float[:,:](order=F)', x : 'float[:]', out : 'float[:]'):
    out[:] = np.matmul(A, x)

def array_float_1d_2d_matmul(x : 'float[:]', A : 'float[:,:]', out : 'float[:]'):
    out[:] = np.matmul(x, A)

def array_float_2d_2d_matmul(A : 'float[:,:]', B : 'float[:,:]', out : 'float[:,:]'):
    out[:,:] = np.matmul(A, B)

def array_float_2d_2d_matmul_F_F(A : 'float[:,:](order=F)', B : 'float[:,:](order=F)', out : 'float[:,:](order=F)'):
    out[:,:] = np.matmul(A, B)

# Mixed order, not supported currently, see #244
def array_float_2d_2d_matmul_mixorder(A : 'float[:,:]', B : 'float[:,:](order=F)', out : 'float[:,:]'):
    out[:,:] = np.matmul(A, B)

def array_float_2d_2d_matmul_operator(A : 'float[:,:]', B : 'float[:,:]', out : 'float[:,:]'):
    out[:,:] = A @ B

def array_float_loopdiff(x : 'float[:]', y : 'float[:]', out : 'float[:]'):
    dxy = x - y
    for k in range(len(x)):
        out[k] = dxy[k]

#==============================================================================
# KEYWORD ARGUMENTS
#==============================================================================

def array_kwargs_full():
    """ full(shape, fill_value, dtype=None, order='C')
    """

    n = 3

    a = np.full((n, n-1), 0.5, 'float', 'C')
    b = np.full((n+1, 2*n), 2.0, order='F')
    c = np.full((1, n), 3)
    d = np.full(2+n, order='F', fill_value=5)
    e = np.full(dtype=int, fill_value=1.0, shape=2*n)

    return np.sum(a) + np.sum(b) + np.sum(c) + np.sum(d) + np.sum(e)

def array_kwargs_ones():
    """ ones(shape, dtype=float, order='C')
    """

    n = 4

    a = np.ones((n, n-1), 'float', 'C')
    b = np.ones((n+1, 2*n), float, order='F')
    c = np.ones((1, n), complex)
    d = np.ones(dtype=int, shape=2+n)

    return np.sum(a) + np.sum(b) + np.sum(c) + np.sum(d)


#==============================================================================
# NEGATIVE INDEXES
#==============================================================================

def constant_negative_index(n : 'int'):
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i

    return a[-1], a[-2]

def almost_negative_index(n : 'int'):
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i
    j = -1

    return a[-j]

@allow_negative_index('a')
def var_negative_index(n : 'int', idx : 'int'):
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i

    return a[idx]

@allow_negative_index('a')
def expr_negative_index(n : 'int', idx_1 : 'int', idx_2 : 'int'):
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i

    return a[idx_1-idx_2]

@allow_negative_index('a')
@allow_negative_index('b')
def test_multiple_negative_index(c : 'int', d : 'int'):
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([1, 2, 3])
    x = a[c]
    y = b[d]

    return x, y

@allow_negative_index('a', 'b')
def test_multiple_negative_index_2(c : 'int', d : 'int'):
    a = np.array([1.2, 2.2, 3.2, 4.2])
    b = np.array([1, 5, 9, 13])

    x = a[c] * d
    y = b[d] * c

    return x, y

@allow_negative_index('a')
@allow_negative_index('b', 'c')
def test_multiple_negative_index_3(d : 'int', e : 'int', f : 'int'):
    a = np.array([1.2, 2.2, 3.2, 4.2])
    b = np.array([1])
    c = np.array([1, 2, 3])

    return a[d], b[e], c[f]

@allow_negative_index('a')
def test_argument_negative_index_1(a : 'int[:]'):
    c = -2
    d = 5
    return a[c], a[d]

@allow_negative_index('a', 'b')
def test_argument_negative_index_2(a : 'int[:]', b : 'int[:]'):
    c = -2
    d = 3
    return a[c], a[d], b[c], b[d]

@allow_negative_index('a', 'b')
def test_c_order_argument_negative_index(a : 'int[:,:]', b : 'int[:,:]'):
    c = -2
    d = 2
    return a[c,0], a[1,d], b[c,1], b[d,0]

@allow_negative_index('a', 'b')
def test_f_order_argument_negative_index(a : 'int[:,:](order=F)', b : 'int[:,:](order=F)'):
    c = -2
    d = 3
    return a[c,0], a[1,d], b[c,1], b[0,d]

#==============================================================================
# SHAPE INITIALISATION
#==============================================================================

def array_random_size():
    a = np.zeros(np.random.randint(23))
    c = np.zeros_like(a)
    return np.shape(a)[0], np.shape(c)[0]

def array_variable_size(n : 'int', m : 'int'):
    s = n
    a = np.zeros(s)
    s = m
    c = np.zeros_like(a)
    return np.shape(a)[0], np.shape(c)[0]

#==============================================================================
# 1D ARRAY SLICING
#==============================================================================

def array_1d_slice_1(a : 'int[:]'):
    b = a[:]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_2(a : 'int[:]'):
    b = a[5:]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_3(a : 'int[:]'):
    b = a[:5]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_4(a : 'int[:]'):
    b = a[5:15]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_5(a : 'int[:]'):
    b = a[:-5]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_6(a : 'int[:]'):
    b = a[-5:]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_7(a : 'int[:]'):
    b = a[-15:-5]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_8(a : 'int[:]'):
    b = a[5:-5]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_9(a : 'int[:]'):
    b = a[-15:15]
    return np.sum(b), b[0], b[-1], len(b)

@allow_negative_index('a')
def array_1d_slice_10(a : 'int[:]'):
    c = -15
    b = a[c:]
    return np.sum(b), b[0], b[-1], len(b)

@allow_negative_index('a')
def array_1d_slice_11(a : 'int[:]'):
    c = -5
    b = a[:c]
    return np.sum(b), b[0], b[-1], len(b)

@allow_negative_index('a')
def array_1d_slice_12(a : 'int[:]'):
    c = -15
    d = -5
    b = a[c:d]
    return np.sum(b), b[0], b[-1], len(b)

#==============================================================================
# 2D ARRAY SLICE ORDER F
#==============================================================================

def array_2d_F_slice_1(a : 'int[:,:](order=F)'):
    b = a[:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_2(a : 'int[:,:](order=F)'):
    b = a[5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_3(a : 'int[:,:](order=F)'):
    b = a[:5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_4(a : 'int[:,:](order=F)'):
    b = a[-15:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_5(a : 'int[:,:](order=F)'):
    b = a[:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_6(a : 'int[:,:](order=F)'):
    b = a[5:15]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_7(a : 'int[:,:](order=F)'):
    b = a[-15:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_8(a : 'int[:,:](order=F)'):
    b = a[::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_9(a : 'int[:,:](order=F)'):
    b = a[5:, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_10(a : 'int[:,:](order=F)'):
    b = a[:5, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_11(a : 'int[:,:](order=F)'):
    b = a[:, 5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_12(a : 'int[:,:](order=F)'):
    b = a[:, :5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_13(a : 'int[:,:](order=F)'):
    b = a[:-5, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_14(a : 'int[:,:](order=F)'):
    b = a[-5:, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_15(a : 'int[:,:](order=F)'):
    b = a[:, -5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_16(a : 'int[:,:](order=F)'):
    b = a[:, :-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_17(a : 'int[:,:](order=F)'):
    b = a[:, 5:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_18(a : 'int[:,:](order=F)'):
    b = a[5:15, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])


def array_2d_F_slice_19(a : 'int[:,:](order=F)'):
    b = a[5:15, -5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_20(a : 'int[:,:](order=F)'):
    b = a[5:15, 5:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_F_slice_21(a : 'int[:,:](order=F)'):
    c = -5
    d = 5
    b = a[d:15, 5:c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_F_slice_22(a : 'int[:,:](order=F)'):
    c = -5
    d = -15
    b = a[d:15, 5:c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_F_slice_23(a : 'int[:,:](order=F)'):
    c = -5
    b = a[:c, :c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

#==============================================================================
# 2D ARRAY SLICE ORDER C
#==============================================================================
def array_2d_C_slice_1(a : 'int[:,:]'):
    b = a[:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_2(a : 'int[:,:]'):
    b = a[5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_3(a : 'int[:,:]'):
    b = a[:5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_4(a : 'int[:,:]'):
    b = a[-15:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_5(a : 'int[:,:]'):
    b = a[:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_6(a : 'int[:,:]'):
    b = a[5:15]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_7(a : 'int[:,:]'):
    b = a[-15:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_8(a : 'int[:,:]'):
    b = a[::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_9(a : 'int[:,:]'):
    b = a[5:, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_10(a : 'int[:,:]'):
    b = a[:5, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_11(a : 'int[:,:]'):
    b = a[:, 5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_12(a : 'int[:,:]'):
    b = a[:, :5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_13(a : 'int[:,:]'):
    b = a[:-5, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_14(a : 'int[:,:]'):
    b = a[-5:, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_15(a : 'int[:,:]'):
    b = a[:, -5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_16(a : 'int[:,:]'):
    b = a[:, :-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_17(a : 'int[:,:]'):
    b = a[:, 5:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_18(a : 'int[:,:]'):
    b = a[5:15, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])


def array_2d_C_slice_19(a : 'int[:,:]'):
    b = a[5:15, -5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_20(a : 'int[:,:]'):
    b = a[5:15, 5:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_C_slice_21(a : 'int[:,:]'):
    c = -5
    d = 5
    b = a[d:15, 5:c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_C_slice_22(a : 'int[:,:]'):
    c = -5
    d = -15
    b = a[d:15, 5:c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_C_slice_23(a : 'int[:,:]'):
    c = -5
    b = a[:c, :c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

#==============================================================================
# 1D ARRAY SLICE STRIDE
#==============================================================================
def array_1d_slice_stride_1(a : 'int[:]'):
    b = a[::1]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_2(a : 'int[:]'):
    b = a[::-1]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_3(a : 'int[:]'):
    b = a[::2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_4(a : 'int[:]'):
    b = a[::-2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_5(a : 'int[:]'):
    b = a[5::2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_6(a : 'int[:]'):
    b = a[5::-2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_7(a : 'int[:]'):
    b = a[:15:2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_8(a : 'int[:]'):
    b = a[:15:-2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_9(a : 'int[:]'):
    b = a[5:15:2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_10(a : 'int[:]'):
    b = a[15:5:-2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_11(a : 'int[:]'):
    b = a[-15:-5:2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_12(a : 'int[:]'):
    b = a[-5:-15:-2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_13(a : 'int[:]'):
    b = a[-5::2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_14(a : 'int[:]'):
    b = a[:-5:-2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_15(a : 'int[:]'):
    b = a[::-5]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_16(a : 'int[:]'):
    b = a[-15::2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_17(a : 'int[:]'):
    b = a[:-15:-2]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_18(a : 'int[:]'):
    b = a[5::-5]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_19(a : 'int[:]'):
    b = a[5:-5:5]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_20(a : 'int[:]'):
    b = a[-5:5:-5]
    return np.sum(b), b[0], b[-1], len(b)

@allow_negative_index('a')
def array_1d_slice_stride_21(a : 'int[:]'):
    c = -5
    b = a[-5:5:c]
    return np.sum(b), b[0], b[-1], len(b)

def array_1d_slice_stride_22(a : 'int[:]'):
    c = 5
    b = a[5:-5:c]
    return np.sum(b), b[0], b[-1], len(b)

@allow_negative_index('a')
def array_1d_slice_stride_23(a : 'int[:]'):
    c = -5
    b = a[::c]
    return np.sum(b), b[0], b[-1], len(b)

#==============================================================================
# 2D ARRAY SLICE STRIDE ORDER F
#==============================================================================

def array_2d_F_slice_stride_1(a : 'int[:,:](order=F)'):
    b = a[::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_2(a : 'int[:,:](order=F)'):
    b = a[::-1]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_3(a : 'int[:,:](order=F)'):
    b = a[::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_4(a : 'int[:,:](order=F)'):
    b = a[::, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_5(a : 'int[:,:](order=F)'):
    b = a[::, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_6(a : 'int[:,:](order=F)'):
    b = a[::2, ::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_7(a : 'int[:,:](order=F)'):
    b = a[::-2, ::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_8(a : 'int[:,:](order=F)'):
    b = a[::2, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_9(a : 'int[:,:](order=F)'):
    b = a[::-2, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_10(a : 'int[:,:](order=F)'):
    b = a[::2, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_11(a : 'int[:,:](order=F)'):
    b = a[::-2, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_12(a : 'int[:,:](order=F)'):
    b = a[5:15:2, 15:5:-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_13(a : 'int[:,:](order=F)'):
    b = a[15:5:-2, 5:15]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_14(a : 'int[:,:](order=F)'):
    b = a[-15:-5:2, -5:-15:-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_15(a : 'int[:,:](order=F)'):
    b = a[-5:-15:-2, -15:-5:2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_16(a : 'int[:,:](order=F)'):
    b = a[::-5, ::5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_17(a : 'int[:,:](order=F)'):
    b = a[::5, ::-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_18(a : 'int[:,:](order=F)'):
    b = a[::-1, ::-1]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_19(a : 'int[:,:](order=F)'):
    b = a[5:15:3, 15:5:-3]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_F_slice_stride_20(a : 'int[:,:](order=F)'):
    b = a[::-10, ::-10]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_F_slice_stride_21(a : 'int[:,:](order=F)'):
    c = -5
    b = a[::c, ::c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_F_slice_stride_22(a : 'int[:,:](order=F)'):
    c = 5
    d = -10
    b = a[::c, ::d]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_F_slice_stride_23(a : 'int[:,:](order=F)'):
    c = 10
    d = -5
    b = a[::d, ::c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

#==============================================================================
# 2D ARRAY SLICE STRIDE ORDER C
#==============================================================================

def array_2d_C_slice_stride_1(a : 'int[:,:]'):
    b = a[::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_2(a : 'int[:,:]'):
    b = a[::-1]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_3(a : 'int[:,:]'):
    b = a[::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_4(a : 'int[:,:]'):
    b = a[::, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_5(a : 'int[:,:]'):
    b = a[::, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_6(a : 'int[:,:]'):
    b = a[::2, ::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_7(a : 'int[:,:]'):
    b = a[::-2, ::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_8(a : 'int[:,:]'):
    b = a[::2, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_9(a : 'int[:,:]'):
    b = a[::-2, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_10(a : 'int[:,:]'):
    b = a[::2, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_11(a : 'int[:,:]'):
    b = a[::-2, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_12(a : 'int[:,:]'):
    b = a[5:15:2, 15:5:-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_13(a : 'int[:,:]'):
    b = a[15:5:-2, 5:15]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_14(a : 'int[:,:]'):
    b = a[-15:-5:2, -5:-15:-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_15(a : 'int[:,:]'):
    b = a[-5:-15:-2, -15:-5:2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_16(a : 'int[:,:]'):
    b = a[::-5, ::5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_17(a : 'int[:,:]'):
    b = a[::5, ::-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_18(a : 'int[:,:]'):
    b = a[::-1, ::-1]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_19(a : 'int[:,:]'):
    b = a[5:15:3, 15:5:-3]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

def array_2d_C_slice_stride_20(a : 'int[:,:]'):
    b = a[::-10, ::-10]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_C_slice_stride_21(a : 'int[:,:]'):
    c = -5
    b = a[::c, ::c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_C_slice_stride_22(a : 'int[:,:]'):
    c = -5
    d = 10
    b = a[::c, ::d]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
def array_2d_C_slice_stride_23(a : 'int[:,:]'):
    c = -10
    d = 5
    b = a[::d, ::c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

#==============================================================================
# Slice assignment
#==============================================================================

def copy_to_slice_issue_1218(n : int):
    x = 2
    arr = np.zeros((3, n))
    arr[0:x, 0:6:2] = np.array([2, 5, 6])
    return arr

def copy_to_slice_1(a : 'float[:]', b : 'float[:]'):
    a[1:-1] = b

def copy_to_slice_2(a : 'float[:,:]', b : 'float[:]'):
    a[:, 1:-1] = b

def copy_to_slice_3(a : 'float[:,:]', b : 'float[:]'):
    a[:, 0] = b

def copy_to_slice_4(a : 'float[:]', b : 'float[:]'):
    a[::2] = b

#==============================================================================
# ARITHMETIC OPERATIONS
#==============================================================================

def arrs_similar_shapes_0():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[2:4]+a[4:6]
    return b

def arrs_similar_shapes_1():
    i = 4
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[2:i]+a[4:i + 2]
    return b

def arrs_different_shapes_0():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[2:4]+a[4:5]
    return b

def arrs_uncertain_shape_1():
    i = 4
    j = 6
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[2:i]+a[4:j]
    return b

def arrs_2d_similar_shapes_0():
    from numpy import shape
    dy = 4
    dx = 2
    arr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x = ((dy**2 * (arr[1:shape(arr)[0]-1, 2:] + arr[1:shape(arr)[0]-1, 0:shape(arr)[1]-2]) +
        dx**2 *(arr[2:, 1:shape(arr)[1]-1] + arr[0:shape(arr)[0]-2, 1:shape(arr)[1]-1])) / (2 * (dx**2 + dy**2)))
    return x

def arrs_2d_different_shapes_0():
    arr1 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    arr2 = np.array([[1, 1, 1]])
    x = arr1 + arr2
    return x

def arrs_1d_negative_index_1():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[:-1]+a[-9:]
    return b

def arrs_1d_negative_index_2():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[1:-1] + a[2:]
    return b

def arrs_1d_int32_index():
    i = np.int32(1)
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[i] + a[i + 2]
    return b

def arrs_1d_int64_index():
    i = np.int64(1)
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[i] + a[i + 2]
    return b

def arrs_1d_negative_index_negative_step():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[-1:1:-2] + a[:2:-2]
    return b

def arrs_1d_negative_step_positive_step():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[1:-1: 3] + a[2::3]
    return b

def arrs_2d_negative_index():
    a = np.array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10],
                  [11, 12, 13, 14, 15, 16, 17, 18, 19,  20],
                  [21, 22, 23, 24, 25, 26, 27, 28, 29,  30],
                  [31, 32, 33, 34, 35, 36, 37, 38, 39,  40],
                  [41, 42, 43, 44, 45, 46, 47, 48, 49,  50],
                  [51, 52, 53, 54, 55, 56, 57, 58, 59,  60],
                  [61, 62, 63, 64, 65, 66, 67, 68, 69,  70],
                  [71, 72, 73, 74, 75, 76, 77, 78, 79,  80],
                  [81, 82, 83, 84, 85, 86, 87, 88, 89,  90],
                  [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]])
    b = a[1:-1, :-1] + a[2:, -9:]
    return b

def arr_tuple_slice_index(a : 'int[:,:]'):
    r = a[(0,1,3),1:]
    return r

#==============================================================================
# NUMPY ARANGE
#==============================================================================

def arr_arange_1():
    a = np.arange(6)
    return np.shape(a)[0], a[0], a[-1]

def arr_arange_2():
    a = np.arange(1, 7)
    return np.shape(a)[0], a[0], a[-1]

def arr_arange_3():
    a = np.arange(0, 10, 0.3)
    return np.shape(a)[0], a[0], a[-1]

def arr_arange_4():
    a = np.arange(1, 28, 3, dtype=float)
    return np.shape(a)[0], a[0], a[-1]

def arr_arange_5():
    a = np.arange(20, 2.2, -2)
    return np.shape(a)[0], a[0], a[-1]

def arr_arange_6():
    a = np.arange(20, 1, -1.1)
    return np.shape(a)[0], a[0], a[-1]

def arr_arange_7(arr : 'int[:,:]'):
    n, m = arr.shape
    for i in range(n):
        arr[i] = np.arange(i, i+m)

#==============================================================================
# NUMPY SUM
#==============================================================================

def arr_bool_sum():
    rows = [True for i in range(100)]
    mat = [rows for j in range(100)]
    a = np.array(mat, dtype=bool)
    return np.sum(a)

def tuple_sum():
    t = (1, 2, 3, 5, 8, 13)
    return np.sum(t)

#==============================================================================
# NUMPY LINSPACE
#==============================================================================

def multiple_np_linspace():
    linspace_index = 5
    x = np.linspace(0, 2, 128)
    y = np.linspace(0, 4, 128)
    z = np.linspace(0, 8, 128)
    return x[0] + y[1] + z[2] + linspace_index

#==============================================================================
# NUMPY ARRAY DATA TYPE CONVERSION
#==============================================================================

def dtype_convert_to_bool(arr : T2D):
    c = np.array(arr, dtype='bool')
    s = np.shape(c)
    return len(s), c[0,0], c[0,1], c[1,0], c[1,1]

def dtype_convert_to_int8(arr : T2D):
    c = np.array(arr, dtype='int8')
    s = np.shape(c)
    return len(s), c[0,0], c[0,1], c[1,0], c[1,1]

def dtype_convert_to_int16(arr : T2D):
    c = np.array(arr, dtype='int16')
    s = np.shape(c)
    return len(s), c[0,0], c[0,1], c[1,0], c[1,1]

def dtype_convert_to_int32(arr : T2D):
    c = np.array(arr, dtype='int32')
    s = np.shape(c)
    return len(s), c[0,0], c[0,1], c[1,0], c[1,1]

def dtype_convert_to_int64(arr : T2D):
    c = np.array(arr, dtype='int64')
    s = np.shape(c)
    return len(s), c[0,0], c[0,1], c[1,0], c[1,1]

def dtype_convert_to_float32(arr : T2D):
    c = np.array(arr, dtype='float32')
    s = np.shape(c)
    return len(s), c[0,0], c[0,1], c[1,0], c[1,1]

def dtype_convert_to_float64(arr : T2D):
    c = np.array(arr, dtype='float64')
    s = np.shape(c)
    return len(s), c[0,0], c[0,1], c[1,0], c[1,1]

def dtype_convert_to_cfloat(arr : T2D):
    c = np.array(arr, dtype='complex64')
    s = np.shape(c)
    return len(s), c[0,0], c[0,1], c[1,0], c[1,1]

def dtype_convert_to_cdouble(arr : T2D):
    c = np.array(arr, dtype='complex128')
    s = np.shape(c)
    return len(s), c[0,0], c[0,1], c[1,0], c[1,1]

def dtype_convert_to_pyint(arr : T2D):
    c = np.array(arr, dtype=int)
    s = np.shape(c)
    return len(s), c[0,0], c[0,1], c[1,0], c[1,1]

def dtype_convert_to_pyfloat(arr : T2D):
    c = np.array(arr, dtype=float)
    s = np.shape(c)
    return len(s), c[0,0], c[0,1], c[1,0], c[1,1]

def src_dest_diff_sizes_dtype_convert_to_bool(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='bool')
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_int8(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='int8')
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_int16(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='int16')
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_int32(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='int32')
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_int64(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='int64')
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_float32(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='float32')
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_float64(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='float64')
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_cfloat(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='complex64')
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_cdouble(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='complex128')
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_pyint(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype=int)
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_pyfloat(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype=float)
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_bool_orderF(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='bool', order="F")
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_int8_orderF(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='int8', order="F")
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_int16_orderF(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='int16', order="F")
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_int32_orderF(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='int32', order="F")
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_int64_orderF(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='int64', order="F")
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_float32_orderF(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='float32', order="F")
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_float64_orderF(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='float64', order="F")
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_cfloat_orderF(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='complex64', order="F")
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_cdouble_orderF(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype='complex128', order="F")
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_pyint_orderF(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype=int, order="F")
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

def src_dest_diff_sizes_dtype_convert_to_pyfloat_orderF(arr1 : T2D, arr2 : T2D, arr3 : T2D):
    a = np.array(arr1)
    b = np.array(arr2)
    c = np.array(arr3)
    d = np.array([a, c, b], dtype=float, order="F")
    s = np.shape(d)
    return s[0], s[1], s[2], d[0,0,0], d[0,0,1], d[1,0,0], d[1,0,1], d[2,0,0], d[2,0,1]

#==============================================================================
# Iteration
#==============================================================================

def iterate_slice(i : int):
    a = np.arange(15)
    res = 0
    for ai in a[:i]:
        res += ai
    return res

def unpack_array(arr : T):
    x, y, z = arr[:]
    return x, y, z

def unpack_array_of_known_size():
    arr = np.array([1,2,3], dtype='float64')
    x, y, z = arr[:]
    return x, y, z

def unpack_array_2D_of_known_size():
    arr = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype='float64')
    x, y, z = arr[:]
    return x.sum(), y.sum(), z.sum()

def assign_slice(a : 'int[:]', n : int):
    a[:n] = [2*i for i in range(n)]

#==============================================================================
# Indexing
#==============================================================================

def multi_layer_index(x : 'int[:]', start : int, stop : int, step : int, idx : int):
    return x[start:stop:step][idx]
