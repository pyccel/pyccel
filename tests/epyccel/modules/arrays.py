# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types, stack_array, allow_negative_index

#==============================================================================
# 1D ARRAYS OF INT-32
#==============================================================================

@types( 'int32[:]', 'int32' )
def array_int32_1d_scalar_add( x, a ):
    x[:] += a

@types( 'int32[:]', 'int32' )
def array_int32_1d_scalar_sub( x, a ):
    x[:] -= a

@types( 'int32[:]', 'int32' )
def array_int32_1d_scalar_mul( x, a ):
    x[:] *= a

@types( 'int32[:]', 'int32' )
def array_int32_1d_scalar_div( x, a ):
    x[:] = x / a

@types( 'int32[:]', 'int32' )
def array_int32_1d_scalar_idiv( x, a ):
    x[:] = x // a

@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_add( x, y ):
    x[:] += y

@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_sub( x, y ):
    x[:] -= y

@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_mul( x, y ):
    x[:] *= y

@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_idiv( x, y ):
    x[:] = x // y

#==============================================================================
# 2D ARRAYS OF INT-32 WITH C ORDERING
#==============================================================================

@types( 'int32[:,:]', 'int32' )
def array_int32_2d_C_scalar_add( x, a ):
    x[:,:] += a

@types( 'int32[:,:]', 'int32' )
def array_int32_2d_C_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'int32[:,:]', 'int32' )
def array_int32_2d_C_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'int32[:,:]', 'int32' )
def array_int32_2d_C_scalar_idiv( x, a ):
    x[:,:] = x // a

@types( 'int32[:,:]', 'int32[:,:]' )
def array_int32_2d_C_add( x, y ):
    x[:,:] += y

@types( 'int32[:,:]', 'int32[:,:]' )
def array_int32_2d_C_sub( x, y ):
    x[:,:] -= y

@types( 'int32[:,:]', 'int32[:,:]' )
def array_int32_2d_C_mul( x, y ):
    x[:,:] *= y

@types( 'int32[:,:]', 'int32[:,:]' )
def array_int32_2d_C_idiv( x, y ):
    x[:,:] = x // y

#==============================================================================
# 2D ARRAYS OF INT-32 WITH F ORDERING
#==============================================================================

@types( 'int32[:,:](order=F)', 'int32' )
def array_int32_2d_F_scalar_add( x, a ):
    x[:,:] += a

@types( 'int32[:,:](order=F)', 'int32' )
def array_int32_2d_F_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'int32[:,:](order=F)', 'int32' )
def array_int32_2d_F_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'int32[:,:](order=F)', 'int32' )
def array_int32_2d_F_scalar_idiv( x, a ):
    x[:,:] = x // a

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)' )
def array_int32_2d_F_add( x, y ):
    x[:,:] += y

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)' )
def array_int32_2d_F_sub( x, y ):
    x[:,:] -= y

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)' )
def array_int32_2d_F_mul( x, y ):
    x[:,:] *= y

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)' )
def array_int32_2d_F_idiv( x, y ):
    x[:,:] = x // y


#==============================================================================
# 1D ARRAYS OF INT-64
#==============================================================================

@types( 'int[:]', 'int' )
def array_int_1d_scalar_add( x, a ):
    x[:] += a

@types( 'int[:]', 'int' )
def array_int_1d_scalar_sub( x, a ):
    x[:] -= a

@types( 'int[:]', 'int' )
def array_int_1d_scalar_mul( x, a ):
    x[:] *= a

@types( 'int[:]', 'int' )
def array_int_1d_scalar_idiv( x, a ):
    x[:] = x // a

@types( 'int[:]', 'int[:]' )
def array_int_1d_add( x, y ):
    x[:] += y

@types( 'int[:]', 'int[:]' )
def array_int_1d_sub( x, y ):
    x[:] -= y

@types( 'int[:]', 'int[:]' )
def array_int_1d_mul( x, y ):
    x[:] *= y

@types( 'int[:]', 'int[:]' )
def array_int_1d_idiv( x, y ):
    x[:] = x // y

#==============================================================================
# 2D ARRAYS OF INT-64 WITH C ORDERING
#==============================================================================

@types( 'int[:,:]', 'int' )
def array_int_2d_C_scalar_add( x, a ):
    x[:,:] += a

@types( 'int[:,:]', 'int' )
def array_int_2d_C_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'int[:,:]', 'int' )
def array_int_2d_C_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'int[:,:]', 'int' )
def array_int_2d_C_scalar_idiv( x, a ):
    x[:,:] = x // a

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_add( x, y ):
    x[:,:] += y

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_sub( x, y ):
    x[:,:] -= y

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_mul( x, y ):
    x[:,:] *= y

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_idiv( x, y ):
    x[:,:] = x // y

@types('int[:,:]')
def array_int_2d_C_initialization(a):
    from numpy import array
    tmp = array([[1, 2, 3], [4, 5, 6]])
    a[:,:] = tmp[:,:]

#==============================================================================
# 2D ARRAYS OF INT-64 WITH F ORDERING
#==============================================================================

@types( 'int[:,:](order=F)', 'int' )
def array_int_2d_F_scalar_add( x, a ):
    x[:,:] += a

@types( 'int[:,:](order=F)', 'int' )
def array_int_2d_F_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'int[:,:](order=F)', 'int' )
def array_int_2d_F_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'int[:,:](order=F)', 'int' )
def array_int_2d_F_scalar_idiv( x, a ):
    x[:,:] = x // a

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_add( x, y ):
    x[:,:] += y

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_sub( x, y ):
    x[:,:] -= y

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_mul( x, y ):
    x[:,:] *= y

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_idiv( x, y ):
    x[:,:] = x // y

@types('int[:,:](order=F)')
def array_int_2d_F_initialization(a):
    from numpy import array
    tmp = array([[1, 2, 3], [4, 5, 6]], dtype='int', order='F')
    a[:,:] = tmp[:,:]


#==============================================================================
# 1D ARRAYS OF REAL
#==============================================================================

@types( 'real[:]', 'real' )
def array_real_1d_scalar_add( x, a ):
    x[:] += a

@types( 'real[:]', 'real' )
def array_real_1d_scalar_sub( x, a ):
    x[:] -= a

@types( 'real[:]', 'real' )
def array_real_1d_scalar_mul( x, a ):
    x[:] *= a

@types( 'real[:]', 'real' )
def array_real_1d_scalar_div( x, a ):
    x[:] /= a

@types( 'real[:]', 'real' )
def array_real_1d_scalar_idiv( x, a ):
    x[:] = x // a

@types( 'real[:]', 'real[:]' )
def array_real_1d_add( x, y ):
    x[:] += y

@types( 'real[:]', 'real[:]' )
def array_real_1d_sub( x, y ):
    x[:] -= y

@types( 'real[:]', 'real[:]' )
def array_real_1d_mul( x, y ):
    x[:] *= y

@types( 'real[:]', 'real[:]' )
def array_real_1d_div( x, y ):
    x[:] /= y

@types( 'real[:]', 'real[:]' )
def array_real_1d_idiv( x, y ):
    x[:] = x // y

#==============================================================================
# 2D ARRAYS OF REAL WITH C ORDERING
#==============================================================================

@types( 'real[:,:]', 'real' )
def array_real_2d_C_scalar_add( x, a ):
    x[:,:] += a

@types( 'real[:,:]', 'real' )
def array_real_2d_C_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'real[:,:]', 'real' )
def array_real_2d_C_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'real[:,:]', 'real' )
def array_real_2d_C_scalar_div( x, a ):
    x[:,:] /= a

@types( 'real[:,:]', 'real[:,:]' )
def array_real_2d_C_add( x, y ):
    x[:,:] += y

@types( 'real[:,:]', 'real[:,:]' )
def array_real_2d_C_sub( x, y ):
    x[:,:] -= y

@types( 'real[:,:]', 'real[:,:]' )
def array_real_2d_C_mul( x, y ):
    x[:,:] *= y

@types( 'real[:,:]', 'real[:,:]' )
def array_real_2d_C_div( x, y ):
    x[:,:] /= y

@types('real[:,:]')
def array_real_2d_C_initialization(a):
    from numpy import array
    tmp = array([[1, 2, 3], [4, 5, 6]], dtype='float')
    a[:,:] = tmp[:,:]


#==============================================================================
# 2D ARRAYS OF REAL WITH F ORDERING
#==============================================================================

@types( 'real[:,:](order=F)', 'real' )
def array_real_2d_F_scalar_add( x, a ):
    x[:,:] += a

@types( 'real[:,:](order=F)', 'real' )
def array_real_2d_F_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'real[:,:](order=F)', 'real' )
def array_real_2d_F_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'real[:,:](order=F)', 'real' )
def array_real_2d_F_scalar_div( x, a ):
    x[:,:] /= a

@types( 'real[:,:](order=F)', 'real[:,:](order=F)' )
def array_real_2d_F_add( x, y ):
    x[:,:] += y

@types( 'real[:,:](order=F)', 'real[:,:](order=F)' )
def array_real_2d_F_sub( x, y ):
    x[:,:] -= y

@types( 'real[:,:](order=F)', 'real[:,:](order=F)' )
def array_real_2d_F_mul( x, y ):
    x[:,:] *= y

@types( 'real[:,:](order=F)', 'real[:,:](order=F)' )
def array_real_2d_F_div( x, y ):
    x[:,:] /= y

@types('real[:,:](order=F)')
def array_real_2d_F_initialization(a):
    from numpy import array
    tmp = array([[1, 2, 3], [4, 5, 6]], dtype='float', order='F')
    a[:,:] = tmp[:,:]


#==============================================================================
# COMPLEX EXPRESSIONS IN 3D : TEST CONSTANT AND UNKNOWN SHAPES
#==============================================================================


@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_complex_3d_expr( x, y ):
    from numpy import full, int32
    z = full(3,5, dtype=int32)
    x[:] = (x // y) * x + z

@types( 'int32[:,:]', 'int32[:,:]' )
def array_int32_2d_C_complex_3d_expr( x, y ):
    from numpy import full, int32
    z = full((2,3),5, dtype=int32)
    x[:] = (x // y) * x + z

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)' )
def array_int32_2d_F_complex_3d_expr( x, y ):
    from numpy import full, int32
    z = full((2,3),5,order='F', dtype=int32)
    x[:] = (x // y) * x + z

@types( 'real[:]', 'real[:]' )
def array_real_1d_complex_3d_expr( x, y ):
    from numpy import full
    z = full(3,5)
    x[:] = (x // y) * x + z

@types( 'real[:,:]', 'real[:,:]' )
def array_real_2d_C_complex_3d_expr( x, y ):
    from numpy import full
    z = full((2,3),5)
    x[:] = (x // y) * x + z

@types( 'real[:,:](order=F)', 'real[:,:](order=F)' )
def array_real_2d_F_complex_3d_expr( x, y ):
    from numpy import full
    z = full((2,3),5,order='F')
    x[:] = (x // y) * x + z

@types( 'int32[:]', 'int32[:]', 'int32[:]' )
def array_int32_in_bool_out_1d_complex_3d_expr( x, y, ri ):
    from numpy import full, int32, empty
    z = full(3,5, dtype=int32)
    r = empty(3, dtype=bool)
    r[:] = (x // y) * x > z
    ri[:] = r

@types( 'int32[:,:]', 'int32[:,:]', 'int32[:,:]' )
def array_int32_in_bool_out_2d_C_complex_3d_expr( x, y, ri ):
    from numpy import full, int32
    z = full((2,3),5, dtype=int32)
    r = full((2,3),5,order='F', dtype=bool)
    r[:] = (x // y) * x > z
    ri[:] = r

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)', 'int32[:,:](order=F)' )
def array_int32_in_bool_out_2d_F_complex_3d_expr( x, y, ri ):
    from numpy import full, int32
    z = full((2,3),5,order='F', dtype=int32)
    r = full((2,3),5,order='F', dtype=bool)
    r[:] = (x // y) * x > z
    ri[:] = r

#==============================================================================
# 1D STACK ARRAYS OF REAL
#==============================================================================

@types('double[:]')
@stack_array('a')
def array_real_1d_sum_stack_array():
    from numpy import zeros
    a = zeros(10)
    s = 0.
    for i in range(10):
        s += a[i]
    return s

@types('double[:]')
@stack_array('a')
def array_real_1d_div_stack_array():
    from numpy import ones
    a = ones(10)
    s = 0.
    for i in range(10):
        s += 1.0 / a[i]
    return s

#==============================================================================
# TEST: Product and matrix multiplication
#==============================================================================

@types('real[:], real[:], real[:]')
def array_real_1d_1d_prod(x, out):
    from numpy import prod
    out[:] = prod(x)

@types('real[:,:], real[:], real[:]')
def array_real_2d_1d_matmul(A, x, out):
    from numpy import matmul
    out[:] = matmul(A, x)

@types('real[:,:](order=F), real[:], real[:]')
def array_real_2d_1d_matmul_order_F(A, x, out):
    from numpy import matmul
    out[:] = matmul(A, x)

@types('real[:], real[:,:], real[:]')
def array_real_1d_2d_matmul(x, A, out):
    from numpy import matmul
    out[:] = matmul(x, A)

@types('real[:,:], real[:,:], real[:,:]')
def array_real_2d_2d_matmul(A, B, out):
    from numpy import matmul
    out[:,:] = matmul(A, B)

@types('real[:,:](order=F), real[:,:](order=F), real[:,:](order=F)')
def array_real_2d_2d_matmul_F_F(A, B, out):
    from numpy import matmul
    out[:,:] = matmul(A, B)

# Mixed order, not supported currently, see #244
@types('real[:,:], real[:,:](order=F), real[:,:]')
def array_real_2d_2d_matmul_mixorder(A, B, out):
    from numpy import matmul
    out[:,:] = matmul(A, B)

@types('real[:], real[:], real[:]')
def array_real_loopdiff(x, y, out):
    dxy = x - y
    for k in range(len(x)):
        out[k] = dxy[k]

#==============================================================================
# KEYWORD ARGUMENTS
#==============================================================================

def array_kwargs_full():
    """ full(shape, fill_value, dtype=None, order='C')
    """

    from numpy import sum as np_sum
    from numpy import full

    n = 3

    a = full((n, n-1), 0.5, 'float', 'C')
    b = full((n+1, 2*n), 2.0, order='F')
    c = full((1, n), 3)
    d = full(2+n, order='F', fill_value=5)
    e = full(dtype=int, fill_value=1.0, shape=2*n)

    return np_sum(a) + np_sum(b) + np_sum(c) + np_sum(d) + np_sum(e)

def array_kwargs_ones():
    """ ones(shape, dtype=float, order='C')
    """

    from numpy import sum as np_sum
    from numpy import ones

    n = 4

    a = ones((n, n-1), 'float', 'C')
    b = ones((n+1, 2*n), float, order='F')
    c = ones((1, n), complex)
    d = ones(dtype=int, shape=2+n)

    return np_sum(a) + np_sum(b) + np_sum(c) + np_sum(d)


#==============================================================================
# NEGATIVE INDEXES
#==============================================================================

@types('int')
def constant_negative_index(n):
    import numpy as np
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i

    return a[-1], a[-2]

@types('int')
def almost_negative_index(n):
    import numpy as np
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i
    j = -1

    return a[-j]

@allow_negative_index('a')
@types('int', 'int')
def var_negative_index(n, idx):
    import numpy as np
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i

    return a[idx]

@allow_negative_index('a')
@types('int', 'int', 'int')
def expr_negative_index(n, idx_1, idx_2):
    import numpy as np
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i

    return a[idx_1-idx_2]


#==============================================================================
# SHAPE INITIALISATION
#==============================================================================

def array_random_size():
    import numpy as np
    a = np.zeros(np.random.randint(23))
    c = np.zeros_like(a)
    return np.shape(a)[0], np.shape(c)[0]

@types('int','int')
def array_variable_size(n,m):
    import numpy as np
    s = n
    a = np.zeros(s)
    s = m
    c = np.zeros_like(a)
    return np.shape(a)[0], np.shape(c)[0]
