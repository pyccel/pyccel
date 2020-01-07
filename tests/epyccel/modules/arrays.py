from pyccel.decorators import types, stack_array

#==============================================================================
# 1D ARRAYS OF INT
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
# 2D ARRAYS OF INT WITH C ORDERING
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

#==============================================================================
# 2D ARRAYS OF INT WITH F ORDERING
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
