from pyccel.decorators import types

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
