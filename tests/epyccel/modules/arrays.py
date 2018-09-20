from pyccel.decorators import types

#==============================================================================
# 1D ARRAYS OF INT
#==============================================================================

@types( 'int[:]', int )
def array_int_1d_scalar_add( x, a ):
    return x + a

@types( 'int[:]', int )
def array_int_1d_scalar_sub( x, a ):
    return x - a

@types( 'int[:]', int )
def array_int_1d_scalar_mul( x, a ):
    return a * x

@types( 'int[:]', int )
def array_int_1d_scalar_div( x, a ):
    return x / a

@types( 'int[:]', int )
def array_int_1d_scalar_idiv( x, a ):
    return x // a

@types( 'int[:]', int )
def array_int_1d_scalar_mod( x, a ):
    return x % a

@types( 'int[:]', 'int[:]' )
def array_int_1d_add( x, y ):
    return x + y

@types( 'int[:]', 'int[:]' )
def array_int_1d_sub( x, y ):
    return y - x

@types( 'int[:]', 'int[:]' )
def array_int_1d_mul( x, y ):
    return x * y

@types( 'int[:]', 'int[:]' )
def array_int_1d_div( x, y ):
    return x / y

@types( 'int[:]', 'int[:]' )
def array_int_1d_idiv( x, y ):
    return x // y

@types( 'int[:]', 'int[:]' )
def array_int_1d_mod( x, y ):
    return x % y

#==============================================================================
# 2D ARRAYS OF INT, with C ordering (hence f2py creates transposed temporary)
#==============================================================================

@types( 'int[:,:]', int )
def array_int_2d_C_scalar_add( x, a ):
    return x + a

@types( 'int[:,:]', int )
def array_int_2d_C_scalar_sub( x, a ):
    return x - a

@types( 'int[:,:]', int )
def array_int_2d_C_scalar_mul( x, a ):
    return a * x

@types( 'int[:,:]', int )
def array_int_2d_C_scalar_div( x, a ):
    return x / a

@types( 'int[:,:]', int )
def array_int_2d_C_scalar_idiv( x, a ):
    return x // a

@types( 'int[:,:]', int )
def array_int_2d_C_scalar_mod( x, a ):
    return x % a

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_add( x, y ):
    return x + y

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_sub( x, y ):
    return x - y

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_mul( x, y ):
    return x * y

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_div( x, y ):
    return x / y

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_idiv( x, y ):
    return x // y

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_mod( x, y ):
    return x % y

#==============================================================================
# 2D ARRAYS OF INT, with Fortran ordering
#==============================================================================

@types( 'int[:,:](order=F)', int )
def array_int_2d_F_scalar_add( x, a ):
    return x + a

@types( 'int[:,:](order=F)', int )
def array_int_2d_F_scalar_sub( x, a ):
    return x - a

@types( 'int[:,:](order=F)', int )
def array_int_2d_F_scalar_mul( x, a ):
    return a * x

@types( 'int[:,:](order=F)', int )
def array_int_2d_F_scalar_div( x, a ):
    return x / a

@types( 'int[:,:](order=F)', int )
def array_int_2d_F_scalar_idiv( x, a ):
    return x // a

@types( 'int[:,:](order=F)', int )
def array_int_2d_F_scalar_mod( x, a ):
    return x % a

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_add( x, y ):
    return x + y

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_sub( x, y ):
    return x - y

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_mul( x, y ):
    return x * y

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_div( x, y ):
    return x / y

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_idiv( x, y ):
    return x // y

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_mod( x, y ):
    return x % y
