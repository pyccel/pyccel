# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types
#==============================================================================

@types( 'int[:,:](order=C)' )
def double_loop_on_2d_array_C( z ):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j
# ...
@types( 'int[:,:](order=F)' )
def double_loop_on_2d_array_F( z ):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j
# ...
@types( 'real[:], real[:]' )
def product_loop_on_real_array( z, out ):

    from numpy     import shape

    n, = shape( z )

    for i in range(n):
        out[i] = z[i]**2

