from pyccel.decorators import types
import pytest
#==============================================================================

@types( 'int[:,:](order=C)' )
def double_loop_on_2d_array_C( z ):

    from numpy import shape

    s = shape( z )
    m = s[0]
    n = s[1]

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j
# ...
@types( 'int[:,:](order=F)' )
def double_loop_on_2d_array_F( z ):

    from numpy import shape

    s = shape( z )
    m = s[0]
    n = s[1]

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j
# ...
@types( 'real[:], real[:]' )
def product_loop_on_real_array( z, out ):

    from numpy     import shape

    s = shape( z )
    n = s[0]

    for i in range(n):
        out[i] = z[i]**2
