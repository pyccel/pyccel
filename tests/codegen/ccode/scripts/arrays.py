# pylint: disable=missing-function-docstring, missing-module-docstring
#==============================================================================

def double_loop_on_2d_array_C( z : 'int[:,:](order=C)' ):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j
# ...
def double_loop_on_2d_array_F( z : 'int[:,:](order=F)' ):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j
# ...
def product_loop_on_real_array(z : 'float[:]', out : 'float[:]'):

    from numpy     import shape

    n, = shape( z )

    for i in range(n):
        out[i] = z[i]**2

