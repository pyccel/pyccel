#coding: utf-8

import numpy as np

from pyccel.epyccel import epyccel

import bsplines


# ................................................
#                 PYCCEL
# ................................................
bplines = epyccel( bsplines )

find_span  = bsplines.find_span
basis_funs = bsplines.basis_funs
# ................................................


#==============================================================================
#@pytest.mark.parametrize( 'lims', ([0,1], [-2,3]) )
#@pytest.mark.parametrize( 'nc', (10, 18, 33) )
#@pytest.mark.parametrize( 'p' , (1,2,3,7,10) )

def test_find_span( lims, nc, p, eps=1e-12 ):

    grid  = np.linspace( *lims, num=nc+1 )
    knots = np.r_[ [grid[0]]*p, grid, [grid[-1]]*p ]

    for i,xi in enumerate( grid ):
        assert find_span( knots, p, x=xi-eps ) == p + max( 0,  i-1 )
        assert find_span( knots, p, x=xi     ) == p + min( i, nc-1 )
        assert find_span( knots, p, x=xi+eps ) == p + min( i, nc-1 )


#==============================================================================
#@pytest.mark.parametrize( 'lims', ([0,1], [-2,3]) )
#@pytest.mark.parametrize( 'nc', (10, 18, 33) )
#@pytest.mark.parametrize( 'p' , (1,2,3,7,10) )

def test_basis_funs( lims, nc, p, tol=1e-14 ):

    grid  = np.linspace( *lims, num=nc+1 )
    knots = np.r_[ [grid[0]]*p, grid, [grid[-1]]*p ]

    basis = np.zeros(p+1, np.float64)

    xx = np.linspace( *lims, num=101 )
    for x in xx:
        span  =  find_span( knots, p, x )
        basis_funs( knots, p, x, span, basis )
        assert np.all( basis >= 0 )
        assert abs( sum( basis ) - 1.0 ) < tol

#############################################
if __name__ == '__main__':

    lims=[0,1] ;  nc=10 ; p=2
    test_find_span( lims, nc, p, eps=1e-12 )
    test_basis_funs( lims, nc, p, tol=1e-14 )
