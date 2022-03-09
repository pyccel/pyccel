# pylint: disable=missing-function-docstring, missing-module-docstring/
from numpy.random import randint

from pyccel.epyccel import epyccel

def test_transpose_shape(language):
    def f1(x : 'int[:,:]'):
        from numpy import transpose
        y = transpose(x)
        n, m = y.shape
        return n, m, y[-1,0], y[0,-1]
    def f2(x : 'int[:,:,:]'):
        from numpy import transpose
        y = transpose(x)
        n, m, p = y.shape
        return n, m, p, y[0,-1,0], y[0,0,-1], y[-1,-1,0]

    x1 = randint(50, size=(2,5))
    x2 = randint(50, size=(2,3,7))

    f1_epyc = epyccel(f1, language=language)
    assert f1( x1 ) == f1_epyc( x1 )

    f2_epyc = epyccel(f2, language=language)
    assert f2( x2 ) == f2_epyc( x2 )

def test_transpose_property(language):
    def f1(x : 'int[:,:]'):
        y = x.T
        n, m = y.shape
        return n, m, y[-1,0], y[0,-1]
    def f2(x : 'int[:,:,:]'):
        y = x.T
        n, m, p = y.shape
        return n, m, p, y[0,-1,0], y[0,0,-1], y[-1,-1,0]

    x1 = randint(50, size=(2,5))
    x2 = randint(50, size=(2,3,7))

    f1_epyc = epyccel(f1, language=language)
    assert f1( x1 ) == f1_epyc( x1 )

    f2_epyc = epyccel(f2, language=language)
    assert f2( x2 ) == f2_epyc( x2 )

def test_transpose_in_expression(language):
    def f1(x : 'int[:,:]'):
        from numpy import transpose
        y = transpose(x)+3
        n, m = y.shape
        return n, m, y[-1,0], y[0,-1]
    def f2(x : 'int[:,:,:]'):
        y = x.T*3
        n, m, p = y.shape
        return n, m, p, y[0,-1,0], y[0,0,-1], y[-1,-1,0]

    x1 = randint(50, size=(2,5))
    x2 = randint(50, size=(2,3,7))

    f1_epyc = epyccel(f1, language=language)
    assert f1( x1 ) == f1_epyc( x1 )

    f2_epyc = epyccel(f2, language=language)
    assert f2( x2 ) == f2_epyc( x2 )

def test_mixed_order(language):
    def f1(x : 'int[:,:]'):
        from numpy import transpose, ones
        n, m = x.shape
        y = ones((m,n), order='F')
        z = x+transpose(y)
        n, m = z.shape
        return n, m, z[-1,0], z[0,-1]
    def f2(x : 'int[:,:]'):
        from numpy import transpose, ones
        n, m = x.shape
        y = ones((m,n), order='F')
        z = x.transpose()+y
        n, m = z.shape
        return n, m, z[-1,0], z[0,-1]
    def f3(x : 'int[:,:,:]'):
        from numpy import transpose, ones
        n, m, p = x.shape
        y = ones((p,m,n))
        z = transpose(x)+y
        n, m, p = z.shape
        return n, m, p, z[0,-1,0], z[0,0,-1], z[-1,-1,0]

    x1 = randint(50, size=(2,5))
    x2 = randint(50, size=(2,3,7))

    f1_epyc = epyccel(f1, language=language)
    assert f1( x1 ) == f1_epyc( x1 )

    f2_epyc = epyccel(f2, language=language)
    assert f2( x1 ) == f2_epyc( x1 )

    f3_epyc = epyccel(f3, language=language)
    assert f3( x2 ) == f3_epyc( x2 )

def test_transpose_pointer(language):
    def f1(x : 'int[:,:]'):
        from numpy import transpose
        y = transpose(x)
        x[0,-1] += 22
        n, m = y.shape
        return n, m, y[-1,0], y[0,-1]
    def f2(x : 'int[:,:,:]'):
        y = x.T
        x[0,-1,0] += 11
        n, m, p = y.shape
        return n, m, p, y[0,-1,0], y[0,0,-1], y[-1,-1,0]

    x1 = randint(50, size=(2,5))
    x1_copy = x1.copy()
    x2 = randint(50, size=(2,3,7))
    x2_copy = x2.copy()

    f1_epyc = epyccel(f1, language=language)
    assert f1( x1 ) == f1_epyc( x1_copy )

    f2_epyc = epyccel(f2, language=language)
    assert f2( x2 ) == f2_epyc( x2_copy )

def test_transpose_of_expression(language):
    def f1(x : 'int[:,:]'):
        from numpy import transpose
        y = transpose(x*2)+3
        n, m = y.shape
        return n, m, y[-1,0], y[0,-1]
    def f2(x : 'int[:,:,:]'):
        y = (x*2).T*3
        n, m, p = y.shape
        return n, m, p, y[0,-1,0], y[0,0,-1], y[-1,-1,0]

    x1 = randint(50, size=(2,5))
    x2 = randint(50, size=(2,3,7))

    f1_epyc = epyccel(f1, language=language)
    assert f1( x1 ) == f1_epyc( x1 )

    f2_epyc = epyccel(f2, language=language)
    assert f2( x2 ) == f2_epyc( x2 )

def test_force_transpose(language):
    def f1(x : 'int[:,:]'):
        from numpy import transpose, empty
        n,m = x.shape
        y = empty((m,n))
        y[:,:] = transpose(x)
        n, m = y.shape
        return n, m, y[-1,0], y[0,-1]
    def f2(x : 'int[:,:,:]'):
        from numpy import empty
        n,m,p = x.shape
        y = empty((p,m,n))
        y[:,:,:] = x.transpose()
        n, m, p = y.shape
        return n, m, p, y[0,-1,0], y[0,0,-1], y[-1,-1,0]

    x1 = randint(50, size=(2,5))
    x2 = randint(50, size=(2,3,7))

    f1_epyc = epyccel(f1, language=language)
    assert f1( x1 ) == f1_epyc( x1 )

    f2_epyc = epyccel(f2, language=language)
    assert f2( x2 ) == f2_epyc( x2 )

def test_transpose_to_inner_indexes(language):
    def f1(x : 'int[:,:]'):
        from numpy import transpose, empty
        n,m = x.shape
        y = empty((1,m,n,1))
        y[0,:,:,0] = transpose(x)
        return y[0,0,0,0], y[0,1,0,0], y[0,0,1,0], y[0,-1,0,0], y[0,0,-1,0]

    x1 = randint(50, size=(2,5))

    f1_epyc = epyccel(f1, language=language)
    assert f1( x1 ) == f1_epyc( x1 )
