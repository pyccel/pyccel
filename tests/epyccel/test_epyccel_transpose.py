# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy import empty, array_equal
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

    def f1(x : 'int[:,:]', y : 'int[:,:,:,:]'):
        y[0,:,:,0] = x.T

    def f2(x : 'int[:,:]', y : 'int[:,:,:,:,:]'):
        y[0,:,0,:,0] = x.T

    def f3(x : 'int[:,:,:]', y : 'int[:,:,:,:,:]'):
        y[0,:,:,:,0] = x.T

    x1 = randint(50, size=(2,5))
    x2 = randint(50, size=(2,5,3))

    y1_pyt = empty((1,5,2,1), dtype=int)
    y2_pyt = empty((1,5,1,2,1), dtype=int)
    y3_pyt = empty((1,3,5,2,1), dtype=int)

    y1_pyc = empty((1,5,2,1), dtype=int)
    y2_pyc = empty((1,5,1,2,1), dtype=int)
    y3_pyc = empty((1,3,5,2,1), dtype=int)

    f1_epyc = epyccel(f1, language=language)
    f1( x1, y1_pyt )
    f1_epyc( x1, y1_pyc )
    assert array_equal(y1_pyt, y1_pyc)

    f2_epyc = epyccel(f2, language=language)
    f2( x1, y2_pyt )
    f2_epyc( x1, y2_pyc )
    assert array_equal(y2_pyt, y2_pyc)

    f3_epyc = epyccel(f3, language=language)
    f3( x2, y3_pyt )
    f3_epyc( x2, y3_pyc )
    assert array_equal(y3_pyt, y3_pyc)
