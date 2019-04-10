# TODO
#g = lambda xs,ys,z: [[x + y*z for x in xs] for y in ys]
#g = lambda xs,y,z: [x + y*z for x in xs]

import numpy as np
import time

from pyccel.decorators import types, pure
from pyccel.ast.datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool
from pyccel.functional.lambdify import _lambdify
from pyccel.functional import TypeVariable, TypeTuple, TypeList
from pyccel.functional import add, mul

# TODO for compatibility
pmap = map

#=========================================================
@pure
@types('double')
def f1(x):
    r = x**2
    return r

@pure
@types('double', 'double')
def f2(x,y):
    r = x*y
    return r

#=========================================================
def test_pmap_list(**settings):
    l = lambda xs: pmap(f1, xs)

    # TODO
    settings['accelerator'] = 'openmp'
#    settings['verbose'] = True

    L = _lambdify( l, namespace = {'f1': f1}, **settings )

    nx = 10000000
    xs = range(0, nx)

#    rs = np.zeros(nx, np.float64)
#    tb = time.time()
#    L(xs, rs)
#    te = time.time()
#    print('[pyccel]  elapsed time = ', te-tb)

    tb = time.time()
    out = L(xs)
    te = time.time()
    print('[pyccel]  elapsed time = ', te-tb)

#    tb = time.time()
#    expected = list(l(xs)) # add list because the result of l is an iterator
#    te = time.time()
#    print('[python]  elapsed time = ', te-tb)
#    assert(np.allclose( out, expected ))

#=========================================================
def test_pmap_zip(**settings):
    L = lambda xs,ys:  pmap(f2, zip(xs,ys))

    L = _lambdify( L, namespace = {'f2': f2}, **settings )

    xs = range(0, 5)
    ys = range(0, 5)
    out = L(xs, ys)
    expected = [0., 1, 4., 9., 16.]
    assert(np.allclose( out, expected ))

#=========================================================
def test_pmap_product(**settings):
    L = lambda xs,ys:  pmap(f2, product(xs,ys))

    L = _lambdify( L, namespace = {'f2': f2}, **settings )
    xs = range(1, 4)
    ys = range(10, 14)
    out = L(xs, ys)
    expected = [10., 11., 12., 13.,
                20., 22., 24., 26.,
                30., 33., 36., 39.]
    assert(np.allclose( out, expected ))

#=========================================================
def test_tpmap_product(**settings):
    L = lambda xs,ys:  tpmap(f2, product(xs,ys))

    L = _lambdify( L, namespace = {'f2': f2}, **settings )
    xs = range(1, 4)
    ys = range(10, 14)
    out = L(xs, ys)
    expected = [[10., 11., 12., 13.],
                [20., 22., 24., 26.],
                [30., 33., 36., 39.]]
    assert(np.allclose( out, expected ))

##=========================================================
#def test_reduce_add_product(**settings):
#    L = lambda xs,ys: reduce(dadd_2, product(xs,ys))
#
#    L = _lambdify( L, namespace = {'dadd_2': dadd_2}, **settings )
#    print(type_L.view())
#
#    print('DONE.')


#########################################
if __name__ == '__main__':
    settings = {}
#    settings = {'ast_only' : True}
#    settings = {'printing_only' : True}

    test_pmap_list(**settings)
#    test_pmap_zip(**settings)
#    test_pmap_product(**settings)
#    test_tpmap_product(**settings)

#    # TODO
##    test_reduce_add_product(**settings)
