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
def test_map_list(**settings):
    l = lambda xs: map(f1, xs)

    L = _lambdify( l, namespace = {'f1': f1}, **settings )

    nx = 5000000
    xs = range(0, nx)

    tb = time.time()
    out = L(xs)
    te = time.time()
    print('[pyccel]  elapsed time = ', te-tb)

#    tb = time.time()
#    out = L(xs, out=out)
#    te = time.time()
#    print('[pyccel]  elapsed time = ', te-tb)

    tb = time.time()
    expected = list(l(xs)) # add list because the result of l is an iterator
    te = time.time()
    print('[python]  elapsed time = ', te-tb)
    assert(np.allclose( out, expected ))

#=========================================================
def test_map_zip(**settings):
    l = lambda xs,ys:  map(f2, xs, ys)

    L = _lambdify( l, namespace = {'f2': f2}, **settings )

    nx = 500000
    xs = range(0, nx)

    ny = 500000
    ys = range(0, ny)

    tb = time.time()
    out = L(xs, ys)
    te = time.time()
    print('[pyccel]  elapsed time = ', te-tb)

    # TODO not working yet
#    tb = time.time()
#    expected = list(l(xs, ys)) # add list because the result of l is an iterator
#    te = time.time()
#    print('[python]  elapsed time = ', te-tb)
#    assert(np.allclose( out, expected ))

#=========================================================
def test_map_product(**settings):
    l = lambda xs,ys:  xmap(f2, xs, ys)

    L = _lambdify( l, namespace = {'f2': f2}, **settings )

    nx = 5000
    xs = range(0, nx)

    ny = 5000
    ys = range(0, ny)

    tb = time.time()
    out = L(xs, ys)
    te = time.time()
    print('[pyccel]  elapsed time = ', te-tb)

    # TODO not working yet
#    tb = time.time()
#    expected = list(l(xs, ys)) # add list because the result of l is an iterator
#    te = time.time()
#    print('[python]  elapsed time = ', te-tb)
#    assert(np.allclose( out, expected ))

#=========================================================
def test_tmap_product(**settings):
    l = lambda xs,ys:  tmap(f2, xs, ys)

    L = _lambdify( l, namespace = {'f2': f2}, **settings )

    nx = 5000
    xs = range(0, nx)

    ny = 5000
    ys = range(0, ny)

    tb = time.time()
    out = L(xs, ys)
    te = time.time()
    print('[pyccel]  elapsed time = ', te-tb)

    # TODO not working yet
#    tb = time.time()
#    expected = list(l(xs, ys)) # add list because the result of l is an iterator
#    te = time.time()
#    print('[python]  elapsed time = ', te-tb)
#    assert(np.allclose( out, expected ))

#=========================================================
def test_reduce_function_list(**settings):
    l = lambda xs: reduce(add, map(f1, xs))

    L = _lambdify( l, namespace = {'f1': f1}, **settings )

    nx = 5000000
    xs = range(1, nx)
    tb = time.time()
    out = L(xs)
    te = time.time()
    print('[pyccel]  elapsed time = ', te-tb)

#=========================================================
def test_reduce_function_zip(**settings):
    l = lambda xs,ys:  reduce(add, map(f2, xs, ys))

    L = _lambdify( l, namespace = {'f2': f2}, **settings )

    nx = 5000
    xs = range(0, nx)

    ny = 5000
    ys = range(0, ny)

    tb = time.time()
    out = L(xs, ys)
    te = time.time()
    print('[pyccel]  elapsed time = ', te-tb)

#=========================================================
def test_reduce_function_product(**settings):
    l = lambda xs,ys:  reduce(add, xmap(f2, xs, ys))

    L = _lambdify( l, namespace = {'f2': f2}, **settings )

    nx = 5000
    xs = range(0, nx)

    ny = 5000
    ys = range(0, ny)

    tb = time.time()
    out = L(xs, ys)
    te = time.time()
    print('[pyccel]  elapsed time = ', te-tb)


#########################################
if __name__ == '__main__':
    settings = {'accelerator': 'openmp'}

    print('======== map    ========')
    test_map_list(**settings)
    test_map_zip(**settings)
    test_map_product(**settings)
    test_tmap_product(**settings)

    print('======== reduce ========')
    test_reduce_function_list(**settings)
    test_reduce_function_zip(**settings)
    test_reduce_function_product(**settings)
##    test_treduce_function_product(**settings)
