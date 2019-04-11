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

    xs = range(0, 5)
    out = L(xs)
    expected = [0., 1, 4., 9., 16.]
    assert(np.allclose( out, expected ))

    print('DONE.')

#=========================================================
def test_map_zip(**settings):
    l = lambda xs,ys:  map(f2, xs, ys)

    L = _lambdify( l, namespace = {'f2': f2}, **settings )

    xs = range(0, 5)
    ys = range(0, 5)
    out = L(xs, ys)
    expected = [0., 1, 4., 9., 16.]
    assert(np.allclose( out, expected ))

    print('DONE.')

#=========================================================
def test_map_product(**settings):
    l = lambda xs,ys:  xmap(f2, xs, ys)

    L = _lambdify( l, namespace = {'f2': f2}, **settings )
    xs = range(1, 4)
    ys = range(10, 14)
    out = L(xs, ys)
    expected = [10., 11., 12., 13.,
                20., 22., 24., 26.,
                30., 33., 36., 39.]
    assert(np.allclose( out, expected ))

    print('DONE.')

#=========================================================
def test_tmap_product(**settings):
    l = lambda xs,ys:  tmap(f2, xs, ys)

    L = _lambdify( l, namespace = {'f2': f2}, **settings )
    xs = range(1, 4)
    ys = range(10, 14)
    out = L(xs, ys)
    expected = [[10., 11., 12., 13.],
                [20., 22., 24., 26.],
                [30., 33., 36., 39.]]
    assert(np.allclose( out, expected ))

    print('DONE.')

#=========================================================
def test_reduce_function_list(**settings):
    l = lambda xs: reduce(add, map(f1, xs))

    L = _lambdify( l, namespace = {'f1': f1}, **settings )

    xs = range(1, 4)
    out = L(xs)
    expected = 14.0
    assert( out == expected )

    print('DONE.')

#=========================================================
def test_reduce_function_zip(**settings):
    l = lambda xs,ys:  reduce(add, map(f2, xs, ys))

    L = _lambdify( l, namespace = {'f2': f2}, **settings )

    xs = range(0, 5)
    ys = range(0, 5)
    out = L(xs, ys)
    expected = 30.0
    assert( out == expected )

    print('DONE.')

#=========================================================
def test_reduce_function_product(**settings):
    l = lambda xs,ys:  reduce(add, xmap(f2, xs, ys))

    L = _lambdify( l, namespace = {'f2': f2}, **settings )
    xs = range(1, 4)
    ys = range(10, 14)
    out = L(xs, ys)
    expected = 276.0
    assert( out == expected )

    print('DONE.')

##=========================================================
#def test_treduce_function_product(**settings):
#    l = lambda xs,ys:  treduce(f2, product(xs,ys))
#
#    L = _lambdify( l, namespace = {'f2': f2}, **settings )
#    xs = range(1, 4)
#    ys = range(10, 14)
#    out = L(xs, ys)
#    expected = [[10., 11., 12., 13.],
#                [20., 22., 24., 26.],
#                [30., 33., 36., 39.]]
#    assert(np.allclose( out, expected ))


#########################################
if __name__ == '__main__':
    settings = {}
#    settings = {'ast_only' : True}
#    settings = {'printing_only' : True}

    print('======== map    ========')
    test_map_list(**settings)
    test_map_zip(**settings)
    test_map_product(**settings)
    test_tmap_product(**settings)

    print('======== reduce ========')
    test_reduce_function_list(**settings)
    test_reduce_function_zip(**settings)
    test_reduce_function_product(**settings)
###    test_treduce_function_product(**settings)
