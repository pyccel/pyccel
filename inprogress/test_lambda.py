# TODO
#g = lambda xs,ys,z: [[x + y*z for x in xs] for y in ys]
#g = lambda xs,y,z: [x + y*z for x in xs]

import numpy as np
import time

from pyccel.decorators import types, pure
from pyccel.ast.datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool
from pyccel.functional.lambdify import _lambdify
from pyccel.functional.ast      import TypeVariable, TypeTuple, TypeList
from pyccel.functional import add, mul

#=========================================================
#         TODO TO BE MOVED TO COMPATIBILITY
#=========================================================
# must have different implementations depending on
#    - type
#    - number of arguments
# same thing must be done for mul
# then in lambdifywe replace 'add' by the right one
# depending on the number of arguments
# then the only thing the user must specify is the dtype
# for this reason, we will have iadd, sadd, dadd, zadd etc
@pure
@types('double', 'double')
def dadd_2(x1,x2):
    r = x1+x2
    return r

@pure
@types('double', 'double', 'double')
def dadd_3(x1,x2,x3):
    r = x1+x2+x3
    return r

@pure
@types('double', 'double', 'double', 'double')
def dadd_4(x1,x2,x3,x4):
    r = x1+x2+x3+x4
    return r
#=========================================================

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
    L = lambda xs: map(f1, xs)

    type_L = _lambdify( L, namespace = {'f1': f1}, **settings )

    assert( isinstance( type_L, TypeList ) )

    parent = type_L.parent
    assert( isinstance( parent.dtype, NativeReal ) )
    assert( parent.rank == 0 )
    assert( parent.precision == 8 )
    assert( not parent.is_stack_array )

    print('DONE.')

#=========================================================
def test_map_zip(**settings):
    L = lambda xs,ys:  map(f2, zip(xs,ys))

    type_L = _lambdify( L, namespace = {'f2': f2}, **settings )

    assert( isinstance( type_L, TypeList ) )

    parent = type_L.parent
    assert( isinstance( parent.dtype, NativeReal ) )
    assert( parent.rank == 0 )
    assert( parent.precision == 8 )
    assert( not parent.is_stack_array )

    print('DONE.')

#=========================================================
def test_map_product(**settings):
    L = lambda xs,ys:  map(f2, product(xs,ys))

    type_L = _lambdify( L, namespace = {'f2': f2}, **settings )

    assert( isinstance( type_L, TypeList ) )

    parent = type_L.parent
    assert( isinstance( parent.dtype, NativeReal ) )
    assert( parent.rank == 0 )
    assert( parent.precision == 8 )
    assert( not parent.is_stack_array )

    print('DONE.')

##=========================================================
## this test will raise an error, which is what we expect
## TODO add error exception and use pytest here
#def test_tmap_zip(**settings):
#    L = lambda xs,ys:  tmap(f2, zip(xs,ys))
#
#    type_L = _lambdify( L, namespace = {'f2': f2}, **settings )
#    print(type_L.view())
#
#    print('DONE.')

#=========================================================
def test_tmap_product(**settings):
    L = lambda xs,ys:  tmap(f2, product(xs,ys))

    type_L = _lambdify( L, namespace = {'f2': f2}, **settings )

    assert( isinstance( type_L, TypeList ) )

    parent = type_L.parent
    assert( isinstance( parent.dtype, NativeReal ) )
    assert( parent.rank == 1 )
    assert( parent.precision == 8 )
    assert( not parent.is_stack_array )

    print('DONE.')

##=========================================================
#def test_reduce_add_product(**settings):
#    L = lambda xs,ys: reduce(dadd_2, product(xs,ys))
#
#    type_L = _lambdify( L, namespace = {'dadd_2': dadd_2}, **settings )
#    print(type_L.view())
#
#    print('DONE.')

#=========================================================
def test_annotate_map_zip(**settings):
    L = lambda xs,ys:  map(f2, zip(xs,ys))

    L = _lambdify( L, namespace = {'f2': f2}, **settings )
    print(L)

    print('DONE.')

#########################################
if __name__ == '__main__':
#    # ... typing
#    # define settings for _lambdify
#    settings = {'type_only' : True}
#
#    test_map_list(**settings)
#    test_map_zip(**settings)
#    test_map_product(**settings)
##    test_tmap_zip(**settings)
#    test_tmap_product(**settings)
#
#    # TODO
##    test_reduce_add_product(**settings)
#    # ...

    # ... annotation
    # define settings for _lambdify
    settings = {'annotation_only' : True}

    test_annotate_map_zip(**settings)
    # ...
