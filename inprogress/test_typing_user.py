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
    assert( isinstance( parent, TypeList ) )
    parent = parent.parent

    assert( isinstance( parent.dtype, NativeReal ) )
    assert( parent.rank == 0 )
    assert( parent.precision == 8 )
    assert( not parent.is_stack_array )

    print('DONE.')

#=========================================================
def test_reduce_function_list(**settings):
    L = lambda xs: reduce(f2, xs)

    type_L = _lambdify( L, namespace = {'f2': f2}, **settings )

    assert( isinstance( type_L, TypeVariable ) )

    assert( isinstance( type_L.dtype, NativeReal ) )
    assert( type_L.rank == 0 )
    assert( type_L.precision == 8 )
    assert( not type_L.is_stack_array )

#########################################
if __name__ == '__main__':
    settings = {'semantic_only' : True}

#    test_map_list(**settings)
#    test_map_zip(**settings)
#    test_map_product(**settings)
#    test_tmap_product(**settings)

    # TODO
    test_reduce_function_list(**settings)
