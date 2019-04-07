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

# define settings for _lambdify
settings = {'type_only' :True}

#=========================================================
def test_map_list():
    L = lambda xs: map(sin, xs)

    type_L = _lambdify( L, **settings )

    assert( isinstance( type_L, TypeList ) )

    parent = type_L.parent
    assert( isinstance( parent.dtype, NativeReal ) )
    assert( parent.rank == 0 )
    assert( parent.precision == 8 )
    assert( not parent.is_stack_array )

    print('DONE.')

#########################################
if __name__ == '__main__':
    test_map_list()
