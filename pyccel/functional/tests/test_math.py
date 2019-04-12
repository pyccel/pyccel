# TODO
#g = lambda xs,ys,z: [[x + y*z for x in xs] for y in ys]
#g = lambda xs,y,z: [x + y*z for x in xs]

import numpy as np
import time

from pyccel.decorators import types, pure
from pyccel.ast.datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool
from pyccel.functional.lambdify import _lambdify
from pyccel.functional import add, mul


#=========================================================
def test_annotate_map_list(**settings):
    L = lambda xs: map(sin, xs)

    L = _lambdify( L, **settings )
    print(L)

    print('DONE.')

#########################################
if __name__ == '__main__':
#    settings = {'ast_only' : True}
    settings = {'printing_only' : True}

    test_annotate_map_list(**settings)
