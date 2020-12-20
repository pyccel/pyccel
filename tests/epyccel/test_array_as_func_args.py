import pytest
import numpy as np

from pyccel.epyccel import epyccel
from pyccel.decorators import types

def test_array_int32_1d_scalar_add(language):
    @types( 'int32[:]', 'int32', 'int')
    def array_int32_1d_scalar_add( x, a, x_len ):
        for i in range(x_len):
            x[i] += a

    f1 = array_int32_1d_scalar_add
    f2 = epyccel(f1, language=language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a, len(x1))
    f2(x2, a, len(x2))

    assert np.array_equal( x1, x2 )
