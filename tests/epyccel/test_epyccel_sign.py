# pylint: disable=missing-function-docstring, missing-module-docstring/

from pyccel.decorators import types
from pyccel.epyccel import epyccel

import numpy as np

def test_sign_complex(language):

    def f_pos():
        import numpy as np
        b = np.sign(complex(1+2j))
        return b

    def f_neg():
        import numpy as np
        b = np.sign(complex(-1-2j))
        return b

    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_complex64(language):

    def f_pos():
        import numpy as np
        b = np.sign(np.complex64(64+64j))
        return b

    def f_neg():
        import numpy as np
        b = np.sign(np.complex64(-64-64j))
        return b

    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_complex128(language):

    def f_pos():
        import numpy as np
        b = np.sign(np.complex128(128+128j))
        return b

    def f_neg():
        import numpy as np
        b = np.sign(np.complex128(-128-128j))
        return b

    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_int16(language):

    def f_pos():
        import numpy as np
        b = np.sign(np.int16(16))
        return b

    def f_neg():
        import numpy as np
        b = np.sign(np.int16(-16))
        return b

    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_int32(language):

    def f_pos():
        import numpy as np
        b = np.sign(np.int32(32))
        return b

    def f_neg():
        import numpy as np
        b = np.sign(np.int32(-32))
        return b

    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_int64(language):

    def f_pos():
        import numpy as np
        b = np.sign(np.int64(64))
        return b

    def f_neg():
        import numpy as np
        b = np.sign(np.int64(-64))
        return b

    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_float32(language):

    def f_pos():
        import numpy as np
        b = np.sign(np.float(32.32))
        return b

    def f_neg():
        import numpy as np
        b = np.sign(np.float(-32.32))
        return b

    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_float64(language):

    def f_pos():
        import numpy as np
        b = np.sign(np.float64(64.64))
        return b

    def f_neg():
        import numpy as np
        b = np.sign(np.float64(-64.64))
        return b

    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_literal_complex(language):

    def f_pos():
        import numpy as np
        b = np.sign(1+2j)
        return b

    def f_neg():
        import numpy as np
        b = np.sign(-1-2j)
        return b

    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_literal_int(language):
    def f_pos():
        import numpy as np
        b = np.sign(42)
        return b

    def f_neg():
        import numpy as np
        b = np.sign(-42)
        return b

    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_literal_float(language):
    def f_pos():
        import numpy as np
        b = np.sign(42.42)
        return b

    def f_neg():
        import numpy as np
        b = np.sign(-42.42)
        return b

    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

# Tests on arrays

def test_sign_arr_int(language):

    @types('int64[:]')
    def f_1d(x):
        import numpy as np
        x[:] = np.sign(x)

    @types('int64[:,:]')
    def f_2d(x):
        import numpy as np
        x[:,:] = np.sign(x)

    f_1d_epyc = epyccel(f_1d, language = language)
    f_2d_epyc = epyccel(f_2d, language = language)

    x1_1d = np.array([2, 0, 2, -13, 37, 42], dtype=np.int64)
    x1_2d = np.array([[2, 0], [2, -13], [37, 42]], dtype=np.int64)
    x2_1d = np.copy(x1_1d)
    x2_2d = np.copy(x1_2d)

    f_1d(x1_1d)
    f_2d(x1_2d)
    f_1d_epyc(x2_1d)
    f_2d_epyc(x2_2d)

    assert np.array_equal(x1_1d, x2_1d)
    assert np.array_equal(x1_2d, x2_2d)

def test_sign_arr_float(language):

    @types('float64[:]')
    def f_1d(x):
        import numpy as np
        x[:] = np.sign(x)

    @types('float64[:,:]')
    def f_2d(x):
        import numpy as np
        x[:,:] = np.sign(x)

    f_1d_epyc = epyccel(f_1d, language = language)
    f_2d_epyc = epyccel(f_2d, language = language)

    x1_1d = np.array([0., 1., 2., -1.3, -3.7, -0.], dtype=np.float64)
    x1_2d = np.array([[0., 1.], [2., -1.3], [-3.7, -0.]], dtype=np.float64)
    x2_1d = np.copy(x1_1d)
    x2_2d = np.copy(x1_2d)

    f_1d(x1_1d)
    f_2d(x1_2d)
    f_1d_epyc(x2_1d)
    f_2d_epyc(x2_2d)

    assert np.array_equal(x1_1d, x2_1d)
    assert np.array_equal(x1_2d, x2_2d)

def test_sign_arr_complex(language):

    @types('complex64[:]')
    def f_1d(x):
        import numpy as np
        x[:] = np.sign(x)

    @types('complex64[:,:]')
    def f_2d(x):
        import numpy as np
        x[:,:] = np.sign(x)

    f_1d_epyc = epyccel(f_1d, language = language)
    f_2d_epyc = epyccel(f_2d, language = language)

    x1_1d = np.array([0.+.1j, -2.-1.3j, -3.7j], dtype=np.complex64)
    x1_2d = np.array([[0.+1.], [-2.-1.3j], [-3.7j]], dtype=np.complex64)
    x2_1d = np.copy(x1_1d)
    x2_2d = np.copy(x1_2d)

    f_1d(x1_1d)
    f_2d(x1_2d)
    f_1d_epyc(x2_1d)
    f_2d_epyc(x2_2d)

    assert np.array_equal(x1_1d, x2_1d)
    assert np.array_equal(x1_2d, x2_2d)
