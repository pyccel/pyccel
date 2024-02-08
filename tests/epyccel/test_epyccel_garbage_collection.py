# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import sys
import numpy as np
from pyccel.epyccel import epyccel

def test_return_pointer(language):
    def return_pointer(x : 'float[:]'):
        return x

    f = epyccel(return_pointer, language=language)

    x_pyc = np.ones(10)
    x_pyt = np.ones(10)

    start_ref_count_pyc = sys.getrefcount(x_pyc)

    y_pyc = f(x_pyc)
    y_pyt = return_pointer(x_pyt)

    ref_count_x_pyc = sys.getrefcount(x_pyc)
    ref_count_x_pyt = sys.getrefcount(x_pyt)
    ref_count_y_pyc = sys.getrefcount(y_pyc)

    assert ref_count_x_pyc == ref_count_x_pyt
    assert ref_count_x_pyc == start_ref_count_pyc+1

    if language != 'python':
        assert ref_count_y_pyc == start_ref_count_pyc
        assert y_pyc.base is x_pyc

    del y_pyc
    del y_pyt

    ref_count_pyc = sys.getrefcount(x_pyc)
    ref_count_pyt = sys.getrefcount(x_pyt)

    assert ref_count_pyc == ref_count_pyt
    assert ref_count_pyc == start_ref_count_pyc

def test_return_multiple_pointers(language):
    def return_pointer(x : 'float[:]', y : 'float[:]'):
        return x, y

    f = epyccel(return_pointer, language=language)

    x_pyc = np.ones(10)
    x_pyt = np.ones(10)

    y_pyc = np.zeros(10)
    y_pyt = np.zeros(10)

    start_ref_count_x_pyc = sys.getrefcount(x_pyc)
    start_ref_count_y_pyc = sys.getrefcount(y_pyc)

    a_pyc, b_pyc = f(x_pyc, y_pyc)
    a_pyt, b_pyt = f(x_pyt, y_pyt)

    ref_count_x_pyc = sys.getrefcount(x_pyc)
    ref_count_x_pyt = sys.getrefcount(x_pyt)
    ref_count_y_pyc = sys.getrefcount(y_pyc)
    ref_count_y_pyt = sys.getrefcount(y_pyt)
    ref_count_a_pyc = sys.getrefcount(a_pyc)
    ref_count_a_pyt = sys.getrefcount(a_pyt)
    ref_count_b_pyc = sys.getrefcount(b_pyc)
    ref_count_b_pyt = sys.getrefcount(b_pyt)

    assert ref_count_x_pyc == ref_count_x_pyt
    assert ref_count_x_pyc == start_ref_count_x_pyc+1
    assert ref_count_y_pyc == ref_count_y_pyt
    assert ref_count_y_pyc == start_ref_count_y_pyc+1
    assert ref_count_a_pyc == ref_count_a_pyt
    assert ref_count_b_pyc == ref_count_b_pyt

    if language != 'python':
        assert a_pyc.base is x_pyc
        assert b_pyc.base is y_pyc

    del a_pyc
    del a_pyt
    del b_pyc
    del b_pyt

    ref_count_x_pyc = sys.getrefcount(x_pyc)
    ref_count_x_pyt = sys.getrefcount(x_pyt)
    ref_count_y_pyc = sys.getrefcount(y_pyc)
    ref_count_y_pyt = sys.getrefcount(y_pyt)

    assert ref_count_x_pyc == ref_count_x_pyt
    assert ref_count_x_pyc == start_ref_count_x_pyc
    assert ref_count_y_pyc == ref_count_y_pyt
    assert ref_count_y_pyc == start_ref_count_y_pyc

def test_return_slice(language):
    def return_pointer(x : 'float[:]'):
        return x[::2]

    f = epyccel(return_pointer, language=language)

    x_pyc = np.ones(10)
    x_pyt = np.ones(10)

    start_ref_count_pyc = sys.getrefcount(x_pyc)

    y_pyc = f(x_pyc)
    y_pyt = return_pointer(x_pyt)

    ref_count_x_pyc = sys.getrefcount(x_pyc)
    ref_count_x_pyt = sys.getrefcount(x_pyt)
    ref_count_y_pyc = sys.getrefcount(y_pyc)
    ref_count_y_pyt = sys.getrefcount(y_pyt)

    assert ref_count_x_pyc == ref_count_x_pyt
    assert ref_count_x_pyc == start_ref_count_pyc+1
    assert ref_count_y_pyc == ref_count_y_pyt

    assert y_pyc.base is x_pyc
    assert y_pyt.base is x_pyt

    del y_pyc
    del y_pyt

    ref_count_pyc = sys.getrefcount(x_pyc)
    ref_count_pyt = sys.getrefcount(x_pyt)

    assert ref_count_pyc == ref_count_pyt
    assert ref_count_pyc == start_ref_count_pyc

def test_return_class_pointer(language):
    import classes.return_class_pointer as mod_pyt

    mod_pyc = epyccel(mod_pyt, language=language)

    a_pyc = mod_pyc.A(4)
    a_pyt = mod_pyt.A(4)

    start_ref_count_pyc = sys.getrefcount(a_pyc)

    a2_pyc = mod_pyc.examine_A(a_pyc)
    a2_pyt = mod_pyt.examine_A(a_pyt)

    ref_count_a_pyc = sys.getrefcount(a_pyc)
    ref_count_a2_pyc = sys.getrefcount(a2_pyc)

    if language != 'python':
        # a now referenced by a2
        assert ref_count_a_pyc == start_ref_count_pyc+1
        assert ref_count_a2_pyc == start_ref_count_pyc
    else:
        assert ref_count_a_pyc > start_ref_count_pyc

    del a2_pyc
    del a2_pyt

    ref_count_pyc = sys.getrefcount(a_pyc)

    assert ref_count_pyc == start_ref_count_pyc
