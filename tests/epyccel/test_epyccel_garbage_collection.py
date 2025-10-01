# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import gc
import sys
import numpy as np
import pytest
from pyccel import epyccel

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

def test_return_unknown_class_pointer(language):
    import classes.return_unknown_class_pointer as mod_pyt

    mod = epyccel(mod_pyt, language=language)

    a1 = mod.A(4)
    a2 = mod.A(4)

    start_ref_count_a1 = sys.getrefcount(a1)
    start_ref_count_a2 = sys.getrefcount(a2)

    a_ptr = mod.choose_A(a1, a2, True)

    ref_count_a1 = sys.getrefcount(a1)
    ref_count_a2 = sys.getrefcount(a2)

    if language != 'python':
        # We don't know what is referenced so both are referenced
        assert ref_count_a1 == start_ref_count_a1+1
        assert ref_count_a2 == start_ref_count_a2+1
        assert ref_count_a1 == ref_count_a2
    else:
        assert ref_count_a1 > start_ref_count_a1

    del a_ptr

    ref_count_a1 = sys.getrefcount(a1)
    ref_count_a2 = sys.getrefcount(a2)

    assert ref_count_a1 == start_ref_count_a1
    assert ref_count_a2 == start_ref_count_a2

    a_ptr = mod.choose_A(a1, a2, False)

    ref_count_a1 = sys.getrefcount(a1)
    ref_count_a2 = sys.getrefcount(a2)

    if language != 'python':
        # We don't know what is referenced so both are referenced
        assert ref_count_a1 == start_ref_count_a1+1
        assert ref_count_a2 == start_ref_count_a2+1
        assert ref_count_a1 == ref_count_a2

    del a_ptr

    ref_count_a1 = sys.getrefcount(a1)
    ref_count_a2 = sys.getrefcount(a2)

    assert ref_count_a1 == start_ref_count_a1
    assert ref_count_a2 == start_ref_count_a2

def test_return_class_array_pointer(language):
    import classes.classes_1 as mod_pyt

    mod = epyccel(mod_pyt, language=language)

    x = np.array([0.,0.,0.])

    start_ref_count_x = sys.getrefcount(x)

    p1 = mod.Point(x)

    start_ref_count_p1 = sys.getrefcount(p1)

    saved_ref_count_x = sys.getrefcount(x)

    # p1 should reference x
    assert saved_ref_count_x > start_ref_count_x

    y = p1.get_x()

    local_ref_count_p1 = sys.getrefcount(p1)

    if language != 'python':
        # y should reference p1
        assert local_ref_count_p1 > start_ref_count_p1

    del p1

    local_ref_count_x = sys.getrefcount(x)

    # p1 should still reference x as it is preserved for use by y
    assert local_ref_count_x == saved_ref_count_x

    del y
    gc.collect()

    ref_count_x = sys.getrefcount(x)

    # All references should now be released
    assert ref_count_x == start_ref_count_x

def test_getter(language):
    import classes.array_attribute as mod_pyt

    mod = epyccel(mod_pyt, language=language)

    a = mod.A(10)

    start_ref_count_a = sys.getrefcount(a)

    b = a.x

    ref_count_a = sys.getrefcount(a)

    if language != 'python':
        assert b.base is a
        assert ref_count_a == start_ref_count_a+1

    del a

    gc.collect()

    a_x_elem = b[0]

    assert a_x_elem == 1

    if language != 'python':
        c = b.base.x

        np.array_equiv(b, c)

def test_setter(language):
    import classes.ptr_in_class as mod_pyt

    mod = epyccel(mod_pyt, language=language)

    target1 = np.ones(10)

    start_ref_count_target = sys.getrefcount(target1)

    cls = mod.A(target1)

    ref_count_target = sys.getrefcount(target1)

    assert ref_count_target == start_ref_count_target + 1

    target2 = np.ones(10)

    cls.x = target2

    ref_count_target1 = sys.getrefcount(target1)
    ref_count_target2 = sys.getrefcount(target2)

    if language == 'python':
        assert ref_count_target1 == start_ref_count_target
        assert ref_count_target2 == start_ref_count_target+1
    else:
        assert ref_count_target1 == start_ref_count_target + 1
        assert ref_count_target2 == start_ref_count_target + 1

    del cls

    ref_count_target1 = sys.getrefcount(target1)
    ref_count_target2 = sys.getrefcount(target2)

    assert ref_count_target1 == start_ref_count_target
    assert ref_count_target2 == start_ref_count_target

@pytest.mark.skipif(sys.version_info >= (3, 12), reason="PEP-0683 introduced immortal objects")
def test_return_bool(language):
    def get_true():
        return True

    f = epyccel(get_true, language=language)

    a = True
    ref_count_1 = sys.getrefcount(a)
    b = f()
    ref_count_2 = sys.getrefcount(a)
    assert ref_count_1 != ref_count_2
