# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import pytest
import numpy as np
import modules.expose_classes as mod
from pyccel import epyccel

RTOL = 2e-14
ATOL = 1e-15

@pytest.fixture(scope="module")
def modnew(language):
    return epyccel(mod, language = language)

def test_empty_class(language):
    class A:
        pass

    epyc_A = epyccel(A, language = language)

    assert isinstance(epyc_A, type)

def test_class_import(language):
    class A:
        def __init__(self : 'A'):
            pass

    epyc_A = epyccel(A, language = language)

    assert isinstance(epyc_A, type)


def test_class_return(modnew):
    a = modnew.get_A()
    assert isinstance(a, modnew.A)
    a_new, i = modnew.get_A_int()
    a_new2, i2 = mod.get_A_int()
    assert isinstance(a_new, modnew.A)
    assert isinstance(a_new2, mod.A)
    assert i == i2

    b = modnew.get_B(3.0)
    assert isinstance(b, modnew.B)


def test_class_argument(modnew):
    b = modnew.get_B(3.0)
    assert isinstance(b, modnew.B)

    x = modnew.get_x_from_B(b)
    assert x == 3.0
    x = modnew.get_an_x_from_B(b)
    assert x == 3.0
    x = modnew.get_an_x_from_B()
    assert x == -2.0

def test_class_function(modnew):
    c = modnew.C()
    assert c.get_3() == 3

def test_classes_1(language):
    import classes.classes_1 as mod
    modnew = epyccel(mod, language = language)

    x1 = np.array([0.,0.,0.])
    x2 = np.array([0.,0.,0.])
    a = np.array([1.,1.,1.])

    p1_py = mod.Point(x1)
    p1_l  = modnew.Point(x2)

    assert np.allclose(p1_py.get_x(), p1_l.get_x(), rtol=RTOL, atol=ATOL)
    assert p1_py.get_X() == p1_l.get_X()

    p1_py.translate(a)
    p1_l.translate(a)

    assert np.allclose(p1_py.get_x(), p1_l.get_x(), rtol=RTOL, atol=ATOL)
    assert p1_py.get_X() == p1_l.get_X()
    assert np.allclose(x1, x2, rtol=RTOL, atol=ATOL)

    p2_py = mod.Point(np.array([6.,6.,6.]))
    p2_l  = modnew.Point(np.array([6.,6.,6.]))

    print(p2_py.get_x(), p2_l.get_x())

    assert np.allclose(p2_py.get_x(), p2_l.get_x(), rtol=RTOL, atol=ATOL)
    print(p2_py.get_x(), p2_l.get_x())
    assert p2_py.get_X() == p2_l.get_X()

    print(p2_py.get_x(), p2_l.get_x())
    p2_py.translate(a)
    print(p2_py.get_x(), p2_l.get_x())
    p2_l.translate(a)
    print(p2_py.get_x(), p2_l.get_x())

    assert np.allclose(p2_py.get_x(), p2_l.get_x(), rtol=RTOL, atol=ATOL)
    assert p2_py.get_X() == p2_l.get_X()
    assert np.allclose(x1, x2, rtol=RTOL, atol=ATOL)

    l_py = mod.Line(p1_py)
    l_l  = modnew.Line(p1_l)

    assert p1_py.get_X() == p1_l.get_X()
    assert np.allclose(l_py.get_x(), l_l.get_x(), rtol=RTOL, atol=ATOL)

def test_classes_2(language):
    import classes.classes_2 as mod
    modnew = epyccel(mod, language = language)

    p_py = mod.Point()
    p_l  = modnew.Point()

    x = np.ones(4)
    y = np.full(4, 3.0)

    a_py = p_py.addition(1.1, 2.0)
    b_py = p_py.subtraction(y, x)

    a_l  = p_l.addition(1.1, 2.0)
    b_l  = p_l.subtraction(y, x)

    assert a_py == a_l
    assert isinstance(a_py, type(a_l))

    assert np.allclose(b_py, b_l, rtol=RTOL, atol=ATOL)

def test_classes_3(language):
    import classes.classes_3 as mod
    modnew = epyccel(mod, language = language)

    p2_py = mod.Point2(2.2)
    p2_l  = modnew.Point2(2.2)
    assert p2_py.test_func() == p2_l.test_func()

    p2_py = mod.Point2(6.5)
    p2_l  = modnew.Point2(6.5)
    assert p2_py.test_func() == p2_l.test_func()

    p_py = mod.Point(3.5, 0.1)
    p_l  = modnew.Point(3.5, 0.1)

    assert p_py.get_coordinates() == p_l.get_coordinates()

    p_py.set_coordinates(2.3, 5.1)
    p_l.set_coordinates(2.3, 5.1)

    assert p_py.get_coordinates() == p_l.get_coordinates()

    assert p_py.x == p_l.x
    assert p_py.X == p_l.X
    assert isinstance(p_py.x, type(p_l.x))
    assert isinstance(p_py.X, type(p_l.X))

    p_py.x = -1.2
    p_py.X = -10.2

    p_l.x = -1.2
    p_l.X = -10.2

    assert p_py.get_coordinates() == p_l.get_coordinates()

def test_classes_4(language):
    import classes.classes_4 as mod
    modnew = epyccel(mod, language = language)

    x1 = np.array([0.,0.,0.])
    x2 = np.array([0.,0.,0.])
    a = np.array([1.,1.,1.])

    p1_py = mod.Point(x1)
    p1_l  = modnew.Point(x2)

    assert np.isclose(p1_py.get_x(), p1_l.get_x(), rtol=RTOL, atol=ATOL)

    p1_py.translate(a)
    p1_l.translate(a)

    assert np.isclose(p1_py.get_x(), p1_l.get_x(), rtol=RTOL, atol=ATOL)
    assert np.allclose(x1, x2, rtol=RTOL, atol=ATOL)

    p1_py = mod.Point()

def test_classes_5(language):
    import classes.classes_5 as mod
    modnew = epyccel(mod, language = language)

    x1 = np.array([0.,0.,0.])
    x2 = np.array([0.,0.,0.])
    a = np.array([1.,1.,1.])

    p1_py = mod.Point(x1)
    p1_l  = modnew.Point(x2)

    assert np.allclose(p1_py.x, p1_l.x, rtol=RTOL, atol=ATOL)

    p1_py.translate(a)
    p1_l.translate(a)

    assert np.allclose(p1_py.x, p1_l.x, rtol=RTOL, atol=ATOL)
    assert np.allclose(x1, x2, rtol=RTOL, atol=ATOL)

    with pytest.raises(AttributeError):
        p1_l.x = 4.0

def test_classes_6(language):
    import classes.classes_6 as mod
    modnew = epyccel(mod, language = language)

    p_py = mod.Point(0.0, 0.0)
    p_l  = modnew.Point(0.0, 0.0)

    assert p_py.get_attributes(3) == p_l.get_attributes(3)
    assert p_py.get_attributes(4.5) == p_l.get_attributes(4.5)

    p_py.translate(1.0, 2.0)
    p_l.translate(1.0, 2.0)

    assert p_py.get_attributes(3) == p_l.get_attributes(3)
    assert p_py.get_attributes(4.5) == p_l.get_attributes(4.5)

def test_classes_7(language):
    import classes.classes_7 as mod
    modnew = epyccel(mod, language = language)

    p_py = mod.get_A()
    p_l  = modnew.get_A()

    assert mod.get_x_from_A() == modnew.get_x_from_A()
    assert mod.get_x_from_A(p_py) == modnew.get_x_from_A(p_l)

    p_py.update(10)
    p_l.update(10)

    assert mod.get_x_from_A(p_py) == modnew.get_x_from_A(p_l)

def test_classes_8(language):
    import classes.classes_8 as mod
    modnew = epyccel(mod, language = language)

    a_py = mod.A(3.0)
    a_l = modnew.A(3.0)

    assert a_py.x == a_l.x
    assert a_py.y == a_l.y

def test_classes_9(language):
    import classes.classes_9 as mod
    modnew = epyccel(mod, language = language)

    a_py = mod.A(3.0)
    a_l = modnew.A(3.0)

    assert a_py.get_A_contents() == a_l.get_A_contents()
    assert a_py.x == a_l.x

def test_classes_10(language):
    import classes.classes_10 as mod
    modnew = epyccel(mod, language = language)

    a = modnew.A(3)

    assert isinstance(a, modnew.A)
    assert isinstance(a[0], modnew.B)

def test_generic_methods(language):
    import classes.generic_methods as mod
    modnew = epyccel(mod, language = language)

    p_py = mod.Point(0.0, 0.0)
    p_l  = modnew.Point(0.0, 0.0)

    assert p_py.get_x() == p_l.get_x()
    assert p_py.get_y() == p_l.get_y()

    p_py.translate(1.0, 2.0)
    p_l.translate(1.0, 2.0)

    assert p_py.get_x() == p_l.get_x()
    assert p_py.get_y() == p_l.get_y()

    p_py.translate(1, 2)
    p_l.translate(1, 2)

    assert p_py.get_x() == p_l.get_x()
    assert p_py.get_y() == p_l.get_y()

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason="Attribute renamed. See #1705"),
            pytest.mark.python]
        )
    )
)
def test_classes(language):
    import classes.classes as mod
    modnew = epyccel(mod, language = language)

    p_py = mod.Point(0.0, 0.0)
    p_l  = modnew.Point(0.0, 0.0)

    assert p_py.x == p_l.x
    assert isinstance(p_py.x, type(p_l.x))

    assert p_py.y == p_l.y
    assert isinstance(p_py.y, type(p_l.y))

    p_py.x = 2.0
    p_l.x = 2.0

    if language != 'python':
        with pytest.raises(TypeError):
            p_l.y = 1j

    assert p_py.x == p_l.x

    p_py.translate(1.0, 2.0)
    p_l.translate(1.0, 2.0)

    assert p_py.x == p_l.x
    assert p_py.y == p_l.y

def test_class_out(language):
    import classes.array_attribute as mod
    modnew = epyccel(mod, language = language)

    p_py = mod.A(5)
    p_l  = modnew.A(5)

    assert np.array_equal(p_py.x, p_l.x)
    assert np.array_equal(p_py.get_x(), p_l.get_x())

    p_py.x[:] = 4
    p_l.x[:] = 4

    assert np.array_equal(p_py.x, p_l.x)

    if language != 'python':
        with pytest.raises(AttributeError):
            p_l.x = np.ones(6)

def test_ptr_in_class(language):
    import classes.ptr_in_class as mod
    modnew = epyccel(mod, language = language)

    x = np.ones(4)
    a_py = mod.A(x)
    a_l = modnew.A(x)

    assert np.array_equal(a_py.x, a_l.x)

    x[2] = 3

    assert np.array_equal(a_py.x, a_l.x)

    y = np.zeros(3)
    a_py.x = y
    a_l.x = y

    assert np.array_equal(a_py.x, a_l.x)

    y[0] = -3

    assert np.array_equal(a_py.x, a_l.x)

def test_class_constness(language):
    import classes.class_constness as mod
    modnew = epyccel(mod, language = language)

    a_py = mod.ArrProperties(4)
    a_l = modnew.ArrProperties(4)

    assert a_py.n_points == a_l.n_points

    x_py = mod.f(a_py)
    x_l = modnew.f(a_l)

    assert np.array_equal(x_py, x_l)

def test_class_magic(language):
    import classes.class_magic as mod
    modnew = epyccel(mod, language = language)

    a_py = mod.A(4)
    a_l = modnew.A(4)

    assert a_py.x == a_l.x

    left_add_py = a_py + 5
    left_add_l = a_l + 5

    assert isinstance(left_add_l, modnew.A)
    assert left_add_py.x == left_add_l.x

    left_sub_py = a_py - 2
    left_sub_l = a_l - 2

    assert isinstance(left_sub_l, modnew.A)
    assert left_sub_py.x == left_sub_l.x

    left_mul_py = a_py * 2
    left_mul_l = a_l * 2

    assert isinstance(left_mul_l, modnew.A)
    assert left_mul_py.x == left_mul_l.x

    left_truediv_py = a_py / 2
    left_truediv_l = a_l / 2

    assert isinstance(left_truediv_l, modnew.A)
    assert left_truediv_py.x == left_truediv_l.x

    left_lshift_py = a_py << 2
    left_lshift_l = a_l << 2

    assert isinstance(left_lshift_l, modnew.A)
    assert left_lshift_py.x == left_lshift_l.x

    left_rshift_py = a_py >> 2
    left_rshift_l = a_l >> 2

    assert isinstance(left_rshift_l, modnew.A)
    assert left_rshift_py.x == left_rshift_l.x

    left_and_py = a_py & 2
    left_and_l = a_l & 2

    assert isinstance(left_and_l, modnew.A)
    assert left_and_py.x == left_and_l.x

    left_or_py = a_py | 2
    left_or_l = a_l | 2

    assert isinstance(left_or_l, modnew.A)
    assert left_or_py.x == left_or_l.x

    a_py += 6
    a_l += 6

    assert a_py.x == a_l.x

    a_py -= 6
    a_l -= 6

    assert a_py.x == a_l.x

    a_py *= 6
    a_l *= 6

    assert a_py.x == a_l.x

    a_py /= 2
    a_l /= 2

    assert a_py.x == a_l.x

    a_py <<= 1
    a_l <<= 1

    assert a_py.x == a_l.x

    a_py >>= 2
    a_l >>= 2

    assert a_py.x == a_l.x

    a_py &= 7
    a_l &= 7

    assert a_py.x == a_l.x

    a_py |= 6
    a_l |= 6

    assert a_py.x == a_l.x

    assert len(a_py) == len(a_l)

    for i in range(5):
        assert a_py[i] == a_l[i]

def test_class_property_name_conflict(language):
    import classes.class_property_name_conflict as mod
    modnew = epyccel(mod, language = language)

    a_py = mod.A(3.0)
    a_l = modnew.A(3.0)

    assert a_py.x == a_l.x

    a_py.translate(4.5)
    a_l.translate(4.5)

    assert a_py.x == a_l.x

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason="Attribute renamed. See #1705"),
            pytest.mark.python]
        )
    )
)
def test_class_globals_visitation_order(language):
    import classes.class_globals_visitation_order as mod
    modnew = epyccel(mod, language = language)
    a_py = mod.A()
    a_l = modnew.A()

    assert a_py.x == a_l.x

def test_class_variables(language):
    import classes.class_variables as mod
    modnew = epyccel(mod, language = language)

    for bool_val in (True, False):
        a_py = mod.A(bool_val)
        a_l = modnew.A(bool_val)

        assert a_py.x == a_l.x
        assert a_py.z == a_l.z
        assert a_py.get_4() == a_l.get_4()
        assert np.array_equal(a_py.y, a_l.y)
        assert a_py.get_y_len() == a_l.get_y_len()
