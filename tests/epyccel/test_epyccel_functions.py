# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

import numpy as np
import pytest
from modules import epyccel_functions
from numpy.random import randint

from pyccel import epyccel

from epyccel_utilities import epyccel_module_with_fallback
from tolerances import ATOL, RTOL


@pytest.fixture(scope="module")
def epyc_epyccel_functions_mod(language):
    return epyccel_module_with_fallback(epyccel_functions, language)


def test_func_no_args_1(language):
    """test function with return value but no args"""

    def free_gift():
        gift = 10
        return gift

    c_gift = epyccel(free_gift, language=language, folder="__pyccel__test_folder__")
    assert c_gift() == free_gift()
    assert isinstance(c_gift(), type(free_gift()))
    unexpected_arg = 0
    with pytest.raises(TypeError):
        c_gift(unexpected_arg)


def test_func_no_args_2(epyc_epyccel_functions_mod):
    """test function with negative return value but no args"""

    p_lose = epyccel_functions.p_lose
    c_lose = epyc_epyccel_functions_mod.p_lose
    assert c_lose() == p_lose()
    assert isinstance(c_lose(), type(p_lose()))
    unexpected_arg = 0
    with pytest.raises(TypeError):
        c_lose(unexpected_arg)


def test_func_no_return_1(epyc_epyccel_functions_mod):
    """Test function with args and no return"""

    p_func = epyccel_functions.p_func
    c_func = epyc_epyccel_functions_mod.p_func
    x = np.random.randint(100)
    assert c_func(x) == p_func(x)
    # Test type return should be NoneType
    x = np.random.randint(100)
    assert isinstance(c_func(x), type(p_func(x)))


def test_func_no_return_2(epyc_epyccel_functions_mod):
    """Test function with no args and no return"""

    p_func = epyccel_functions.func_no_return_2
    c_func = epyc_epyccel_functions_mod.func_no_return_2

    assert c_func() == p_func()
    assert isinstance(c_func(), type(p_func()))
    unexpected_arg = 0
    with pytest.raises(TypeError):
        c_func(unexpected_arg)


def test_func_no_args_f1(epyc_epyccel_functions_mod):
    f1 = epyccel_functions.func_no_args_f1
    f = epyc_epyccel_functions_mod.func_no_args_f1
    assert np.isclose(f(), f1(), rtol=RTOL, atol=ATOL)


def test_func_return_constant(epyc_epyccel_functions_mod):
    f1 = epyccel_functions.func_return_constant
    f = epyc_epyccel_functions_mod.func_return_constant
    assert np.isclose(f(), f1(), rtol=RTOL, atol=ATOL)


# ------------------------------------------------------------------------------
def test_decorator_f1(epyc_epyccel_functions_mod):
    f1 = epyccel_functions.decorator_f1
    f = epyc_epyccel_functions_mod.decorator_f1

    # ...
    assert f(3) == f1(3)
    # ...


# ------------------------------------------------------------------------------
def test_decorator_f2(epyc_epyccel_functions_mod):
    f2 = epyccel_functions.decorator_f2
    f = epyc_epyccel_functions_mod.decorator_f2

    # ...
    x = np.array([3, 4, 5, 6], dtype=int)
    assert f(x) == f2(x)
    # ...

    # ...
    x = np.array([3, 4, 5, 6], dtype=int)
    assert f(x) == f2(x)
    # ...


# ------------------------------------------------------------------------------
def test_decorator_f3(epyc_epyccel_functions_mod):
    f3 = epyccel_functions.decorator_f3
    f = epyc_epyccel_functions_mod.decorator_f3
    x = np.array([3, 4, 5, 6], dtype=int)
    assert np.all(f(x) == f3(x))


# ------------------------------------------------------------------------------
def test_decorator_f4(epyc_epyccel_functions_mod):
    f4 = epyccel_functions.decorator_f4
    f = epyc_epyccel_functions_mod.decorator_f4
    x = np.array([[3, 4, 5, 6], [3, 4, 5, 6]], dtype=float)
    assert np.all(f(x) == f4(x))


# ------------------------------------------------------------------------------
def test_decorator_f5(epyc_epyccel_functions_mod):
    f5 = epyccel_functions.decorator_f5
    f = epyc_epyccel_functions_mod.decorator_f5

    # ...
    m1 = 3

    x = np.zeros(m1)
    f(m1, x)

    x_expected = np.zeros(m1)
    f5(m1, x_expected)

    assert np.allclose(x, x_expected, rtol=RTOL, atol=ATOL)
    # ...


# ------------------------------------------------------------------------------
def test_decorator_f6(epyc_epyccel_functions_mod):
    f6_1 = epyccel_functions.f6_1
    f = epyc_epyccel_functions_mod.f6_1

    # ...
    m1 = 2
    m2 = 3

    x = np.zeros((m1, m2))
    f(m1, m2, x)

    x_expected = np.zeros((m1, m2))
    f6_1(m1, m2, x_expected)

    assert np.allclose(x, x_expected, rtol=RTOL, atol=ATOL)
    # ...


# ------------------------------------------------------------------------------
# in order to call the pyccelized function here, we have to create x with
# Fortran ordering
def test_decorator_f7(epyc_epyccel_functions_mod):

    f7 = epyccel_functions.decorator_f7
    f = epyc_epyccel_functions_mod.decorator_f7

    # ...
    m1 = 2
    m2 = 3
    x_expected = np.zeros((m1, m2))
    f7(m1, m2, x_expected)

    x = np.zeros((m1, m2), order="F")
    f(m1, m2, x)

    assert np.allclose(x, x_expected, rtol=RTOL, atol=ATOL)
    # ...


# ------------------------------------------------------------------------------
def test_decorator_f8(epyc_epyccel_functions_mod):
    f8 = epyccel_functions.decorator_f8
    f = epyc_epyccel_functions_mod.decorator_f8

    # ...
    assert f(3, True) == f8(3, True)
    assert f(3, False) == f8(3, False)
    # ...


def test_arguments_f9(epyc_epyccel_functions_mod):
    f9 = epyccel_functions.arguments_f9
    f = epyc_epyccel_functions_mod.arguments_f9

    x = np.zeros(10, dtype="int64")
    x_expected = x.copy()

    f9(x)
    f(x_expected)
    assert np.array_equal(x, x_expected)


def test_arguments_f10(epyc_epyccel_functions_mod):
    f10 = epyccel_functions.arguments_f10
    f = epyc_epyccel_functions_mod.arguments_f10

    x = np.zeros(10, dtype="int64")
    x_expected = x.copy()

    f10(x)
    f(x_expected)
    assert np.array_equal(x, x_expected)


def test_multiple_returns_f11(epyc_epyccel_functions_mod):
    ackermann = epyccel_functions.ackermann
    f = epyc_epyccel_functions_mod.ackermann
    assert f(2, 3) == ackermann(2, 3)


def test_multiple_returns_f12(epyc_epyccel_functions_mod):
    non_negative = epyccel_functions.non_negative
    f = epyc_epyccel_functions_mod.non_negative
    assert f(2) == non_negative(2)
    assert f(-1) == non_negative(-1)


def test_multiple_returns_f13(epyc_epyccel_functions_mod):
    get_min = epyccel_functions.get_min
    f = epyc_epyccel_functions_mod.get_min
    assert f(2, 3) == get_min(2, 3)


def test_multiple_returns_f14(epyc_epyccel_functions_mod):
    g = epyccel_functions.multiple_returns_f14
    f = epyc_epyccel_functions_mod.multiple_returns_f14
    assert f(2, 1) == g(2, 1)


def test_decorator_f15(epyc_epyccel_functions_mod):
    f15 = epyccel_functions.decorator_f15
    f = epyc_epyccel_functions_mod.decorator_f15
    assert f(True, np.int8(1), np.int16(2), np.int32(3), np.int64(4)) == f15(
        True, np.int8(1), np.int16(2), np.int32(3), np.int64(4)
    )
    assert f(False, np.int8(1), np.int16(2), np.int32(3), np.int64(4)) == f15(
        False, np.int8(1), np.int16(2), np.int32(3), np.int64(4)
    )


def test_decorator_f16(epyc_epyccel_functions_mod):
    f16 = epyccel_functions.decorator_f16
    f = epyc_epyccel_functions_mod.decorator_f16
    assert f(np.int16(17)) == f16(np.int16(17))


def test_decorator_f17(epyc_epyccel_functions_mod):
    f17 = epyccel_functions.decorator_f17
    f = epyc_epyccel_functions_mod.decorator_f17
    assert f(np.int8(2)) == f17(np.int8(2))


def test_decorator_f18(epyc_epyccel_functions_mod):
    f18 = epyccel_functions.decorator_f18
    f = epyc_epyccel_functions_mod.decorator_f18
    assert f(np.int32(5)) == f18(np.int32(5))


def test_decorator_f19(epyc_epyccel_functions_mod):
    f19 = epyccel_functions.decorator_f19
    f = epyc_epyccel_functions_mod.decorator_f19
    assert f(np.int64(1)) == f19(np.int64(1))


def test_decorator_f20(epyc_epyccel_functions_mod):
    f20 = epyccel_functions.decorator_f20
    f = epyc_epyccel_functions_mod.decorator_f20
    assert f(complex(1, 2.2)) == f20(complex(1, 2.2))


def test_decorator_f21(epyc_epyccel_functions_mod):
    f21 = epyccel_functions.decorator_f21
    f = epyc_epyccel_functions_mod.decorator_f21
    assert f(np.complex64(1 + 2.2j)) == f21(np.complex64(1 + 2.2j))


def test_decorator_f22(epyc_epyccel_functions_mod):
    f22 = epyccel_functions.decorator_f22
    f = epyc_epyccel_functions_mod.decorator_f22
    assert f(np.complex128(1 + 2.2j)) == f22(np.complex128(1 + 2.2j))


def test_union_type(epyc_epyccel_functions_mod):
    square = epyccel_functions.square
    f = epyc_epyccel_functions_mod.square
    x = np.random.randint(40)
    y = np.random.uniform()

    assert np.isclose(f(x), square(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(square(x)))
    assert np.isclose(f(y), square(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(square(y)))


def test_return_annotation(epyc_epyccel_functions_mod):
    get_2 = epyccel_functions.get_2
    f = epyc_epyccel_functions_mod.get_2
    assert f() == get_2()


@pytest.mark.skipif_by_language(True, language="python", reason="no error from Python")
def test_wrong_argument_type(epyc_epyccel_functions_mod):
    epyc_f = epyc_epyccel_functions_mod.wrong_argument_type
    test_arg = 3.5
    with pytest.raises(TypeError) as err:
        epyc_f(test_arg)
    assert "integer_arg" in str(err.value)
    assert str(type(test_arg)) in str(err.value)


@pytest.mark.skipif_by_language(True, language="python", reason="no error from Python")
def test_wrong_known_argument_type_in_interface(epyc_epyccel_functions_mod):
    epyc_f = epyc_epyccel_functions_mod.wrong_known_argument_type_in_interface

    test_arg = 4.5
    with pytest.raises(TypeError) as err:
        epyc_f(3.5, test_arg)
    assert "integer_arg" in str(err.value)
    assert str(type(test_arg)) in str(err.value)


@pytest.mark.skipif_by_language(True, language="python", reason="no error from Python")
def test_wrong_known_argument_type_in_interface_with_default(
    epyc_epyccel_functions_mod,
):
    epyc_f = (
        epyc_epyccel_functions_mod.wrong_known_argument_type_in_interface_with_default
    )

    test_arg = 4.5
    with pytest.raises(TypeError) as err:
        epyc_f(3.5, test_arg)
    assert "integer_arg" in str(err.value)
    assert str(type(test_arg)) in str(err.value)


@pytest.mark.skipif_by_language(True, language="python", reason="no error from Python")
def test_wrong_unknown_argument_type_in_interface(epyc_epyccel_functions_mod):
    epyc_f = epyc_epyccel_functions_mod.wrong_known_argument_type_in_interface

    test_arg = 3.5 + 1j
    with pytest.raises(TypeError) as err:
        epyc_f(test_arg, 4.5)
    assert "templated_arg" in str(err.value)
    assert str(type(test_arg)) in str(err.value)


@pytest.mark.skipif_by_language(True, language="python", reason="no error from Python")
def test_wrong_argument_combination_in_interface(epyc_epyccel_functions_mod, language):
    epyc_f = epyc_epyccel_functions_mod.wrong_argument_combination_in_interface

    with pytest.raises(TypeError):
        epyc_f(3.5, 4)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param("c", marks=pytest.mark.c),
    ),
)
def test_argument_checks_with_interfaces(language):
    from modules import Module_12 as mod

    modnew = epyccel(mod, language=language)
    with pytest.raises(TypeError):
        modnew.times_3(1)
    with pytest.raises(TypeError):
        modnew.add_2(1)


def test_container_interface(epyc_epyccel_functions_mod):
    f = epyccel_functions.container_interface
    epyc_f = epyc_epyccel_functions_mod.container_interface
    assert f([1, 2]) == epyc_f([1, 2])
    assert f({1, 2}) == epyc_f({1, 2})
    assert f(np.array([1, 2])) == epyc_f(np.array([1, 2]))


def test_lambda(epyc_epyccel_functions_mod):
    f = epyccel_functions.lambda_f
    epyc_f = epyc_epyccel_functions_mod.lambda_f
    val = randint(20)
    assert f(val) == epyc_f(val)
    assert isinstance(epyc_f(val), type(epyc_f(val)))


def test_lambda_2(epyc_epyccel_functions_mod):
    f = epyccel_functions.lambda_2
    epyc_f = epyc_epyccel_functions_mod.lambda_2
    val = randint(20)
    assert f(val) == epyc_f(val)
    assert isinstance(epyc_f(val), type(epyc_f(val)))


@pytest.mark.language_agnostic
def test_argument_types():
    def f(a: int, /, b: int, *args: int, c: int, **kwargs: int):
        my_sum = sum(v for v in kwargs.values())
        return my_sum + 2 * a + 3 * b + 5 * c + 7 * sum(args)

    epyc_f = epyccel(f, language="python")
    a = 8
    b = 9
    c = 25
    args = (7, 14, 21)
    kwargs = {"d": 11, "f": 13}
    assert f(a, b, *args, c=c, **kwargs) == epyc_f(a, b, *args, c=c, **kwargs)


def test_positional_only_arguments(language):
    def f(a: int, /, b: int):
        return 2 * a + 3 * b

    epyc_f = epyccel(f, language=language)
    a = 8
    b = 9
    assert f(a, b) == epyc_f(a, b)
    assert f(a, b=b) == epyc_f(a, b=b)
    with pytest.raises(TypeError):
        epyc_f(a=a, b=b)


def test_keyword_only_arguments(language):
    def f(a: int, *, b: int):
        return 2 * a + 3 * b

    epyc_f = epyccel(f, language=language)
    a = 8
    b = 9
    assert f(a, b=b) == epyc_f(a, b=b)
    with pytest.raises(TypeError):
        epyc_f(a, b)


def test_lambda_usage(language):
    f = lambda x: x + 1  # pylint: disable=unnecessary-lambda-assignment

    def g(a: "int64[:]"):
        for i, ai in enumerate(a):
            a[i] = f(ai)

    epyc_g = epyccel(g, language=language)
    val = randint(20, size=(10,), dtype=np.int64)
    val_epyc = val.copy()
    g(val)
    epyc_g(val_epyc)
    assert np.array_equal(val, val_epyc)


def test_func_usage(language):
    def f(x: int):
        return x + 1

    def g(a: "int64[:]"):
        for i, ai in enumerate(a):
            a[i] = f(ai)

    epyc_g = epyccel(g, language=language)
    val = randint(20, size=(10,), dtype=np.int64)
    val_epyc = val.copy()
    g(val)
    epyc_g(val_epyc)
    assert np.array_equal(val, val_epyc)
