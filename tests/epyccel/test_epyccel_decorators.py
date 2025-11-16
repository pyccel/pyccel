# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8
from typing import TypeVar, Final
import pytest
import numpy as np
from pyccel import epyccel
from pyccel.decorators import private, inline

@pytest.mark.parametrize( 'lang', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
    )
)
def test_private(lang):
    @private
    def f():
        print("hidden")

    # Attribute error when extracting f from module
    with pytest.raises(AttributeError):
        epyccel(f, language=lang)

def test_inline_1_out(language):
    def f():
        @inline
        def cube(s : int):
            return s * s * s
        a = cube(3)
        b = cube(8+3)
        c = cube((b-a)//20)
        d = cube(a)
        return a,b,c,d

    g = epyccel(f, language=language)

    assert f() == g()

def test_inline_0_out(language):
    def f(x : 'int[:]'):
        @inline
        def set_3(s : 'int[:]', i : int):
            s[i] = 3
        set_3(x, 0)
        set_3(x, 1)

    g = epyccel(f, language=language)

    x = np.ones(4, dtype=int)
    y = np.ones(4, dtype=int)

    f(x)
    g(y)

    assert all(x == y)

def test_inline_local(language):
    def f():
        @inline
        def power_4(s : int):
            x = s * s
            return x * x
        a = power_4(3)
        b = power_4(8+3)
        c = power_4((b-a)//2000)
        g = 4
        d = power_4(g)
        return a,b,c,d

    g = epyccel(f, language=language)

    assert f() == g()

def test_inline_local_name_clash(language):
    def f():
        @inline
        def power_4(s : int):
            x = s * s
            return x * x
        a = power_4(3)
        b = power_4(8+3)
        c = power_4((b-a)//2000)
        x = 2
        d = power_4(x)
        return a,b,c,d,x

    g = epyccel(f, language=language)

    assert f() == g()

def test_inline_optional(language):
    def f():
        @inline
        def get_val(x : int = None , y : int = None):
            if x is None :
                a = 3
            else:
                a = x
            if y is not None :
                b = 4
            else:
                b = 5
            return a + b
        a = get_val(2,7)
        b = get_val()
        c = get_val(6)
        d = get_val(y=0)
        return a,b,c,d

    g = epyccel(f, language=language)

    assert f() == g()

def test_inline_array(language):
    def f():
        from numpy import empty
        @inline
        def fill_array(a : 'float[:]'):
            for i in range(a.shape[0]):
                a[i] = 3.14
        arr = empty(4)
        fill_array(arr)
        return arr[0], arr[-1]

    g = epyccel(f, language=language)

    assert f() == g()

def test_nested_inline_call(language):
    def f():
        @inline
        def get_val(x : int = None , y : int = None):
            if x is None :
                a = 3
            else:
                a = x
            if y is not None :
                b = 4
            else:
                b = 5
            return a + b

        a = get_val(get_val(2)+3,7)
        return a

    g = epyccel(f, language=language)

    assert f() == g()

def test_inline_return(language):
    def f():
        @inline
        def tmp():
            a = 1
            return a

        b = tmp()
        c = tmp()
        d = tmp() + 3
        e = tmp() * 4
        return b,c,d,e

    g = epyccel(f, language=language)

    assert f() == g()

def test_inline_multiple_results(language):
    def f():
        @inline
        def get_2_vals(a : int):
            return a*2, a-5

        get_2_vals(5)
        x = get_2_vals(7)
        y0,y1 = get_2_vals(3)
        return x, y0, y1

    g = epyccel(f, language=language)

    assert f() == g()

def test_inline_literal_return(language):
    def f():
        @inline
        def tmp():
            return 2

        b = tmp()
        c = tmp()
        d = tmp() + 3
        e = tmp() * 4
        return b,c,d,e

    g = epyccel(f, language=language)

    assert f() == g()

def test_inline_array_return(language):
    def f():
        @inline
        def tmp():
            return np.ones(2, dtype=int)

        b = tmp()
        c = np.sum(tmp())
        return b,c

    g = epyccel(f, language=language)

    out_pyth = f()
    out_pycc = g()
    assert np.array_equal(out_pyth[0], out_pycc[0])
    assert out_pyth[1] == out_pycc[1]

def test_inline_multiple_return(language):
    def f():
        @inline
        def tmp():
            a = 1
            b = 4
            return a, b

        b,c = tmp()
        d,e = tmp()
        return b,c,d,e

    g = epyccel(f, language=language)

    assert f() == g()

def test_inline_homogeneous_tuple_result(language):
    def f():
        @inline
        def get_2_vals(a : int):
            b = (a*2, a-5)
            return b

        get_2_vals(5)
        x = get_2_vals(7)
        y0,y1 = get_2_vals(3)
        return x, y0, y1

    g = epyccel(f, language=language)

    assert f() == g()

def test_inline_inhomogeneous_tuple_result(language):
    def f():
        @inline
        def get_2_vals(a : int):
            b : tuple[int,int] = (a*2, a-5)
            return b

        get_2_vals(5)
        x = get_2_vals(7)
        y0,y1 = get_2_vals(3)
        return x, y0, y1

    g = epyccel(f, language=language)

    assert f() == g()

def test_inhomogeneous_tuple_in_inline(language):
    def f():
        @inline
        def tmp():
            a = (1, False)
            return a[0] + 2

        b = tmp()
        return b

    g = epyccel(f, language=language)

    assert f() == g()

def test_multi_level_inhomogeneous_tuple_in_inline(language):
    def f():
        @inline
        def tmp():
            a = ((1, False), 3.0)
            return a[0][0] + 2

        b = tmp()
        return b

    g = epyccel(f, language=language)

    assert f() == g()

def test_indexed_template(language):
    T = TypeVar('T', 'float[:]', 'complex[:]')

    def my_sum(v: Final[T]):
        return v.sum()

    pyccel_sum = epyccel(my_sum, language=language)

    x = np.ones(4, dtype=float)

    python_fl = my_sum(x)
    pyccel_fl = pyccel_sum(x)

    assert python_fl == pyccel_fl
    assert isinstance(python_fl, type(pyccel_fl))

    y = np.full(4, 1 + 3j)

    python_cmplx = my_sum(y)
    pyccel_cmplx = pyccel_sum(y)

    assert python_cmplx == pyccel_cmplx
    assert isinstance(python_cmplx, type(pyccel_cmplx))

@pytest.mark.parametrize("language", (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="lists not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
        )
)
def test_allow_negative_index_list(language):
    def allow_negative_index_annotation():
        a = [1,2,3,4]
        return a[-1], a[-2], a[-3], a[0]

    epyc_allow_negative_index_annotation = epyccel(allow_negative_index_annotation, language=language)

    assert epyc_allow_negative_index_annotation() == allow_negative_index_annotation()
    assert isinstance(epyc_allow_negative_index_annotation(), type(allow_negative_index_annotation()))

