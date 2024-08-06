# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

import pytest
import numpy as np
from pyccel import epyccel
from pyccel.decorators import private, inline, template

@pytest.mark.parametrize( 'lang', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
    )
)
def test_private(lang):
    @private
    def f():
        print("hidden")

    g = epyccel(f, language=lang)

    with pytest.raises(NotImplementedError):
        g()

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

def test_indexed_template(language):
    @template(name='T', types=[float, complex])
    def my_sum(v: 'T[:]'):
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

