# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

import pytest
import numpy as np
from pyccel.epyccel import epyccel
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
        c = power_4((b-a)//20)
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
        c = power_4((b-a)//20)
        x = 2
        d = power_4(x)
        return a,b,c,d,x

    g = epyccel(f, language=language)

    assert f() == g()
