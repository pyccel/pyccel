# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

import pytest
from pyccel.epyccel import epyccel
from pyccel.decorators import private, inline

@pytest.fixture(params=[
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c)
    ]
)
def language(request):
    return request.param

def test_private(language):
    @private
    def f():
        print("hidden")

    g = epyccel(f, language=language)

    with pytest.raises(NotImplementedError):
        g()

def test_inline(language):
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
