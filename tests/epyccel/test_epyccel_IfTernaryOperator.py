# pylint: disable=missing-function-docstring, missing-module-docstring/

import pytest

from pyccel.epyccel import epyccel
from pyccel.decorators import types

#------------------------------------------------------------------------------
def test_f1(language):
    @types('int')
    def f1(x):
        a = 5 if x < 5 else x
        return a

    f = epyccel(f1, language = language)

    # ...
    assert f(6) == f1(6)
    assert f(4) == f1(4)
    # ...
#------------------------------------------------------------------------------

def test_f2(language):
    @types('int')
    def f2(x):
        a = 5.5 if x < 5 else x
        return a

    f = epyccel(f2, language = language)

    # ...
    assert f(6) == f2(6)
    assert f(4) == f2(4)
    # ...
#------------------------------------------------------------------------------
def test_f3(language):
    @types('int')
    def f3(x):
        a = x if x < 5 else 5 + 2
        return a

    f = epyccel(f3, language = language)

    # ...
    assert f(6) == f3(6)
    assert f(4) == f3(4)
    # ...
#------------------------------------------------------------------------------

def test_f4(language):
    @types('int')
    def f4(x):
        a = x if x < 5 else 5 >> 2
        return a

    f = epyccel(f4, language = language)

    # ...
    assert f(6) == f4(6)
    assert f(4) == f4(4)
    # ...
#------------------------------------------------------------------------------
def test_f5(language):
    @types('int')
    def f5(x):
        a = x if x < 5 else 5 if x == 5 else 5.5
        return a

    f = epyccel(f5, language = language)

    # ...
    assert f(6) == f5(6)
    assert f(4) == f5(4)
    assert f(5) == f5(5)
    # ...
#------------------------------------------------------------------------------
def test_f6(language):
    @types('int')
    def f6(x):
        # a = x if x < 0 else (1 if x < 5 else (complex(0, 1) if x == 5 else 6.5))
        a = x if x < 0 else 1 if x < 5 else complex(0, 1) if x == 5 else 6.5
        return a

    f = epyccel(f6, language = language)

    # ...
    assert f(6) == f6(6)
    assert f(4) == f6(4)
    assert f(5) == f6(5)
    # ...
#------------------------------------------------------------------------------

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Arrays are not yet implemented for C language"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_f7(language):
    @types('int')
    def f7(x):
        a = [1,2,3] if x < 5 else [1.5,6.5,7.5]
        return a[0]

    f = epyccel(f7, language = language)

    # ...
    assert f(6) == f7(6)
    assert f(4) == f7(4)
    # ...
#------------------------------------------------------------------------------

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Tuples are not yet implemented for C language"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_f8(language):
    @types('int')
    def f8(x):
        a = (1, 2) if x < 5 else (complex(5, 1), complex(2, 2))
        return a[0]

    f = epyccel(f8, language = language)

    # ...
    assert f(6) == f8(6)
    assert f(4) == f8(4)
    # ...
#------------------------------------------------------------------------------
