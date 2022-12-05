# pylint: disable=missing-function-docstring, missing-module-docstring/

import pytest

from pyccel.epyccel import epyccel
from pyccel.decorators import types
# wp suffix means With Parentheses
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

    @types('int')
    def f3wp(x):
        a = (x if x < 5 else 5) + 2
        return a

    f = epyccel(f3, language = language)
    fwp = epyccel(f3wp, language = language)

    # ...
    assert f(6) == f3(6)
    assert f(4) == f3(4)

    assert fwp(6) == f3wp(6)
    assert fwp(4) == f3wp(4)
    # ...
#------------------------------------------------------------------------------

def test_f4(language):
    @types('int')
    def f4(x):
        a = x if x < 5 else 5 >> 2
        return a

    @types('int')
    def f4wp(x):
        a = (x if x < 5 else 5) >> 2
        return a

    f = epyccel(f4, language = language)
    fwp = epyccel(f4wp, language = language)

    # ...
    assert f(6) == f4(6)
    assert f(4) == f4(4)

    assert fwp(6) == f4wp(6)
    assert fwp(4) == f4wp(4)
    # ...
#------------------------------------------------------------------------------
def test_f5(language):
    @types('int')
    def f5(x):
        a = x if x < 5 else 5 if x == 5 else 5.5
        return a

    @types('int')
    def f5wp(x):
        a = x if x < 5 else (5 if x == 5 else 5.5)
        return a

    f = epyccel(f5, language = language)
    fwp = epyccel(f5wp, language = language)

    # ...
    assert f(6) == f5(6)
    assert f(4) == f5(4)
    assert f(5) == f5(5)

    assert fwp(6) == f5wp(6)
    assert fwp(4) == f5wp(4)
    assert fwp(5) == f5wp(5)
    # ...
#------------------------------------------------------------------------------
def test_f6(language):
    @types('int')
    def f6(x):
        # a = x if x < 0 else (1 if x < 5 else (complex(0, 1) if x == 5 else 6.5))
        a = x if x < 0 else 1 if x < 5 else complex(0, 1) if x == 5 else 6.5
        return a

    @types('int')
    def f6wp(x):
        a = x if x < 0 else (1 if x < 5 else (complex(0, 1) if x == 5 else 6.5))
        return a

    f = epyccel(f6, language = language)
    fwp = epyccel(f6wp, language = language)

    # ...
    assert f(6) == f6(6)
    assert f(4) == f6(4)
    assert f(5) == f6(5)

    assert fwp(6) == f6wp(6)
    assert fwp(4) == f6wp(4)
    assert fwp(5) == f6wp(5)
    # ...
#------------------------------------------------------------------------------

def test_f7(language):
    @types('int')
    def f7(x):
        import numpy as np
        a = np.array([1.,2.,3.]) if x < 5 else np.array([1.5,6.5,7.5])
        return a[0]

    @types('int')
    def f7wp(x):
        import numpy as np
        a = np.array([1.,2.,3.]) if x < 5 else (np.array([1.5,6.5,7.5]) if x > 5 else np.array([3.1,9.5,2.8]))
        return a[0]

    f = epyccel(f7, language = language)
    fwp = epyccel(f7wp, language = language)

    # ...
    assert f(6) == f7(6)
    assert f(4) == f7(4)

    assert fwp(6) == f7wp(6)
    assert fwp(4) == f7wp(4)
    # ...
#------------------------------------------------------------------------------

def test_f8(language):
    @types('int')
    def f8(x):
        a = (1+0j, 2+0j) if x < 5 else (complex(5, 1), complex(2, 2))
        return a[0]

    @types('int')
    def f8wp(x):
        a = (1+0j, 2+0j) if x < 5 else ((complex(5, 1), complex(2, 2)) if x > 5 else (complex(7, 2), complex(3, 3)) )
        return a[0]

    f = epyccel(f8, language = language)
    fwp = epyccel(f8wp, language = language)

    # ...
    assert f(6) == f8(6)
    assert f(4) == f8(4)

    assert fwp(6) == f8wp(6)
    assert fwp(4) == f8wp(4)
    # ...
#------------------------------------------------------------------------------

def test_f9(language):
    @types('int')
    def f9(x):
        a = 1 + 2 if x < 5 else 3
        return a

    @types('int')
    def f9wp1(x):
        a = 1 + (2 if x < 5 else 3)
        return a

    @types('int')
    def f9wp2(x):
        a = (1 + 2) if x < 5 else 3
        return a

    f = epyccel(f9, language = language)
    fwp1 = epyccel(f9wp1, language = language)
    fwp2 = epyccel(f9wp2, language = language)
    # ...
    assert f(6) == f9(6)
    assert f(4) == f9(4)

    assert fwp1(6) == f9wp1(6)
    assert fwp1(4) == f9wp1(4)

    assert fwp2(6) == f9wp2(6)
    assert fwp2(4) == f9wp2(4)
    # ...
#------------------------------------------------------------------------------

def test_f10(language):
    @types('int')
    def f10(x):
        a = 2 if x < 5 else 3 + 1
        return a

    @types('int')
    def f10wp1(x):
        a = (2 if x < 5 else 3) + 1
        return a

    @types('int')
    def f10wp2(x):
        a = 2 if x < 5 else (3 + 1)
        return a

    f = epyccel(f10, language = language)
    fwp1 = epyccel(f10wp1, language = language)
    fwp2 = epyccel(f10wp2, language = language)
    # ...
    assert f(6) == f10(6)
    assert f(4) == f10(4)

    assert fwp1(6) == f10wp1(6)
    assert fwp1(4) == f10wp1(4)

    assert fwp2(6) == f10wp2(6)
    assert fwp2(4) == f10wp2(4)
    # ...
#------------------------------------------------------------------------------

def test_f11(language):
    @types('int')
    def f11(x):
        a = 2 if (x + 2)*5 < 5 else 3
        return a

    f = epyccel(f11, language = language)
    # ...
    assert f(6) == f11(6)
    assert f(-4) == f11(-4)
    # ...
#------------------------------------------------------------------------------

def test_f12(language):
    @types('int')
    def f12(x):
        import numpy as np
        a = np.array([1.,2.,3.,4.]) if x < 5 else np.array([1.5,6.5,7.5])
        return a[0]

    @types('int')
    def f12wp(x):
        import numpy as np
        a = np.array([1.,2.,3.]) if x < 5 else (np.array([1.5,6.5,7.5]) if x > 5 else np.array([3.1,9.5,2.8,2.9]))
        return a[0]

    f = epyccel(f12, language = language)
    fwp = epyccel(f12wp, language = language)

    # ...
    assert f(6) == f12(6)
    assert f(4) == f12(4)

    assert fwp(6) == f12wp(6)
    assert fwp(4) == f12wp(4)
    # ...
#------------------------------------------------------------------------------

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="Can't return a string"),
            pytest.mark.fortran]
        ),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Can't declare a string"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_f13(language):
    def f13(b : bool):
        a = 'hello' if b else 'world!'
        return a

    def f13wp(b1 : bool, b2 : bool):
        a = 'hello' if b1 else ('world' if b2 else 'hello world')
        return a

    f = epyccel(f13, language = language)
    fwp = epyccel(f13wp, language = language)

    # ...
    assert f(True) == f13(True)
    assert f(False) == f13(False)

    assert fwp(True,True)   == f13wp(True,True)
    assert fwp(True,False)  == f13wp(True,False)
    assert fwp(False,True)  == f13wp(False,True)
    assert fwp(False,False) == f13wp(False,False)
    # ...
#------------------------------------------------------------------------------
