# pylint: disable=missing-function-docstring, missing-module-docstring/

from pyccel.epyccel import epyccel
from pyccel.decorators import types

def test_abs_i(language):
    @types('int')
    def f1(x):
        return abs(x)

    f2 = epyccel(f1, language=language)

    assert f1(0) == f2(0)
    assert f1(-5) == f2(-5)
    assert f1(11) == f2(11)

def test_abs_r(language):
    @types('real')
    def f1(x):
        return abs(x)

    f2 = epyccel(f1, language=language)

    assert f1(0.00000) == f2(0.00000)
    assert f1(-3.1415) == f2(-3.1415)
    assert f1(2.71828) == f2(2.71828)



def test_abs_c(language):
    @types('complex')
    def f1(x):
        return abs(x)

    f2 = epyccel(f1, language=language)

    assert f1(3j + 4) == f2(3j + 4)
    assert f1(3j - 4) == f2(3j - 4)
    assert f1(5j + 0) == f2(5j + 0)
    assert f1(0j + 5) == f2(0j + 5)
    assert f1(0j + 0) == f2(0j + 0)
