# pylint: disable=missing-function-docstring, missing-module-docstring/

from pyccel.epyccel import epyccel
from pyccel.decorators import types

def test_abs_i(language):
    @types('int')
    def f1(x):
        return abs(x)

    f2 = epyccel(f1, language=language)

    assert f1(11) == f2(11)

def test_abs_r(language):
    @types('real')
    def f1(x):
        return abs(x)

    f2 = epyccel(f1, language=language)

    assert f1(-3.1415) == f2(-3.1415)


def test_abs_c(language):
    @types('complex')
    def f1(x):
        return abs(x)

    f2 = epyccel(f1, language=language)

    assert f1(3j+2) == f2(3j+2)
