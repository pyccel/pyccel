# pylint: disable=missing-function-docstring, missing-module-docstring/
from numpy.random import randint

from pyccel.epyccel import epyccel

def test_sum_range(language):
    def f(a0 : 'int[:]'):
        return sum(a0[i] for i in range(len(a0)))

    n = randint(50)
    x = randint(100,size=n)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_var(language):
    def f(a : 'int[:]'):
        return sum(ai for ai in a)

    n = randint(50)
    x = randint(100,size=n)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_var2(language):
    def f(a : 'int[:,:]'):
        return sum(aii for ai in a for aii in ai)

    n1 = randint(5)
    n2 = randint(5)
    x = randint(10,size=(n1,n2))

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)
