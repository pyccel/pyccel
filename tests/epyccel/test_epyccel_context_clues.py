# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring
import sys
from typing import TypeVar, Final
import numpy as np
import pytest
from pyccel import epyccel
from pyccel.errors.errors import PyccelError

def test_numpy_context(language):
    def f():
        a = np.ones(5)
        return a.sum()

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_literal_context(language):
    a = 4
    b = np.float32(4.5)
    def f():
        return a + b

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()
    assert isinstance(f(), type(epyc_f()))

def test_type_alias_context(language):
    T = int

    def f(a : T) -> T:
        return 2*a

    epyc_f = epyccel(f, language=language)
    assert f(3) == epyc_f(3)
    assert isinstance(f(2), type(epyc_f(2)))

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Union of types implemented in Python 3.10")
def test_type_union_context(language):
    T = int | float #pylint: disable=unsupported-binary-operation

    def f(a : T):
        return 2*a

    epyc_f = epyccel(f, language=language)
    assert f(3) == epyc_f(3)
    assert isinstance(f(2), type(epyc_f(2)))
    assert f(3.5) == epyc_f(3.5)
    assert isinstance(f(2.3), type(epyc_f(2.3)))

def test_type_enclosing_context(language):
    def get_func(b : bool):
        T = int if b else float

        def f(a : T):
            return 2*a

        return f

    epyc_f = epyccel(get_func(True), language=language)
    py_f = get_func(True)
    assert py_f(3) == epyc_f(3)
    assert isinstance(py_f(2), type(epyc_f(2)))
    epyc_f = epyccel(get_func(False), language=language)
    py_f = get_func(False)
    assert py_f(3.5) == epyc_f(3.5)
    assert isinstance(py_f(2.3), type(epyc_f(2.3)))

def test_bad_type_var_context(language):
    T = TypeVar('T', int, float)
    S = TypeVar('T', int, float)

    def f(a : T, b : S) -> T:
        return 2*a

    with pytest.raises(PyccelError):
        epyccel(f, language=language)

def test_class_context(language):
    T = int
    T2 = float
    class A:
        def __init__(self, x : T):
            self._x : T2 = T2(x)

        def times(self, y : T):
            return self._x * y

        def __iadd__(self, y : T):
            self._x += y
            return self

    epyc_A = epyccel(A, language=language)

    a = A(3)
    epyc_a = epyc_A(3)
    assert a.times(2) == epyc_a.times(2)
    assert isinstance(a.times(2), type(epyc_a.times(2)))
    a += 5
    epyc_a += 5
    assert a.times(1) == epyc_a.times(1)
    assert isinstance(a.times(1), type(epyc_a.times(1)))

def test_numpy_cast_context(language):
    T = int
    def f():
        a = np.ones(5, dtype=T)
        return a.sum()

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_container_type_alias_context_1(language):
    T = list[int]

    def f(a : Final[T]):
        return a[0]

    b = [4,5,6]
    epyc_f = epyccel(f, language=language)
    assert f(b) == epyc_f(b)
    assert isinstance(f(b), type(epyc_f(b)))

def test_container_type_alias_context_2(language):
    T = Final[list[int]]

    def f(a : T):
        return a[0]

    a = [3,2,1]
    epyc_f = epyccel(f, language=language)
    assert f(a) == epyc_f(a)
    assert isinstance(f(a), type(epyc_f(a)))
