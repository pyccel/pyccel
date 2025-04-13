# pylint: disable=missing-function-docstring, missing-module-docstring
import sys
from typing import TypeVar
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
        epyc_f = epyccel(f, language=language)
