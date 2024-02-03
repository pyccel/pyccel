# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# coding: utf-8
""" Tests for list methods.
"""

import pytest
from pyccel.epyccel import epyccel

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="append not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="append not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_append_basic(language):
    def f():
        a = [1, 2, 3]
        a.append(4)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="append not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="append not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_append_multiple(language):
    def f():
        a = [1, 2, 3]
        a.append(4)
        a.append(5)
        a.append(6)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="append not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="append not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_append_list(language):
    def f():
        a = [1, 2, 3]
        a.append([4, 5, 6])
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="append not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="append not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_append_range(language):
    def f():
        a = [1, 2, 3]
        for i in range(0, 1000):
            a.append(i)
        a.append(1000)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="append not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="append not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_append_range_list(language):
    def f():
        a = [1, 2, 3]
        for i in range(0, 1000):
            a.append([i, i + 1])
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()
