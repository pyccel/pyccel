# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# coding: utf-8
""" Tests for list methods.
"""

import pytest
from pyccel.epyccel import epyccel

@pytest.fixture( params=[
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="list methods not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="list methods not implemented in c"),
            pytest.mark.c]),
        pytest.param("python", marks = pytest.mark.python)
    ],
    scope = "module"
)
def language(request):
    return request.param

def test_append_basic(language):
    def f():
        a = [1, 2, 3]
        a.append(4)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_multiple(language):
    def f():
        a = [1, 2, 3]
        a.append(4)
        a.append(5)
        a.append(6)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_list(language):
    def f():
        a = [[1, 2, 3]]
        a.append([4, 5, 6])
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_range(language):
    def f():
        a = [1, 2, 3]
        for i in range(0, 1000):
            a.append(i)
        a.append(1000)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_range_list(language):
    def f():
        a = [[1, 2, 3]]
        for i in range(0, 1000):
            a.append([i, i + 1])
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_range_tuple(language):
    def f():
        a = [[1, 2, 3]]
        for i in range(0, 1000):
            a.append((i, i + 1))
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="insert() not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="insert() not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_insert_basic(language):
    def f():
        a = [1, 2, 3]
        a.insert(4, 4)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="insert() not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="insert() not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_insert_multiple(language):
    def f():
        a = [1, 2, 3]
        a.insert(4, 4)
        a.insert(5, 5)
        a.insert(6, 6)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="insert() not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="insert() not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_insert_list(language):
    def f():
        a = [[1, 2, 3]]
        a.insert(2, [4, 5, 6])
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="insert() not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="insert() not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_insert_range(language):
    def f():
        a = [1, 2, 3]
        for i in range(4, 1000):
            a.insert(i ,i)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="insert() not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="insert() not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_insert_range_list(language):
    def f():
        a = [[1, 2, 3]]
        for i in range(4, 1000):
            a.insert(i, [i, i + 1])
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="insert() not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="insert() not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_insert_range_tuple(language):
    def f():
        a = [[1, 2, 3]]
        for i in range(4, 1000):
            a.insert(i, (i, i + 1))
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()
