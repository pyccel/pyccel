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

def test_insert_basic(language):
    def f():
        a = [1, 2, 3]
        a.insert(4, 4)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_insert_multiple(language):
    def f():
        a = [1, 2, 3]
        a.insert(4, 4)
        a.insert(2, 5)
        a.insert(1, 6)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_insert_list(language):
    def f():
        a = [[1, 2, 3]]
        a.insert(1, [4, 5, 6])
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_insert_range(language):
    def f():
        a = [1, 2, 3]
        for i in range(4, 1000):
            a.insert(i - 1 ,i)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_insert_range_list(language):
    def f():
        a = [[1, 2, 3]]
        for i in range(4, 1000):
            a.insert(i, [i, i + 1])
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_insert_range_tuple(language):
    def f():
        a = [[1, 2, 3]]
        for i in range(4, 1000):
            a.insert(i, (i, i + 1))
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_insert_user_defined_objects(language):
    def f():
        class A:
            pass
        a = A()
        b = A()
        c = A()
        d = A()
        e = A()
        lst = [a, b, c]
        lst.insert(0, d)
        lst.insert(1, e)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()
