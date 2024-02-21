# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from  pyccel.epyccel import epyccel

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

def test_add_1(language) :
    def add_int():
        a = {1,3,45}
        a.add(4)
        return a
    epyc_add_element = epyccel(add_int, language = language)
    pyccel_result = epyc_add_element()
    python_result = add_int()
    assert python_result == pyccel_result

def test_add_element2(language) :
    def add_complex():
        a = {6j,7j,8j}
        a.add(9j)
        return a
    epyc_add_element = epyccel(add_complex, language = language)
    pyccel_result = epyc_add_element()
    python_result = add_complex()
    assert python_result == pyccel_result

def test_add_element_3(language):
    def add_element_range():
        a = {1, 2, 3}
        for i in range(50, 100):
            a.add(i)
        return a
    epyc_add_element = epyccel(add_element_range, language = language)
    pyccel_result = epyc_add_element()
    python_result = add_element_range()
    assert python_result == pyccel_result
