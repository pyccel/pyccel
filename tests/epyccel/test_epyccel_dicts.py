# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from pyccel import epyccel

@pytest.fixture( params=[
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="dict methods not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="dict methods not implemented in c"),
            pytest.mark.c]),
        pytest.param("python", marks = pytest.mark.python)
    ],
    scope = "module"
)
def language(request):
    return request.param

def test_pop_element(language) :
    def pop_element():
        a = {1:1.0, 2:2.0}
        return a.pop(1)
    epyc_element = epyccel(pop_element, language = language)
    pyccel_result = epyc_element()
    python_result = pop_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_default_element(language) :
    def pop_default_element():
        a = {1:True, 2:False}
        return a.pop(3, True)
    epyc_default_element = epyccel(pop_default_element, language = language)
    pyccel_result = epyc_default_element()
    python_result = pop_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_str_keys(language) :
    def pop_str_keys():
        a = {'a':1, 'b':2}
        return a.pop('a')
    epyc_str_keys = epyccel(pop_str_keys, language = language)
    pyccel_result = epyc_str_keys()
    python_result = pop_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result
