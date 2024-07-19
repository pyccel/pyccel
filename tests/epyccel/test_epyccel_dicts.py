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

def test_dict_init(language):
    def dict_init():
        a = {1:1.0, 2:2.0}
        return a
    epyc_dict_init = epyccel(dict_init, language = language)
    pyccel_result = epyc_dict_init()
    python_result = dict_init()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_dict_str_keys(language):
    def dict_str_keys():
        a = {'a':1, 'b':2}
        return a
    epyc_str_keys = epyccel(dict_str_keys, language = language)
    pyccel_result = epyc_str_keys()
    python_result = dict_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_dict_empty_init(language):
    def dict_empty_init():
        a : 'dict[int, float]' = {}
        return a
    epyc_dict_empty_init = epyccel(dict_empty_init, language = language)
    pyccel_result = epyc_dict_empty_init()
    python_result = dict_empty_init()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_dict_copy(language):
    def dict_copy():
        a = {1:1.0,2:2.0}
        b = dict(a)
        return b

    epyc_dict_copy = epyccel(dict_copy, language = language)
    pyccel_result = epyc_dict_copy()
    python_result = dict_copy()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_dict_kwarg_init(language):
    def kwarg_init():
        b = dict(a=1, b=2) #pylint: disable=use-dict-literal
        return b

    epyc_kwarg_init = epyccel(kwarg_init, language = language)
    pyccel_result = epyc_kwarg_init()
    python_result = kwarg_init()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result
