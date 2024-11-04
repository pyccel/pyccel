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

@pytest.mark.skip("Returning tuples is not yet implemented. See #337")
def test_pop_item(language):
    def pop_item():
        a = {1:1.0, 2:2.0}
        return a.popitem()
    epyc_default_element = epyccel(pop_item, language = language)
    pyccel_result = epyc_default_element()
    python_result = pop_item()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_item_elements(language):
    def pop_item():
        a = {1:1.0, 2:2.0}
        b = a.popitem()
        return b[0], b[1]
    epyc_default_element = epyccel(pop_item, language = language)
    pyccel_result = epyc_default_element()
    python_result = pop_item()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_item_str_keys(language):
    def pop_item_str_keys():
        a = {'a':1, 'b':2}
        b = a.popitem()
        return b[0], b[1]
    epyc_default_element = epyccel(pop_item_str_keys, language = language)
    pyccel_result = epyc_default_element()
    python_result = pop_item_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_item_key(language):
    def pop_item():
        a = {1:1.0, 2:2.0}
        return a.popitem()[0]
    epyc_default_element = epyccel(pop_item, language = language)
    pyccel_result = epyc_default_element()
    python_result = pop_item()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_element(language) :
    def get_element():
        a = {1:1.0, 2:2.0}
        return a.get(1)
    epyc_element = epyccel(get_element, language = language)
    pyccel_result = epyc_element()
    python_result = get_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_default_element(language) :
    def get_default_element():
        a = {1:True, 2:False}
        return a.get(3, True)
    epyc_default_element = epyccel(get_default_element, language = language)
    pyccel_result = epyc_default_element()
    python_result = get_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_str_keys(language) :
    def get_str_keys():
        a = {'a':1, 'b':2}
        return a.get('a')
    epyc_str_keys = epyccel(get_str_keys, language = language)
    pyccel_result = epyc_str_keys()
    python_result = get_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_default_str_keys(language) :
    def get_default_str_keys():
        a = {'a':1, 'b':2}
        return a.get('c', 4)
    epyc_str_keys = epyccel(get_default_str_keys, language = language)
    pyccel_result = epyc_str_keys()
    python_result = get_default_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_dict_clear(language):
    def dict_clear():
        a = {1:1.0, 2:2.0}
        a.clear()
        return a
    epyc_dict_clear = epyccel(dict_clear, language = language)
    pyccel_result = epyc_dict_clear()
    python_result = dict_clear()
    assert python_result == pyccel_result
