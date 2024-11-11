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
def python_only_language(request):
    return request.param

def test_dict_init(python_only_language):
    def dict_init():
        a = {1:1.0, 2:2.0}
        return a
    epyc_dict_init = epyccel(dict_init, language = python_only_language)
    pyccel_result = epyc_dict_init()
    python_result = dict_init()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_dict_str_keys(python_only_language):
    def dict_str_keys():
        a = {'a':1, 'b':2}
        return a
    epyc_str_keys = epyccel(dict_str_keys, language = python_only_language)
    pyccel_result = epyc_str_keys()
    python_result = dict_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_dict_empty_init(python_only_language):
    def dict_empty_init():
        a : 'dict[int, float]' = {}
        return a
    epyc_dict_empty_init = epyccel(dict_empty_init, language = python_only_language)
    pyccel_result = epyc_dict_empty_init()
    python_result = dict_empty_init()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_dict_copy(python_only_language):
    def dict_copy():
        a = {1:1.0,2:2.0}
        b = dict(a)
        return b

    epyc_dict_copy = epyccel(dict_copy, language = python_only_language)
    pyccel_result = epyc_dict_copy()
    python_result = dict_copy()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_dict_kwarg_init(python_only_language):
    def kwarg_init():
        b = dict(a=1, b=2) #pylint: disable=use-dict-literal
        return b

    epyc_kwarg_init = epyccel(kwarg_init, language = python_only_language)
    pyccel_result = epyc_kwarg_init()
    python_result = kwarg_init()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_element(python_only_language):
    def pop_element():
        a = {1:1.0, 2:2.0}
        return a.pop(1)
    epyc_element = epyccel(pop_element, language = python_only_language)
    pyccel_result = epyc_element()
    python_result = pop_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_default_element(python_only_language):
    def pop_default_element():
        a = {1:True, 2:False}
        return a.pop(3, True)
    epyc_default_element = epyccel(pop_default_element, language = python_only_language)
    pyccel_result = epyc_default_element()
    python_result = pop_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_str_keys(python_only_language):
    def pop_str_keys():
        a = {'a':1, 'b':2}
        return a.pop('a')
    epyc_str_keys = epyccel(pop_str_keys, language = python_only_language)
    pyccel_result = epyc_str_keys()
    python_result = pop_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

@pytest.mark.skip("Returning tuples is not yet implemented. See #337")
def test_pop_item(python_only_language):
    def pop_item():
        a = {1:1.0, 2:2.0}
        return a.popitem()
    epyc_default_element = epyccel(pop_item, language = python_only_language)
    pyccel_result = epyc_default_element()
    python_result = pop_item()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_item_elements(python_only_language):
    def pop_item():
        a = {1:1.0, 2:2.0}
        b = a.popitem()
        return b[0], b[1]
    epyc_default_element = epyccel(pop_item, language = python_only_language)
    pyccel_result = epyc_default_element()
    python_result = pop_item()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_item_str_keys(python_only_language):
    def pop_item_str_keys():
        a = {'a':1, 'b':2}
        b = a.popitem()
        return b[0], b[1]
    epyc_default_element = epyccel(pop_item_str_keys, language = python_only_language)
    pyccel_result = epyc_default_element()
    python_result = pop_item_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_item_key(python_only_language):
    def pop_item():
        a = {1:1.0, 2:2.0}
        return a.popitem()[0]
    epyc_default_element = epyccel(pop_item, language = python_only_language)
    pyccel_result = epyc_default_element()
    python_result = pop_item()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_element(python_only_language):
    def get_element():
        a = {1:1.0, 2:2.0}
        return a.get(1)
    epyc_element = epyccel(get_element, language = python_only_language)
    pyccel_result = epyc_element()
    python_result = get_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_default_element(python_only_language):
    def get_default_element():
        a = {1:True, 2:False}
        return a.get(3, True)
    epyc_default_element = epyccel(get_default_element, language = python_only_language)
    pyccel_result = epyc_default_element()
    python_result = get_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_str_keys(python_only_language):
    def get_str_keys():
        a = {'a':1, 'b':2}
        return a.get('a')
    epyc_str_keys = epyccel(get_str_keys, language = python_only_language)
    pyccel_result = epyc_str_keys()
    python_result = get_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_default_str_keys(python_only_language):
    def get_default_str_keys():
        a = {'a':1, 'b':2}
        return a.get('c', 4)
    epyc_str_keys = epyccel(get_default_str_keys, language = python_only_language)
    pyccel_result = epyc_str_keys()
    python_result = get_default_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_dict_contains(language):
    def dict_contains():
        a = {1:1.0, 2:2.0, 3:3.0}
        return (1 in a), (5 in a), (4.0 in a)
    epyc_func = epyccel(dict_contains, language = language)
    pyccel_result = epyc_func()
    python_result = dict_contains()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_set_default(python_only_language):
    def set_default():
        a = {1: 1.0, 2:2.0}
        b = a.setdefault(1, 3.0)
        c = a.setdefault(3, 4.0)
        return a, b, c
    epyc_str_keys = epyccel(set_default, language = python_only_language)
    pyccel_result = epyc_str_keys()
    python_result = set_default()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_dict_ptr(python_only_language):
    def dict_ptr():
        a = {1:1.0, 2:2.0, 3:3.0}
        b = a
        c = b.pop(2)
        return len(a), len(b), c

    epyc_func = epyccel(dict_ptr, language = python_only_language)
    pyccel_result = epyc_func()
    python_result = dict_ptr()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_dict_clear(python_only_language):
    def dict_clear():
        a = {1:1.0, 2:2.0}
        a.clear()
        return a
    epyc_dict_clear = epyccel(dict_clear, language = python_only_language)
    pyccel_result = epyc_dict_clear()
    python_result = dict_clear()
    assert python_result == pyccel_result


def test_dict_copy_method(python_only_language):
    def dict_copy():
        a = {1:1.0, 2:2.0}
        b = a.copy()
        return b
    epyc_dict_copy = epyccel(dict_copy, language = python_only_language)
    pyccel_result = epyc_dict_copy()
    python_result = dict_copy()
    assert python_result == pyccel_result