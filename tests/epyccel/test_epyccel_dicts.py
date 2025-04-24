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

def test_dict_init(language):
    def dict_init():
        a = {1:1.0, 2:2.0}
        return a
    epyc_dict_init = epyccel(dict_init, language = language)
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

def test_dict_empty_init(language):
    def dict_empty_init():
        a : 'dict[int, float]' = {}
        return a
    epyc_dict_empty_init = epyccel(dict_empty_init, language = language)
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

def test_pop_element(language):
    def pop_element():
        a = {1:1.0, 2:2.0}
        return a.pop(1)
    epyc_element = epyccel(pop_element, language = language)
    pyccel_result = epyc_element()
    python_result = pop_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_default_element(language):
    def pop_default_element():
        a = {1:True, 2:False}
        return a.pop(3, True)
    epyc_default_element = epyccel(pop_default_element, language = language)
    pyccel_result = epyc_default_element()
    python_result = pop_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_bool_keys(language):
    def pop_default_element():
        a = {True:1, False:2}
        return a.pop(False)
    epyc_default_element = epyccel(pop_default_element, language = language)
    pyccel_result = epyc_default_element()
    python_result = pop_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_falsy_int_default_element(language):
    def pop_falsy_int_default_element():
        a = {1:2, 2:3}
        return a.pop(3, 0)
    epyc_func = epyccel(pop_falsy_int_default_element, language = language)
    pyccel_result = epyc_func()
    python_result = pop_falsy_int_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_falsy_bool_default_element(language):
    def pop_falsy_bool_default_element():
        a = {1:True, 2:False}
        return a.pop(3, False)
    epyc_default_element = epyccel(pop_falsy_bool_default_element, language = language)
    pyccel_result = epyc_default_element()
    python_result = pop_falsy_bool_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_str_keys(stc_language):
    def pop_str_keys():
        a = {'a':1, 'b':2}
        return a.pop('a')
    epyc_str_keys = epyccel(pop_str_keys, language = stc_language)
    pyccel_result = epyc_str_keys()
    python_result = pop_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_non_literal_str_keys(stc_language):
    def pop_str_keys():
        a = {'a':1, 'b':2}
        my_str = 'a'
        return a.pop(my_str)
    epyc_str_keys = epyccel(pop_str_keys, language = stc_language)
    pyccel_result = epyc_str_keys()
    python_result = pop_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_item(language):
    def pop_item():
        a = {1:1.0, 2:2.0}
        return a.popitem()

    original_dict = {1:1.0, 2:2.0}
    epyc_default_element = epyccel(pop_item, language = language)
    pyccel_result = epyc_default_element()
    assert pyccel_result[0] in original_dict
    assert pyccel_result[1] == original_dict[pyccel_result[0]]

def test_pop_item_elements(language):
    def pop_item():
        a = {1:1.0, 2:2.0}
        b = a.popitem()
        return b[0], b[1]

    original_dict = {1:1.0, 2:2.0}
    epyc_default_element = epyccel(pop_item, language = language)
    pyccel_result = epyc_default_element()
    assert pyccel_result[0] in original_dict
    assert pyccel_result[1] == original_dict[pyccel_result[0]]

def test_pop_item_str_keys(stc_language):
    def pop_item_str_keys():
        a = {'a':1, 'b':2}
        b = a.popitem()
        return b[0], b[1]

    original_dict = {'a':1, 'b':2}
    epyc_default_element = epyccel(pop_item_str_keys, language = stc_language)
    pyccel_result = epyc_default_element()
    assert pyccel_result[0] in original_dict
    assert pyccel_result[1] == original_dict[pyccel_result[0]]

def test_pop_item_key(language):
    def pop_item():
        a = {1:1.0, 2:2.0}
        return a.popitem()[0]

    original_dict = {1:1.0, 2:2.0}
    epyc_default_element = epyccel(pop_item, language = language)
    pyccel_result = epyc_default_element()
    assert pyccel_result in original_dict

def test_pop_item_expression(language):
    def pop_item():
        a = {1:1.0, 2:2.0}
        return a.popitem()[0] + 4

    possible_results = {5, 6}
    epyc_default_element = epyccel(pop_item, language = language)
    pyccel_result = epyc_default_element()
    assert pyccel_result in possible_results

def test_pop_item_unpacking(language):
    def pop_item():
        a = {1:1.0, 2:2.0}
        b, c = a.popitem()
        return b, c

    original_dict = {1:1.0, 2:2.0}
    epyc_default_element = epyccel(pop_item, language = language)
    pyccel_result = epyc_default_element()
    assert pyccel_result[0] in original_dict
    assert pyccel_result[1] == original_dict[pyccel_result[0]]

def test_get_element(python_only_language):
    def get_element():
        a = {1:1.0, 2:2.0}
        return a.get(1)
    epyc_element = epyccel(get_element, language = python_only_language)
    pyccel_result = epyc_element()
    python_result = get_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_default_element(stc_language):
    def get_default_element():
        a = {1:True, 2:False}
        return a.get(3, True)
    epyc_default_element = epyccel(get_default_element, language = stc_language)
    pyccel_result = epyc_default_element()
    python_result = get_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_array(python_only_language):
    def get_array():
        import numpy as np
        a = {1:np.ones(6), 2:np.zeros(4)}
        tmp = a.get(1)
        return tmp[1]
    epyc_array = epyccel(get_array, language = python_only_language)
    pyccel_result = epyc_array()
    python_result = get_array()
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

def test_get_default_str_keys(stc_language):
    def get_default_str_keys():
        a = {'a':1, 'b':2}
        return a.get('c', 4)
    epyc_str_keys = epyccel(get_default_str_keys, language = stc_language)
    pyccel_result = epyc_str_keys()
    python_result = get_default_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_falsy_int_default_element(stc_language):
    def get_falsy_int_default_element():
        a = {1:2, 2:3}
        return a.get(3, 0)
    epyc_func = epyccel(get_falsy_int_default_element, language = stc_language)
    pyccel_result = epyc_func()
    python_result = get_falsy_int_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_get_falsy_bool_default_element(stc_language):
    def get_falsy_bool_default_element():
        a = {1:True, 2:False}
        return a.get(3, False)
    epyc_func = epyccel(get_falsy_bool_default_element, language = stc_language)
    pyccel_result = epyc_func()
    python_result = get_falsy_bool_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_getitem_element(language):
    def getitem_element():
        a = {1:1.0, 2:2.0}
        return a[1]
    epyc_element = epyccel(getitem_element, language = language)
    pyccel_result = epyc_element()
    python_result = getitem_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_getitem_str_keys(stc_language):
    def getitem_str_keys():
        a = {'a':1, 'b':2}
        return a['a']
    epyc_str_keys = epyccel(getitem_str_keys, language = stc_language)
    pyccel_result = epyc_str_keys()
    python_result = getitem_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_getitem_array_element(python_only_language):
    def getitem_array_element():
        import numpy as np
        a = {1:np.ones(6), 2:np.zeros(4)}
        tmp = a[1]
        return tmp[2]
    epyc_array_element = epyccel(getitem_array_element, language = python_only_language)
    pyccel_result = epyc_array_element()
    python_result = getitem_array_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_getitem_modify_element(language):
    def getitem_modify_element():
        a = {1:1.0, 2:2.0}
        a[1] = 3.0
        return a[1]
    epyc_modify_element = epyccel(getitem_modify_element, language = language)
    pyccel_result = epyc_modify_element()
    python_result = getitem_modify_element()
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

def test_dict_clear(language):
    def dict_clear():
        a = {1:1.0, 2:2.0}
        a.clear()
        return len(a)
    epyc_dict_clear = epyccel(dict_clear, language = language)
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

def test_dict_items(language):
    def dict_items():
        a = {1:1.0, 2:2.0, 3:3.0, 5:4.7}
        key_sum = 0
        val_sum = 0.0
        for key, val in a.items():
            key_sum += key
            val_sum += val

        return key_sum, val_sum

    epyc_dict_items = epyccel(dict_items, language = language)
    pyccel_result = epyc_dict_items()
    python_result = dict_items()
    assert python_result == pyccel_result
    assert isinstance(python_result[0], type(pyccel_result[0]))
    assert isinstance(python_result[1], type(pyccel_result[1]))

def test_dict_keys(language):
    def dict_keys():
        a = {1:1.0, 2:2.0, 3:3.0, 5:4.7}
        key_sum = 0
        for key in a.keys(): #pylint:disable=consider-iterating-dictionary
            key_sum += key

        return key_sum

    epyc_dict_keys = epyccel(dict_keys, language = language)
    pyccel_result = epyc_dict_keys()
    python_result = dict_keys()
    assert python_result == pyccel_result

def test_dict_keys_iter(language):
    def dict_keys():
        a = {1:1.0, 2:2.0, 3:3.0, 5:4.7}
        key_sum = 0
        for key in a:
            key_sum += key

        return key_sum

    epyc_dict_keys = epyccel(dict_keys, language = language)
    pyccel_result = epyc_dict_keys()
    python_result = dict_keys()
    assert python_result == pyccel_result

def test_dict_values(language):
    def dict_values():
        a = {1:1.0, 2:2.0, 3:3.0, 5:4.7}
        value_sum = 0.0
        for value in a.values(): #pylint:disable=consider-iterating-dictionary
            value_sum += value

        return value_sum

    epyc_dict_values = epyccel(dict_values, language = language)
    pyccel_result = epyc_dict_values()
    python_result = dict_values()
    assert python_result == pyccel_result
