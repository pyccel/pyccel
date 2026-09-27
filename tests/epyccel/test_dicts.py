# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from modules import dicts

from pyccel import epyccel

from epyccel_utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_dicts_mod(language):
    return epyccel_module_with_fallback(dicts, language)


@pytest.fixture(
    params=[
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.skip(reason="dict methods not implemented in fortran"),
                pytest.mark.fortran,
            ],
        ),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="dict methods not implemented in c"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ],
    scope="module",
)
def python_only_language(request):
    return request.param


def test_dict_init(epyc_dicts_mod):
    dict_init = dicts.dict_init
    epyc_dict_init = epyc_dicts_mod.dict_init
    pyccel_result = epyc_dict_init()
    python_result = dict_init()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_dict_str_keys(python_only_language):
    def dict_str_keys():
        a = {"a": 1, "b": 2}
        return a

    epyc_str_keys = epyccel(dict_str_keys, language=python_only_language)
    pyccel_result = epyc_str_keys()
    python_result = dict_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_dict_empty_init(epyc_dicts_mod):
    dict_empty_init = dicts.dict_empty_init
    epyc_dict_empty_init = epyc_dicts_mod.dict_empty_init
    pyccel_result = epyc_dict_empty_init()
    python_result = dict_empty_init()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_dict_copy(python_only_language):
    def dict_copy():
        a = {1: 1.0, 2: 2.0}
        b = dict(a)
        return b

    epyc_dict_copy = epyccel(dict_copy, language=python_only_language)
    pyccel_result = epyc_dict_copy()
    python_result = dict_copy()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_dict_kwarg_init(python_only_language):
    def kwarg_init():
        b = dict(a=1, b=2)  # pylint: disable=use-dict-literal
        return b

    epyc_kwarg_init = epyccel(kwarg_init, language=python_only_language)
    pyccel_result = epyc_kwarg_init()
    python_result = kwarg_init()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_element(epyc_dicts_mod):
    pop_element = dicts.pop_element
    epyc_element = epyc_dicts_mod.pop_element
    pyccel_result = epyc_element()
    python_result = pop_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_default_element(epyc_dicts_mod):
    pop_default_element = dicts.pop_default_element
    epyc_default_element = epyc_dicts_mod.pop_default_element
    pyccel_result = epyc_default_element()
    python_result = pop_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_bool_keys(epyc_dicts_mod):
    pop_default_element = dicts.pop_bool_keys
    epyc_default_element = epyc_dicts_mod.pop_bool_keys
    pyccel_result = epyc_default_element()
    python_result = pop_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_falsy_int_default_element(epyc_dicts_mod):
    pop_falsy_int_default_element = dicts.pop_falsy_int_default_element
    epyc_func = epyc_dicts_mod.pop_falsy_int_default_element
    pyccel_result = epyc_func()
    python_result = pop_falsy_int_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_falsy_bool_default_element(epyc_dicts_mod):
    pop_falsy_bool_default_element = dicts.pop_falsy_bool_default_element
    epyc_default_element = epyc_dicts_mod.pop_falsy_bool_default_element
    pyccel_result = epyc_default_element()
    python_result = pop_falsy_bool_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_str_keys(stc_language):
    def pop_str_keys():
        a = {"a": 1, "b": 2}
        return a.pop("a")

    epyc_str_keys = epyccel(pop_str_keys, language=stc_language)
    pyccel_result = epyc_str_keys()
    python_result = pop_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_non_literal_str_keys(stc_language):
    def pop_str_keys():
        a = {"a": 1, "b": 2}
        my_str = "a"
        return a.pop(my_str)

    epyc_str_keys = epyccel(pop_str_keys, language=stc_language)
    pyccel_result = epyc_str_keys()
    python_result = pop_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_item(epyc_dicts_mod):
    original_dict = {1: 1.0, 2: 2.0}
    epyc_default_element = epyc_dicts_mod.pop_item
    pyccel_result = epyc_default_element()
    assert pyccel_result[0] in original_dict
    assert pyccel_result[1] == original_dict[pyccel_result[0]]


def test_pop_item_elements(epyc_dicts_mod):
    original_dict = {1: 1.0, 2: 2.0}
    epyc_default_element = epyc_dicts_mod.pop_item_elements
    pyccel_result = epyc_default_element()
    assert pyccel_result[0] in original_dict
    assert pyccel_result[1] == original_dict[pyccel_result[0]]


def test_pop_item_str_keys(stc_language):
    def pop_item_str_keys():
        a = {"a": 1, "b": 2}
        b = a.popitem()
        return b[0], b[1]

    original_dict = {"a": 1, "b": 2}
    epyc_default_element = epyccel(pop_item_str_keys, language=stc_language)
    pyccel_result = epyc_default_element()
    assert pyccel_result[0] in original_dict
    assert pyccel_result[1] == original_dict[pyccel_result[0]]


def test_pop_item_key(epyc_dicts_mod):
    original_dict = {1: 1.0, 2: 2.0}
    epyc_default_element = epyc_dicts_mod.pop_item_key
    pyccel_result = epyc_default_element()
    assert pyccel_result in original_dict


def test_pop_item_expression(epyc_dicts_mod):
    possible_results = {5, 6}
    epyc_default_element = epyc_dicts_mod.pop_item_expression
    pyccel_result = epyc_default_element()
    assert pyccel_result in possible_results


def test_pop_item_unpacking(epyc_dicts_mod):
    original_dict = {1: 1.0, 2: 2.0}
    epyc_default_element = epyc_dicts_mod.pop_item_unpacking
    pyccel_result = epyc_default_element()
    assert pyccel_result[0] in original_dict
    assert pyccel_result[1] == original_dict[pyccel_result[0]]


def test_get_element(python_only_language):
    def get_element():
        a = {1: 1.0, 2: 2.0}
        return a.get(1)

    epyc_element = epyccel(get_element, language=python_only_language)
    pyccel_result = epyc_element()
    python_result = get_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_get_default_element(stc_language):
    def get_default_element():
        a = {1: True, 2: False}
        return a.get(3, True)

    epyc_default_element = epyccel(get_default_element, language=stc_language)
    pyccel_result = epyc_default_element()
    python_result = get_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_get_array(python_only_language):
    def get_array():
        import numpy as np

        a = {1: np.ones(6), 2: np.zeros(4)}
        tmp = a.get(1)
        return tmp[1]

    epyc_array = epyccel(get_array, language=python_only_language)
    pyccel_result = epyc_array()
    python_result = get_array()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_get_str_keys(python_only_language):
    def get_str_keys():
        a = {"a": 1, "b": 2}
        return a.get("a")

    epyc_str_keys = epyccel(get_str_keys, language=python_only_language)
    pyccel_result = epyc_str_keys()
    python_result = get_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_get_default_str_keys(stc_language):
    def get_default_str_keys():
        a = {"a": 1, "b": 2}
        return a.get("c", 4)

    epyc_str_keys = epyccel(get_default_str_keys, language=stc_language)
    pyccel_result = epyc_str_keys()
    python_result = get_default_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_get_falsy_int_default_element(stc_language):
    def get_falsy_int_default_element():
        a = {1: 2, 2: 3}
        return a.get(3, 0)

    epyc_func = epyccel(get_falsy_int_default_element, language=stc_language)
    pyccel_result = epyc_func()
    python_result = get_falsy_int_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_get_falsy_bool_default_element(stc_language):
    def get_falsy_bool_default_element():
        a = {1: True, 2: False}
        return a.get(3, False)

    epyc_func = epyccel(get_falsy_bool_default_element, language=stc_language)
    pyccel_result = epyc_func()
    python_result = get_falsy_bool_default_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_getitem_element(epyc_dicts_mod):
    getitem_element = dicts.getitem_element
    epyc_element = epyc_dicts_mod.getitem_element
    pyccel_result = epyc_element()
    python_result = getitem_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_getitem_str_keys(stc_language):
    def getitem_str_keys():
        a = {"a": 1, "b": 2}
        return a["a"]

    epyc_str_keys = epyccel(getitem_str_keys, language=stc_language)
    pyccel_result = epyc_str_keys()
    python_result = getitem_str_keys()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_getitem_array_element(python_only_language):
    def getitem_array_element():
        import numpy as np

        a = {1: np.ones(6), 2: np.zeros(4)}
        tmp = a[1]
        return tmp[2]

    epyc_array_element = epyccel(getitem_array_element, language=python_only_language)
    pyccel_result = epyc_array_element()
    python_result = getitem_array_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_getitem_modify_element(epyc_dicts_mod):
    getitem_modify_element = dicts.getitem_modify_element
    epyc_modify_element = epyc_dicts_mod.getitem_modify_element
    pyccel_result = epyc_modify_element()
    python_result = getitem_modify_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_dict_contains(epyc_dicts_mod):
    dict_contains = dicts.dict_contains
    epyc_func = epyc_dicts_mod.dict_contains
    pyccel_result = epyc_func()
    python_result = dict_contains()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_set_default(python_only_language):
    def set_default():
        a = {1: 1.0, 2: 2.0}
        b = a.setdefault(1, 3.0)
        c = a.setdefault(3, 4.0)
        return a, b, c

    epyc_str_keys = epyccel(set_default, language=python_only_language)
    pyccel_result = epyc_str_keys()
    python_result = set_default()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_dict_ptr(python_only_language):
    def dict_ptr():
        a = {1: 1.0, 2: 2.0, 3: 3.0}
        b = a
        c = b.pop(2)
        return len(a), len(b), c

    epyc_func = epyccel(dict_ptr, language=python_only_language)
    pyccel_result = epyc_func()
    python_result = dict_ptr()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_dict_clear(epyc_dicts_mod):
    dict_clear = dicts.dict_clear
    epyc_dict_clear = epyc_dicts_mod.dict_clear
    pyccel_result = epyc_dict_clear()
    python_result = dict_clear()
    assert python_result == pyccel_result


def test_dict_copy_method(python_only_language):
    def dict_copy():
        a = {1: 1.0, 2: 2.0}
        b = a.copy()
        return b

    epyc_dict_copy = epyccel(dict_copy, language=python_only_language)
    pyccel_result = epyc_dict_copy()
    python_result = dict_copy()
    assert python_result == pyccel_result


def test_dict_items(epyc_dicts_mod):
    dict_items = dicts.dict_items
    epyc_dict_items = epyc_dicts_mod.dict_items
    pyccel_result = epyc_dict_items()
    python_result = dict_items()
    assert python_result == pyccel_result
    assert isinstance(python_result[0], type(pyccel_result[0]))
    assert isinstance(python_result[1], type(pyccel_result[1]))


def test_dict_keys(epyc_dicts_mod):
    dict_keys = dicts.dict_keys
    epyc_dict_keys = epyc_dicts_mod.dict_keys
    pyccel_result = epyc_dict_keys()
    python_result = dict_keys()
    assert python_result == pyccel_result


def test_dict_keys_iter(epyc_dicts_mod):
    dict_keys = dicts.dict_keys_iter
    epyc_dict_keys = epyc_dicts_mod.dict_keys_iter
    pyccel_result = epyc_dict_keys()
    python_result = dict_keys()
    assert python_result == pyccel_result


def test_dict_values(epyc_dicts_mod):
    dict_values = dicts.dict_values
    epyc_dict_values = epyc_dicts_mod.dict_values
    pyccel_result = epyc_dict_values()
    python_result = dict_values()
    assert python_result == pyccel_result
