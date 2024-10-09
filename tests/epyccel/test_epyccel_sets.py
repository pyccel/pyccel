# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from pyccel import epyccel

@pytest.fixture( params=[
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="set methods not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="set methods not implemented in c"),
            pytest.mark.c]),
        pytest.param("python", marks = pytest.mark.python)
    ],
    scope = "module"
)
def python_only_language(request):
    return request.param

def test_add_literal_int(language) :
    def add_int():
        a = {1,3,45}
        a.add(4)
        return len(a)
    epyc_add_element = epyccel(add_int, language = language)
    pyccel_result = epyc_add_element()
    python_result = add_int()
    assert python_result == pyccel_result

def test_add_literal_complex(language) :
    def add_complex():
        a = {6j,7j,8j}
        a.add(9j)
        return len(a)
    epyc_add_element = epyccel(add_complex, language = language)
    pyccel_result = epyc_add_element()
    python_result = add_complex()
    assert python_result == pyccel_result

def test_add_variable_int(language):
    def add_element_range():
        a = {1, 2, 3}
        for i in range(50, 100):
            a.add(i)
        return len(a)
    epyc_add_element = epyccel(add_element_range, language = language)
    pyccel_result = epyc_add_element()
    python_result = add_element_range()
    assert python_result == pyccel_result

def test_clear_int(language):
    def clear_int():
        se = {1,2,4,5}
        se.clear()
        return len(se)
    epyccel_clear = epyccel(clear_int, language = language)
    pyccel_result = epyccel_clear()
    python_result = clear_int()
    assert python_result == pyccel_result

def test_clear_float(language):
    def clear_float():
        se = {7.2, 2.1, 9.8, 6.4}
        se.clear()
        return len(se)
    epyccel_clear = epyccel(clear_float, language = language)
    pyccel_result = epyccel_clear()
    python_result = clear_float()
    assert python_result == pyccel_result

def test_clear_complex(language):
    def clear_complex():
        se = {3j, 6j, 2j}
        se.clear()
        return len(se)
    epyccel_clear = epyccel(clear_complex, language = language)
    pyccel_result = epyccel_clear()
    python_result = clear_complex()
    assert python_result == pyccel_result

def test_copy_int(python_only_language):
    def copy_int():
        se = {1, 2, 4, 5}
        cop = se.copy()
        return cop
    epyccel_copy = epyccel(copy_int, language = python_only_language)
    pyccel_result = epyccel_copy()
    python_result = copy_int()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result
    assert all(isinstance(elem, type(pyccel_result.pop())) for elem in python_result)


def test_copy_float(python_only_language):
    def copy_float():
        se = {5.7, 6.2, 4.3, 9.8}
        cop = se.copy()
        return cop
    epyccel_copy = epyccel(copy_float, language = python_only_language)
    pyccel_result = epyccel_copy()
    python_result = copy_float()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result
    assert all(isinstance(elem, type(pyccel_result.pop())) for elem in python_result)

def test_copy_complex(python_only_language):
    def copy_complex():
        se = {7j, 6j, 9j}
        cop = se.copy()
        return cop
    epyccel_copy = epyccel(copy_complex, language = python_only_language)
    pyccel_result = epyccel_copy()
    python_result = copy_complex()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result
    assert all(isinstance(elem, type(pyccel_result.pop())) for elem in python_result)

def test_remove_complex(python_only_language):
    def remove_complex():
        se = {1j, 3j, 8j}
        se.remove(3j)
        return se
    epyccel_remove = epyccel(remove_complex, language = python_only_language)
    pyccel_result = epyccel_remove()
    python_result = remove_complex()
    assert python_result == pyccel_result

def test_remove_int(python_only_language):
    def remove_int():
        se = {2, 4, 9}
        se.remove(4)
        return se
    epyccel_remove = epyccel(remove_int, language = python_only_language)
    pyccel_result = epyccel_remove()
    python_result = remove_int()
    assert python_result == pyccel_result

def test_remove_float(python_only_language):
    def remove_float():
        se = {5.7, 2.4, 8.1}
        se.remove(8.1)
        return se
    epyccel_remove = epyccel(remove_float, language = python_only_language)
    pyccel_result = epyccel_remove()
    python_result = remove_float()
    assert python_result == pyccel_result

def test_Discard_int(python_only_language):
    def Discard_int():
        se = {2.7, 4.3, 9.2}
        se.discard(4.3)
        return se
    epyccel_remove = epyccel(Discard_int, language = python_only_language)
    pyccel_result = epyccel_remove()
    python_result = Discard_int()
    assert python_result == pyccel_result

def test_Discard_complex(python_only_language):
    def Discard_complex():
        se = {2j, 5j, 3j, 7j}
        se.discard(5j)
        return se
    epyccel_remove = epyccel(Discard_complex, language = python_only_language)
    pyccel_result = epyccel_remove()
    python_result = Discard_complex()
    assert python_result == pyccel_result

def test_Discard_wrong_arg(python_only_language):
    def Discard_wrong_arg():
        se = {4.7, 1.3, 8.2}
        se.discard(8.6)
        return se
    epyccel_remove = epyccel(Discard_wrong_arg, language = python_only_language)
    pyccel_result = epyccel_remove()
    python_result = Discard_wrong_arg()
    assert python_result == pyccel_result

def test_update_basic(python_only_language):
    def update_basic():
        a = {1, 2, 3}
        b = {4, 5, 6}
        a.update(b)
        return a

    epyccel_update = epyccel(update_basic, language=python_only_language)
    pyccel_result = epyccel_update()
    python_result =  update_basic()
    assert python_result == pyccel_result

def test_update_multiple(python_only_language):
    def update_multiple():
        a = {1, 2, 3}
        a.update({4, 5})
        a.update({6, 7, 8, 9})
        a.update({10})
        return a

    epyccel_update = epyccel(update_multiple, language=python_only_language)
    pyccel_result = epyccel_update()
    python_result =  update_multiple()
    assert python_result == pyccel_result


def test_update_boolean_tuple(python_only_language):
    def update_boolean_tuple():
        a = {True}
        b = (False, True, False)
        a.update(b)
        return a
    epyccel_update = epyccel(update_boolean_tuple, language=python_only_language)
    pyccel_result = epyccel_update()
    python_result =  update_boolean_tuple()
    assert python_result == pyccel_result


def test_update_complex_list(python_only_language):
    def update_complex_list():
        a = {1j, 2 + 3j, 0 + 0j}
        b = {4j, 5j, 1 + 6j}
        a.update(b)
        return a
    epyccel_update = epyccel(update_complex_list, language=python_only_language)
    pyccel_result = epyccel_update()
    python_result =  update_complex_list()
    assert python_result == pyccel_result

def test_update_range(python_only_language):
    def update_range():
        a = {1, 2, 3}
        a.update(range(4, 9))
        return a
    epyccel_update = epyccel(update_range, language=python_only_language)
    pyccel_result = epyccel_update()
    python_result =  update_range()
    assert python_result == pyccel_result

def test_update_set_as_arg(python_only_language):
    def update_set_as_arg():
        a = {1, 2, 3}
        a.update({4, 5, 6})
        return a

    epyccel_update = epyccel(update_set_as_arg, language=python_only_language)
    pyccel_result = epyccel_update()
    python_result =  update_set_as_arg()
    assert python_result == pyccel_result

def test_update_tuple_as_arg(python_only_language):
    def update_tuple_as_arg():
        a = {1, 2, 3}
        a.update((4, 5, 6))
        return a
    epyccel_update = epyccel(update_tuple_as_arg, language=python_only_language)
    pyccel_result = epyccel_update()
    python_result =  update_tuple_as_arg()
    assert python_result == pyccel_result

def test_set_with_list(python_only_language):
    def set_With_list():
        a = [1.6, 6.3, 7.2]
        b = set(a)
        return b

    epyc_set_With_list = epyccel(set_With_list, language = python_only_language)
    pyccel_result = epyc_set_With_list()
    python_result = set_With_list()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_set_with_tuple(python_only_language):
    def set_With_tuple():
        a = (1j, 6j, 7j)
        b = set(a)
        return b

    epyc_set_With_tuple = epyccel(set_With_tuple, language = python_only_language)
    pyccel_result = epyc_set_With_tuple()
    python_result = set_With_tuple()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_set_with_set(python_only_language):
    def set_With_set():
        a = {True, False, True}  #pylint: disable=duplicate-value
        b = set(a)
        return b

    epyc_set_With_set = epyccel(set_With_set, language = python_only_language)
    pyccel_result = epyc_set_With_set()
    python_result = set_With_set()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_init_with_set(python_only_language):
    def init_with_set():
        b = set({4.6, 7.9, 2.5})
        return b

    epyc_init_with_set = epyccel(init_with_set, language = python_only_language)
    pyccel_result = epyc_init_with_set()
    python_result = init_with_set()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_set_init_with_list(python_only_language):
    def init_with_list():
        b = set([4.6, 7.9, 2.5])
        return b

    epyc_init_with_list = epyccel(init_with_list, language = python_only_language)
    pyccel_result = epyc_init_with_list()
    python_result = init_with_list()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_set_copy_from_arg1(python_only_language):
    def copy_from_arg1(a : 'list[float]'):
        b = set(a)
        return b
    a = [2.5, 1.4, 9.2]
    epyc_copy_from_arg = epyccel(copy_from_arg1, language = python_only_language)
    pyccel_result = epyc_copy_from_arg(a)
    python_result = copy_from_arg1(a)
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_set_copy_from_arg2(python_only_language):
    def copy_from_arg2(a : 'set[float]'):
        b = set(a)
        return b
    a = {2.5, 1.4, 9.2}
    epyc_copy_from_arg = epyccel(copy_from_arg2, language = python_only_language)
    pyccel_result = epyc_copy_from_arg(a)
    python_result = copy_from_arg2(a)
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_Pop_int(stc_language):
    def Pop_int():
        se = {2, 4, 9}
        el1 = se.pop()
        el2 = se.pop()
        el3 = se.pop()
        return el1, el2, el3
    epyccel_remove = epyccel(Pop_int, language = stc_language)
    pyccel_result = set(epyccel_remove())
    python_result = set(Pop_int())
    assert python_result == pyccel_result

def test_Pop_float(stc_language):
    def Pop_float():
        se = {2.3 , 4.1, 9.5}
        el1 = se.pop()
        el2 = se.pop()
        el3 = se.pop()
        return el1, el2, el3
    epyccel_remove = epyccel(Pop_float, language = stc_language)
    pyccel_result = set(epyccel_remove())
    python_result = set(Pop_float())
    assert python_result == pyccel_result


def test_Pop_complex(stc_language):
    def Pop_complex():
        se = {4j , 1j, 7j}
        el1 = se.pop()
        el2 = se.pop()
        el3 = se.pop()
        return el1, el2, el3
    epyccel_remove = epyccel(Pop_complex, language = stc_language)
    pyccel_result = set(epyccel_remove())
    python_result = set(Pop_complex())
    assert python_result == pyccel_result
