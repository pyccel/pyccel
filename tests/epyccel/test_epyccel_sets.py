# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from  pyccel.epyccel import epyccel

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
def language(request):
    return request.param

def test_add_literal_int(language) :
    def add_int():
        a = {1,3,45}
        a.add(4)
        return a
    epyc_add_element = epyccel(add_int, language = language)
    pyccel_result = epyc_add_element()
    python_result = add_int()
    assert python_result == pyccel_result

def test_add_literal_complex(language) :
    def add_complex():
        a = {6j,7j,8j}
        a.add(9j)
        return a
    epyc_add_element = epyccel(add_complex, language = language)
    pyccel_result = epyc_add_element()
    python_result = add_complex()
    assert python_result == pyccel_result

def test_add_variable_int(language):
    def add_element_range():
        a = {1, 2, 3}
        for i in range(50, 100):
            a.add(i)
        return a
    epyc_add_element = epyccel(add_element_range, language = language)
    pyccel_result = epyc_add_element()
    python_result = add_element_range()
    assert python_result == pyccel_result

def test_clear_int(language):
    def clear_int():
        se = {1,2,4,5}
        se.clear()
        return se
    epyccel_clear = epyccel(clear_int, language = language)
    pyccel_result = epyccel_clear()
    python_result = clear_int()
    assert python_result == pyccel_result

def test_clear_float(language):
    def clear_float():
        se = {7.2, 2.1, 9.8, 6.4}
        se.clear()
        return se
    epyccel_clear = epyccel(clear_float, language = language)
    pyccel_result = epyccel_clear()
    python_result = clear_float()
    assert python_result == pyccel_result

def test_clear_complex(language):
    def clear_complex():
        se = {3j, 6j, 2j}
        se.clear()
        return se
    epyccel_clear = epyccel(clear_complex, language = language)
    pyccel_result = epyccel_clear()
    python_result = clear_complex()
    assert python_result == pyccel_result

def test_copy_int(language):
    def copy_int():
        se = {1, 2, 4, 5}
        cop = se.copy()
        return cop
    epyccel_copy = epyccel(copy_int, language = language)
    pyccel_result = epyccel_copy()
    python_result = copy_int()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result
    assert all(isinstance(elem, type(pyccel_result.pop())) for elem in python_result)


def test_copy_float(language):
    def copy_float():
        se = {5.7, 6.2, 4.3, 9.8}
        cop = se.copy()
        return cop
    epyccel_copy = epyccel(copy_float, language = language)
    pyccel_result = epyccel_copy()
    python_result = copy_float()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result
    assert all(isinstance(elem, type(pyccel_result.pop())) for elem in python_result)

def test_copy_complex(language):
    def copy_complex():
        se = {7j, 6j, 9j}
        cop = se.copy()
        return cop
    epyccel_copy = epyccel(copy_complex, language = language)
    pyccel_result = epyccel_copy()
    python_result = copy_complex()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result
    assert all(isinstance(elem, type(pyccel_result.pop())) for elem in python_result)

def test_Pop_int(language):
    def Pop_int():
        se = {2, 4, 9}
        se.pop()
        return se
    epyccel_remove = epyccel(Pop_int, language = language)
    pyccel_result = epyccel_remove()
    python_result = Pop_int()
    assert python_result == pyccel_result

def test_Pop_float(language):
    def Pop_float():
        se = {2.7, 4.3, 9.2}
        se.pop()
        return se
    epyccel_remove = epyccel(Pop_float, language = language)
    pyccel_result = epyccel_remove()
    python_result = Pop_float()
    assert python_result == pyccel_result

def test_Pop_complex(language):
    def Pop_complex():
        se = {1j, 3j, 6j}
        se.pop()
        return se
    epyccel_remove = epyccel(Pop_complex, language = language)
    pyccel_result = epyccel_remove()
    python_result = Pop_complex()
    assert python_result == pyccel_result

def test_remove_complex(language):
    def remove_complex():
        se = {1j, 3j, 8j}
        se.remove(3j)
        return se
    epyccel_remove = epyccel(remove_complex, language = language)
    pyccel_result = epyccel_remove()
    python_result = remove_complex()
    assert python_result == pyccel_result

def test_remove_int(language):
    def remove_int():
        se = {2, 4, 9}
        se.remove(4)
        return se
    epyccel_remove = epyccel(remove_int, language = language)
    pyccel_result = epyccel_remove()
    python_result = remove_int()
    assert python_result == pyccel_result

def test_remove_float(language):
    def remove_float():
        se = {5.7, 2.4, 8.1}
        se.remove(8.1)
        return se
    epyccel_remove = epyccel(remove_float, language = language)
    pyccel_result = epyccel_remove()
    python_result = remove_float()
    assert python_result == pyccel_result

