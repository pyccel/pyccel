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

def test_pop_last_element(language) :
    def pop_last_element():
        a = [1,3,45]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_specific_index(language) :
    def pop_specific_index():
        a = [1j,3j,45j]
        return a.pop(1)
    epyc_specific_index = epyccel(pop_specific_index, language = language)
    python_result = pop_specific_index()
    pyccel_result = epyc_specific_index()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_negative_index(language) :
    def pop_negative_index():
        a = [1j,3j,45j]
        return a.pop(-1)
    epyc_negative_index = epyccel(pop_negative_index, language = language)
    python_result = pop_negative_index()
    pyccel_result = epyc_negative_index()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_2(language) :
    def pop_2():
        a = [1.7,2.7,45.0]
        a.pop()
        return a.pop(-1)
    pop_2_epyc = epyccel(pop_2, language = language)
    python_result = pop_2()
    pyccel_result = pop_2_epyc()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

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
    import modules.list_user_defined_objs as mod

    modnew = epyccel(mod, language=language)
    python_list = mod.fn()
    accelerated_list = modnew.fn()
    assert len(python_list) == len(accelerated_list)
    for python_elem, accelerated_elem in zip(python_list, accelerated_list):
        assert python_elem.x == accelerated_elem.x

def test_clear_1(language):

    def clear_1():
        a = [1, 2, 3]
        a.clear()
        return a

    epyc_clear_1 = epyccel(clear_1, language = language)
    pyccel_result = epyc_clear_1()
    python_result = clear_1()
    assert python_result == pyccel_result

def test_clear_2(language):

    def clear_2():
        a = []
        a.clear()
        return a

    epyc_clear_2 = epyccel(clear_2, language = language)
    pyccel_result = epyc_clear_2()
    python_result = clear_2()
    assert python_result == pyccel_result

def test_clear_3(language):

    def clear_3():
        a = [[1, 2, 3]]
        a.clear()
        return a

    epyc_clear_3 = epyccel(clear_3, language = language)
    pyccel_result = epyc_clear_3()
    python_result = clear_3()
    assert python_result == pyccel_result
