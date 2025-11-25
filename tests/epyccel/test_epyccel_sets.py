# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar, Final
import pytest
from pyccel import epyccel

@pytest.fixture( params=[
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="set method not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="set method not implemented in c"),
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

def test_copy_int(language):
    def copy_int():
        se = {1, 2, 4, 5}
        cop = se.copy()
        size = len(cop)
        a,b,c,d = cop.pop(), cop.pop(), cop.pop(), cop.pop()
        return size, len(se), a,b,c,d
    epyccel_copy = epyccel(copy_int, language = language)
    pyccel_result = epyccel_copy()
    python_result = copy_int()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result[0] == pyccel_result[0]
    assert python_result[1] == pyccel_result[1]
    assert set(python_result[2:]) == set(pyccel_result[2:])


def test_copy_float(language):
    def copy_float():
        se = {5.7, 6.2, 4.3, 9.8}
        cop = se.copy()
        return len(cop), cop.pop(), cop.pop(), cop.pop(), cop.pop(), len(se)
    epyccel_copy = epyccel(copy_float, language = language)
    pyccel_result = epyccel_copy()
    python_result = copy_float()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result[0] == pyccel_result[0]
    assert python_result[-1] == pyccel_result[-1]
    assert set(python_result[1:-1]) == set(pyccel_result[1:-1])

def test_copy_complex(language):
    def copy_complex():
        se = {7j, 6j, 9j}
        cop = se.copy()
        return len(cop), cop.pop(), cop.pop(), cop.pop()
    epyccel_copy = epyccel(copy_complex, language = language)
    pyccel_result = epyccel_copy()
    python_result = copy_complex()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

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

def test_Discard_int(language):
    def Discard_int():
        se = {2.7, 4.3, 9.2}
        se.discard(4.3)
        return se
    epyccel_remove = epyccel(Discard_int, language = language)
    pyccel_result = epyccel_remove()
    python_result = Discard_int()
    assert python_result == pyccel_result

def test_Discard_complex(language):
    def Discard_complex():
        se = {2j, 5j, 3j, 7j}
        se.discard(5j)
        return se
    epyccel_remove = epyccel(Discard_complex, language = language)
    pyccel_result = epyccel_remove()
    python_result = Discard_complex()
    assert python_result == pyccel_result

def test_Discard_wrong_arg(language):
    def Discard_wrong_arg():
        se = {4.7, 1.3, 8.2}
        se.discard(8.6)
        return se
    epyccel_remove = epyccel(Discard_wrong_arg, language = language)
    pyccel_result = epyccel_remove()
    python_result = Discard_wrong_arg()
    assert python_result == pyccel_result

def test_update_basic(language):
    def update_basic():
        a = {1, 2, 3}
        b = {4, 5, 6}
        a.update(b)
        return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()

    epyccel_update = epyccel(update_basic, language=language)
    pyccel_result = epyccel_update()
    python_result =  update_basic()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_update_multiple(language):
    def update_multiple():
        a = {1, 2, 3}
        a.update({4, 5})
        a.update({6, 7, 8, 9})
        a.update({10})
        return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()

    epyccel_update = epyccel(update_multiple, language=language)
    pyccel_result = epyccel_update()
    python_result =  update_multiple()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_update_multiple_args(language):
    def update_multiple():
        a = {1, 2, 3}
        a.update({4, 5}, {6, 7, 8, 9}, {10})
        return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()

    epyccel_update = epyccel(update_multiple, language=language)
    pyccel_result = epyccel_update()
    python_result =  update_multiple()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_update_boolean_tuple(language):
    def update_boolean_tuple():
        a = {True}
        b = (False, True, False)
        a.update(b)
        return len(a), a.pop(), a.pop()
    epyccel_update = epyccel(update_boolean_tuple, language=language)
    pyccel_result = epyccel_update()
    python_result =  update_boolean_tuple()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_update_complex_list(language):
    def update_complex_list():
        a = {1j, 2 + 3j, 0 + 0j}
        b = [4j, 5j, 1 + 6j]
        a.update(b)
        return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()
    epyccel_update = epyccel(update_complex_list, language=language)
    pyccel_result = epyccel_update()
    python_result =  update_complex_list()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_update_range(language):
    def update_range():
        a = {1, 2, 3}
        a.update(range(4, 9))
        return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()
    epyccel_update = epyccel(update_range, language=language)
    pyccel_result = epyccel_update()
    python_result =  update_range()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_update_set_as_arg(language):
    def update_set_as_arg():
        a = {1, 2, 3}
        a.update({4, 5, 6})
        return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()

    epyccel_update = epyccel(update_set_as_arg, language=language)
    pyccel_result = epyccel_update()
    python_result =  update_set_as_arg()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_update_tuple_as_arg(language):
    def update_tuple_as_arg():
        a = {1, 2, 3}
        a.update((4, 5, 6))
        return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()
    epyccel_update = epyccel(update_tuple_as_arg, language=language)
    pyccel_result = epyccel_update()
    python_result =  update_tuple_as_arg()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_with_list(language):
    def set_With_list():
        a = [1.6, 6.3, 7.2]
        b = set(a)
        return b

    epyc_set_With_list = epyccel(set_With_list, language = language)
    pyccel_result = epyc_set_With_list()
    python_result = set_With_list()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_set_with_tuple(language):
    def set_With_tuple():
        a = (1j, 6j, 7j)
        b = set(a)
        return b

    epyc_set_With_tuple = epyccel(set_With_tuple, language = language)
    pyccel_result = epyc_set_With_tuple()
    python_result = set_With_tuple()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_set_with_set(language):
    def set_With_set():
        a = {True, False, True}  #pylint: disable=duplicate-value
        b = set(a)
        return b

    epyc_set_With_set = epyccel(set_With_set, language = language)
    pyccel_result = epyc_set_With_set()
    python_result = set_With_set()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_init_with_set(language):
    def init_with_set():
        b = set({4.6, 7.9, 2.5})
        return len(b), b.pop(), b.pop(), b.pop()

    epyc_init_with_set = epyccel(init_with_set, language = language)
    pyccel_result = epyc_init_with_set()
    python_result = init_with_set()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_init_with_list(language):
    def init_with_list():
        b = set([4.6, 7.9, 2.5])
        return len(b), b.pop(), b.pop(), b.pop()

    epyc_init_with_list = epyccel(init_with_list, language = language)
    pyccel_result = epyc_init_with_list()
    python_result = init_with_list()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


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

def test_set_copy_from_arg2(language):
    def copy_from_arg2(a : 'set[float]'):
        b = set(a)
        return b
    a = {2.5, 1.4, 9.2}
    epyc_copy_from_arg = epyccel(copy_from_arg2, language = language)
    pyccel_result = epyc_copy_from_arg(a)
    python_result = copy_from_arg2(a)
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_Pop_int(language):
    def Pop_int():
        se = {2, 4, 9}
        el1 = se.pop()
        el2 = se.pop()
        el3 = se.pop()
        return el1, el2, el3
    epyccel_remove = epyccel(Pop_int, language = language)
    pyccel_result = set(epyccel_remove())
    python_result = set(Pop_int())
    assert python_result == pyccel_result

def test_Pop_float(language):
    def Pop_float():
        se = {2.3 , 4.1, 9.5}
        el1 = se.pop()
        el2 = se.pop()
        el3 = se.pop()
        return el1, el2, el3
    epyccel_remove = epyccel(Pop_float, language = language)
    pyccel_result = set(epyccel_remove())
    python_result = set(Pop_float())
    assert python_result == pyccel_result


def test_Pop_complex(language):
    def Pop_complex():
        se = {4j , 1j, 7j}
        el1 = se.pop()
        el2 = se.pop()
        el3 = se.pop()
        return el1, el2, el3
    epyccel_remove = epyccel(Pop_complex, language = language)
    pyccel_result = set(epyccel_remove())
    python_result = set(Pop_complex())
    assert python_result == pyccel_result

def test_set_union_int(language):
    def union_int():
        a = {1,2,3,4}
        b = {5,6,7,2}
        c = a.union(b)
        return a, b, c

    epyccel_func = epyccel(union_int, language = language)
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result == pyccel_result

def test_set_union_no_args(language):
    def union_int():
        a = {1,2,3,4}
        c = a.union()
        a.add(5)
        return len(c), c.pop(), c.pop(), c.pop(), c.pop()

    epyccel_func = epyccel(union_int, language = language)
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_union_2_args(language):
    def union_int():
        a = {1,2,3,4}
        b = {5,6,7}
        c = {8,9,10,4}
        d = a.union(b, c)
        return len(d), d.pop(), d.pop(), d.pop(), d.pop(), d.pop(), d.pop(), d.pop(), d.pop(), d.pop(), d.pop()

    epyccel_func = epyccel(union_int, language = language)
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_union_temporaries(language):
    def union_int():
        c = {1,2,3,4}.union({5,6,7,2})
        return len(c), c.pop(), c.pop(), c.pop(), c.pop(), c.pop(), c.pop(), c.pop()

    epyccel_func = epyccel(union_int, language = language)
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Can't use a pointer to a temporary object."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_temporary_set_union(language):
    def union_int():
        a = {1,2}
        b = {2}
        d = a.union(b).pop()
        return d

    epyccel_func = epyccel(union_int, language = language)
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result == pyccel_result

def test_temporary_set_union_2(language):
    def union_int():
        a = [1,2]
        b = {2}
        d = set(a).union(b)
        return d

    epyccel_func = epyccel(union_int, language = language)
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result == pyccel_result

def test_set_union_list(language):
    def union_list():
        a = {1.2, 2.3}
        b = [1.2, 5.0]
        d = a.union(b)
        return len(d), d.pop(), d.pop(), d.pop()

    epyccel_func = epyccel(union_list, language = language)
    pyccel_result = epyccel_func()
    python_result = union_list()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_union_tuple(language):
    def union_tuple():
        a = {True}
        b = (False,)
        d = a.union(b)
        return len(d), d.pop(), d.pop()

    epyccel_func = epyccel(union_tuple, language = language)
    pyccel_result = epyccel_func()
    python_result = union_tuple()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_union_operator(language):
    def union_int():
        a = {1,2,3,4}
        b = {5,6,7,2}
        c = a | b
        return len(c), c.pop(), c.pop(), c.pop(), c.pop(), c.pop(), c.pop(), c.pop()

    epyccel_func = epyccel(union_int, language = language)
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_union_augoperator(language):
    def union_int():
        a = {1,2,3,4}
        b = {5,6,7,2}
        a |= b
        return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()

    epyccel_func = epyccel(union_int, language = language)
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_intersection_int(language):
    def intersection_int():
        a = {1,2,3}
        b = {2,3,4}
        c = a.intersection(b)
        return len(c), c.pop(), c.pop()

    epyccel_func = epyccel(intersection_int, language = language)
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_intersection_no_args(language):
    def intersection_int():
        a = {1,2,3,4}
        c = a.intersection()
        a.add(5)
        return len(c), c.pop(), c.pop(), c.pop(), c.pop()

    epyccel_func = epyccel(intersection_int, language = language)
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_intersection_2_args(language):
    def intersection_int():
        a = {1,2,3,4}
        b = {5,6,7,2,1,3}
        c = {7,6,10,4,2,3,1}
        d = a.intersection(b, c)
        return len(d), d.pop(), d.pop(), d.pop()

    epyccel_func = epyccel(intersection_int, language = language)
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_intersection_int_temporaries(language):
    def intersection_int():
        c = {1,2,3}.intersection({2,3,4})
        return len(c), c.pop(), c.pop()

    epyccel_func = epyccel(intersection_int, language = language)
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_temporary_set_intersection(language):
    def intersection_int():
        a = {1,2}
        b = {2}
        d = a.intersection(b).pop()
        return d

    epyccel_func = epyccel(intersection_int, language = language)
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result == pyccel_result

def test_set_intersection_operator(language):
    def intersection_int():
        a = {1,2,3,4,8}
        b = {5,2,3,7,8}
        c = a & b
        return len(c), c.pop(), c.pop(), c.pop()

    epyccel_func = epyccel(intersection_int, language = language)
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_intersection_operator_2(language):
    def intersection_int():
        a = {1,2,3,4,8}
        b = {5,2,3,7,8}
        return a & b

    epyccel_func = epyccel(intersection_int, language = language)
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert set(python_result) == set(pyccel_result)

def test_set_intersection_update(language):
    def intersection_int():
        a = {1,2,3,4,8}
        b = {5,2,3,7,8}
        a.intersection_update(b)
        return len(a), a.pop(), a.pop(), a.pop()

    epyccel_func = epyccel(intersection_int, language = language)
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_intersection_multiple_update(language):
    def intersection_int():
        a = {1,2,3,4,8}
        b = {5,2,3,7,8}
        c = {10,2,20}
        a.intersection_update(b, c)
        return len(a), a.pop()

    epyccel_func = epyccel(intersection_int, language = language)
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_intersection_augoperator(language):
    def intersection_int():
        a = {1,2,3,4}
        b = {2,3,4}
        a &= b
        return len(a), a.pop(), a.pop(), a.pop()

    epyccel_func = epyccel(intersection_int, language = language)
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_contains(language):
    def union_int():
        a = {1,2,3,4,5,6,7,8}
        b = 2 in a
        return b, (4 in a), (9 in a)

    epyccel_func = epyccel(union_int, language = language)
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result == pyccel_result

def test_set_ptr(language):
    def set_ptr():
        a = {1,2,3,4,5,6,7,8}
        b = a
        b.pop()
        return len(a), len(b)

    epyccel_func = epyccel(set_ptr, language = language)
    pyccel_result = epyccel_func()
    python_result = set_ptr()
    assert python_result == pyccel_result

def test_set_iter(language):
    def set_sum_int():
        a =  {1,2,3,4,5,6,7,8,9,12}
        sum_a = 0
        for ai in a:
            sum_a += ai
        return sum_a

    epyccel_func = epyccel(set_sum_int, language = language)
    pyccel_result = epyccel_func()
    python_result = set_sum_int()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))

def test_set_iter_prod(language):
    def set_iter_prod():
        # Integers must be used to get an exact result to compare sets
        from itertools import product
        a = {1,2,3,4,5,6,7,8,9,12}
        b = {2, 4, 9, 1, 8}
        assemble = 0.0
        for ai, bi in product(a,b):
            assemble += ai*bi
        return assemble

    epyccel_func = epyccel(set_iter_prod, language = language)
    pyccel_result = epyccel_func()
    python_result = set_iter_prod()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))

def test_set_const_arg(language):
    T = TypeVar('T', int, float, complex)

    def set_arg(arg : Final[set[T]], my_sum : T):
        for ai in arg:
            my_sum += ai
        return my_sum

    epyccel_func = epyccel(set_arg, language = language)
    int_arg = {1,2,3,4,5,6,7}
    float_arg = {1.5, 2.5, 3.5, 4.5, 6.7}
    complex_arg = {1+0j,4j,2.5+2j}
    for arg in (int_arg, float_arg, complex_arg):
        start = type(next(iter(arg)))(0)
        pyccel_result = epyccel_func(arg, start)
        python_result = set_arg(arg, start)
        assert python_result == pyccel_result
        assert isinstance(pyccel_result, type(python_result))

def test_set_arg(stc_language):
    def set_arg(arg : 'set[int]', n : int):
        arg.update(range(n))

    epyccel_func = epyccel(set_arg, language = stc_language)
    arg_pyc = {7,8,9,10}
    arg_pyt = arg_pyc.copy()
    n = 6
    epyccel_func(arg_pyc, n)
    set_arg(arg_pyt, n)
    assert arg_pyc == arg_pyt

def test_set_return(language):
    def set_return():
        a = {1,2,3,4,5}
        return a

    epyccel_func = epyccel(set_return, language = language)
    pyccel_result = epyccel_func()
    python_result = set_return()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))
    assert isinstance(python_result.pop(), type(pyccel_result.pop()))

def test_set_min_max(language):
    def set_min_max():
        a_int = {1,2,3,4,5}
        a_float = {1.1,2.2,3.3,4.4,5.5}
        return min(a_int), min(a_float), max(a_int), max(a_float)

    epyccel_func = epyccel(set_min_max, language = language)
    pyccel_result = epyccel_func()
    python_result = set_min_max()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))

def test_set_is_disjoint(language):
    def set_is_disjoint(a : set[int], b : set[int]):
        return a.isdisjoint(b)

    epyccel_func = epyccel(set_is_disjoint, language = language)
    example_set1 = {1,2,3,4}
    example_set2 = {5,6,7,8}
    example_set3 = {7,8,2}
    assert set_is_disjoint(example_set1, example_set2) == epyccel_func(example_set1, example_set2)
    assert set_is_disjoint(example_set1, example_set3) == epyccel_func(example_set1, example_set3)
    assert set_is_disjoint(example_set3, example_set2) == epyccel_func(example_set3, example_set2)

def test_set_difference_int(language):
    def difference_int():
        a = {1,2,3}
        b = {2,3,4}
        c = a.difference(b)
        return len(c), c.pop()

    epyccel_func = epyccel(difference_int, language = language)
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_difference_no_args(language):
    def difference_int():
        a = {1,2,3,4}
        c = a.difference()
        a.add(5)
        return len(c), c.pop(), c.pop(), c.pop(), c.pop()

    epyccel_func = epyccel(difference_int, language = language)
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_difference_2_args(language):
    def difference_int():
        a = {1,2,3,4}
        b = {5,6,7,2,1}
        c = {7,6,10,2,3,1}
        d = a.difference(b, c)
        return len(d), d.pop()

    epyccel_func = epyccel(difference_int, language = language)
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_difference_int_temporaries(language):
    def difference_int():
        c = {1,2,3}.difference({3,4})
        return len(c), c.pop(), c.pop()

    epyccel_func = epyccel(difference_int, language = language)
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_temporary_set_difference(language):
    def difference_int():
        a = {1,2}
        b = {2}
        d = a.difference(b).pop()
        return d

    epyccel_func = epyccel(difference_int, language = language)
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result == pyccel_result

def test_set_difference_operator(language):
    def difference_int():
        a = {1,2,3,4,8}
        b = {5,2,3,7,8}
        c = a - b
        return len(c), c.pop(), c.pop()

    epyccel_func = epyccel(difference_int, language = language)
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_difference_update(language):
    def difference_int():
        a = {1,2,3,4,8}
        b = {5,2,3,7,8}
        a.difference_update(b)
        return len(a), a.pop(), a.pop()

    epyccel_func = epyccel(difference_int, language = language)
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_difference_multiple_update(language):
    def difference_int():
        a = {1,2,3,4,8}
        b = {5,2,3,7,8}
        c = {10,4,20}
        a.difference_update(b, c)
        return len(a), a.pop()

    epyccel_func = epyccel(difference_int, language = language)
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])

def test_set_difference_augoperator(language):
    def difference_int():
        a = {1,2,3,4}
        b = {2,3,4}
        a -= b
        return len(a), a.pop()

    epyccel_func = epyccel(difference_int, language = language)
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])
