# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import Final

import numpy as np
import pytest
from modules import lists
from utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_lists_mod(language):
    return epyccel_module_with_fallback(lists, language)


@pytest.fixture(
    params=[
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.skip(reason="list methods not implemented in fortran"),
                pytest.mark.fortran,
            ],
        ),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="list methods not implemented in c"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ],
    scope="module",
)
def limited_language(request):
    return request.param


def test_pop_last_element(epyc_lists_mod):
    epyc_last_element = epyc_lists_mod.pop_last_element
    pyccel_result = epyc_last_element()
    python_result = lists.pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_list_bool(epyc_lists_mod):
    epyc_last_element = epyc_lists_mod.pop_list_bool
    pyccel_result = epyc_last_element()
    python_result = lists.pop_list_bool()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_list_float(epyc_lists_mod):
    epyc_last_element = epyc_lists_mod.pop_list_float
    pyccel_result = epyc_last_element()
    python_result = lists.pop_list_float()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_list_of_lists(stc_language):
    def pop_last_element():
        a = [[4.6, 3.3], [4.2, 9.1], [2.3, 6.8]]
        b = a.pop()
        return a.pop(), b

    epyc_last_element = epyccel(pop_last_element, language=stc_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_list_of_lists_var(stc_language):
    def pop_last_element():
        a = [[4.6, 3.3], [4.2, 9.1], [2.3, 6.8]]
        b = a.pop()
        return b

    epyc_last_element = epyccel(pop_last_element, language=stc_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_list_of_lists_ref(stc_language):
    def pop_last_element():
        a = [1, 2]
        b = [3, 4]
        c = [a, b]
        d = c.pop()
        return d[0] + d[1]

    epyc_last_element = epyccel(pop_last_element, language=stc_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_list_of_lists_ref_2(stc_language):
    def pop_last_element():
        a = [1, 2]
        b = [3, 4]
        c = [a, b]
        d = c.pop()
        e = [d, b]
        return d[0] + d[1] + e[1][1]

    epyc_last_element = epyccel(pop_last_element, language=stc_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_list_of_tuples(limited_language):
    def pop_last_element():
        a = [(4.6, 3.3), (4.2, 9.1), (2.3, 6.8)]
        return a.pop()

    epyc_last_element = epyccel(pop_last_element, language=limited_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_list_of_ndarrays(limited_language):
    def pop_last_element():
        from numpy import array

        array1 = array([[1, 2, 3], [4, 5, 6]])
        array2 = array([[7, 8, 9], [10, 11, 12]])
        array3 = array([[13, 14, 15], [16, 17, 18]])
        a = [array1, array2, array3]
        r = array(a.pop())
        return r

    epyc_last_element = epyccel(pop_last_element, language=limited_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert np.array_equal(python_result, pyccel_result)


def test_pop_specific_index(epyc_lists_mod):
    epyc_specific_index = epyc_lists_mod.pop_specific_index
    python_result = lists.pop_specific_index()
    pyccel_result = epyc_specific_index()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_negative_index(epyc_lists_mod):
    epyc_negative_index = epyc_lists_mod.pop_negative_index
    python_result = lists.pop_negative_index()
    pyccel_result = epyc_negative_index()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_2(epyc_lists_mod):
    pop_2_epyc = epyc_lists_mod.pop_2
    python_result = lists.pop_2()
    pyccel_result = pop_2_epyc()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_expression(epyc_lists_mod):
    epyc_last_element = epyc_lists_mod.pop_expression
    pyccel_result = epyc_last_element()
    python_result = lists.pop_expression()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_pop_as_arg(epyc_lists_mod):
    epyc_as_arg = epyc_lists_mod.pop_as_arg
    pyccel_result = epyc_as_arg()
    python_result = lists.pop_as_arg()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_append_basic(epyc_lists_mod):
    epyc_f = epyc_lists_mod.append_basic
    assert lists.append_basic() == epyc_f()


def test_append_multiple(epyc_lists_mod):
    epyc_f = epyc_lists_mod.append_multiple
    assert lists.append_multiple() == epyc_f()


def test_append_list(stc_language):
    def f():
        a = [[1, 2, 3]]
        a.append([4, 5, 6])
        return len(a)

    epyc_f = epyccel(f, language=stc_language)
    assert f() == epyc_f()


def test_append_range(epyc_lists_mod):
    epyc_f = epyc_lists_mod.append_range
    assert lists.append_range() == epyc_f()


def test_append_range_list(limited_language):
    def f():
        a = [[1, 2, 3]]
        for i in range(0, 1000):
            a.append([i, i + 1])
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_append_bool(epyc_lists_mod):
    epyc_f = epyc_lists_mod.append_bool
    assert lists.append_bool() == epyc_f()


def test_append_float(epyc_lists_mod):
    epyc_f = epyc_lists_mod.append_float
    assert lists.append_float() == epyc_f()


def test_append_complex(epyc_lists_mod):
    epyc_f = epyc_lists_mod.append_complex
    assert lists.append_complex() == epyc_f()


def test_append_ndarrays(limited_language):
    def f():
        from numpy import array

        array1 = array([[1, 2, 3], [4, 5, 6]])
        array2 = array([[7, 8, 9], [10, 11, 12]])
        array3 = array([[13, 14, 15], [16, 17, 18]])
        array4 = array([[5, 4, 66], [69, 2, 180]])
        array5 = array([[13, 14, 15], [6, 27, 0]])
        array6 = array([[13, 1, 5], [16, 17, 18]])
        a = [array1, array2, array3]
        a.append(array4)
        a.append(array5)
        a.append(array6)
        return len(a), a[0][0, 0], a[-1][1, 0]

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_append_user_defined_objects(limited_language):
    import modules.list_user_defined_objs1 as mod

    modnew = epyccel(mod, language=limited_language)
    python_list = mod.fn()
    accelerated_list = modnew.fn()
    assert len(python_list) == len(accelerated_list)
    for pi, ai in zip(python_list, accelerated_list):
        assert pi.x == ai.x


def test_insert_basic(limited_language):
    def f():
        a = [1, 2, 3]
        a.insert(4, 4)
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_insert_booleans(epyc_lists_mod):
    epyc_f = epyc_lists_mod.insert_booleans
    assert lists.insert_booleans() == epyc_f()


def test_insert_complex(epyc_lists_mod):
    epyc_f = epyc_lists_mod.insert_complex
    assert lists.insert_complex() == epyc_f()


def test_insert_float(epyc_lists_mod):
    epyc_f = epyc_lists_mod.insert_float
    assert lists.insert_float() == epyc_f()


def test_insert_ndarrays(limited_language):
    def f():
        from numpy import array

        array1 = array([[1, 2, 3], [4, 5, 6]])
        array2 = array([[7, 8, 9], [10, 11, 12]])
        array3 = array([[13, 14, 15], [16, 17, 18]])
        array4 = array([[5, 4, 66], [69, 2, 180]])
        array5 = array([[13, 14, 15], [6, 27, 0]])
        array6 = array([[13, 1, 5], [16, 17, 18]])
        a = [array1, array2]
        a.insert(-100, array3)
        a.insert(0, array4)
        a.insert(100, array5)
        a.insert(-3, array6)
        return len(a), a[0][0, 1], a[-1][1, 2]

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_insert_multiple(epyc_lists_mod):
    epyc_f = epyc_lists_mod.insert_multiple
    assert lists.insert_multiple() == epyc_f()


def test_insert_list(limited_language):
    def f():
        a = [[1, 2, 3]]
        a.insert(1, [4, 5, 6])
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_insert_range(epyc_lists_mod):
    epyc_f = epyc_lists_mod.insert_range
    assert lists.insert_range() == epyc_f()


def test_insert_range_list(limited_language):
    def f():
        a = [[1, 2, 3]]
        for i in range(4, 1000):
            a.insert(i, [i, i + 1])
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_insert_user_defined_objects(limited_language):
    import modules.list_user_defined_objs as mod

    modnew = epyccel(mod, language=limited_language)
    python_list = mod.fn()
    accelerated_list = modnew.fn()
    assert python_list == accelerated_list


def test_clear_1(epyc_lists_mod):

    epyc_clear_1 = epyc_lists_mod.clear_1
    pyccel_result = epyc_clear_1()
    python_result = lists.clear_1()
    assert python_result == pyccel_result


def test_clear_2(epyc_lists_mod):

    epyc_clear_2 = epyc_lists_mod.clear_2
    pyccel_result = epyc_clear_2()
    python_result = lists.clear_2()
    assert python_result == pyccel_result


def test_clear_3(limited_language):

    def clear_3():
        a = [[1, 2, 3]]
        a.clear()
        return a

    epyc_clear_3 = epyccel(clear_3, language=limited_language)
    pyccel_result = epyc_clear_3()
    python_result = clear_3()
    assert python_result == pyccel_result


def test_extend_basic(limited_language):
    def f():
        a = [1, 2, 3]
        b = [4, 5, 6]
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_multiple(limited_language):
    def f():
        a = [1, 2, 3]
        a.extend([4, 5])
        a.extend([6, 7, 8, 9])
        a.extend([10])
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_nested_list(limited_language):
    def f():
        a = [[1, 2, 3]]
        a.extend([[4, 5, 6]])
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_tuple_with_list(limited_language):
    def f():
        a = [1, 2, 3]
        b = (4, 5, 6)
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_boolean_tuple(limited_language):
    def f():
        a = [True, False, True]
        b = (False, True, False)
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_float_tuple(limited_language):
    def f():
        a = [3.4, 2.1, 3.9]
        b = (4.1, 5.9, 0.3)
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_complex_list(limited_language):
    def f():
        a = [1j, 2 + 3j, 0 + 0j]
        b = [4j, 5j, 1 + 6j]
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_range(limited_language):
    def f():
        a = [1, 2, 3]
        a.extend(range(4, 9))
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_function_return(limited_language):
    def f():
        def g():
            a = [4, 5]
            return a

        lst = [1, 2, 3]
        lst.extend(g())

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_sort_basic(limited_language):
    def f():
        a = [4, 0, 1, -1]
        a.sort()
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_sort_bool(limited_language):
    def f():
        a = [True, False, False, True, False]
        a.sort()
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_sort_float(limited_language):
    def f():
        a = [3.4, 1.0, -4.5, 0.0, 2.1]
        a.sort()
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_list_as_arg(limited_language):
    def f():
        a = [1, 2, 3]
        a.extend([4, 5, 6])
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_tuple_as_arg(limited_language):
    def f():
        a = [1, 2, 3]
        a.extend((4, 5, 6))
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_np_int(limited_language):
    def f():
        from numpy import int64, ones

        a = [int64(1), int64(2), int64(3)]
        b = ones(3, dtype=int64)
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_np_float(limited_language):
    def f():
        from numpy import float64, ones

        a = [float64(1.0), float64(2.0), float64(3.0)]
        b = ones(3, dtype=float64)
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_user_defined_objects(limited_language):
    import modules.list_user_defined_objs2 as mod

    modnew = epyccel(mod, language=limited_language)
    python_list = mod.fn()
    accelerated_list = modnew.fn()
    assert python_list == accelerated_list


def test_remove_basic(limited_language):
    def f():
        lst = [1, 2, 3, 4]
        lst.remove(3)
        return lst

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_remove_float(limited_language):
    def f():
        lst = [1.4, 2.3, 3.2, 4.4]
        lst.remove(3.2)
        return lst

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_remove_complex(limited_language):
    def f():
        lst = [1j, 3j, 8j]
        lst.remove(3j)
        return lst

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_remove_list_from_list(limited_language):
    def f():
        lst = [[True, False, True], [False, True]]
        lst.remove([False, True])
        return lst

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_list_class_attribute(limited_language):
    import modules.list_class_attr as mod

    modnew = epyccel(mod, language=limited_language)
    python_list = mod.fn()
    accelerated_list = modnew.fn()
    assert python_list == accelerated_list


def test_copy_basic(limited_language):
    def f():
        a = [1, 2, 3]
        b = a.copy()
        return b

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_copy_nested(limited_language):
    def f():
        a = [[1, 2], [3, 4]]
        b = a.copy()
        return b

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_copy_modify_nested_values(limited_language):
    def f():
        a = [[1, 2], [3, 4]]
        b = a.copy()
        a[0][0] = 0
        a[0][1] = 0
        return b

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_mixed_list_methods(limited_language):
    def f():
        a = [(1, 4, 5), (33, 12, 5), (3, 5)]
        a.append((0, 1, 2))
        a.pop()
        a.clear()
        a.insert(-10, (2, 4, 3))
        a.extend(((4, 5, 6), (3, 3)))
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_extend_returned_list(limited_language):
    def f():
        def g():
            return [4, 5, 6]

        lst = [1, 2, 3]
        lst.extend(g())

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()


def test_mutable_indexing(stc_language):
    def f():
        a = [1, 2, 3, 4]
        a[0] = 5
        a[2] += 6
        return a[0], a[1], a[2], a[3]

    epyc_f = epyccel(f, language=stc_language)
    assert f() == epyc_f()


def test_mutable_multi_level_indexing(stc_language):
    def f():
        a = [1, 2, 3, 4]
        b = [a]
        b[0][0] = 5
        b[0][2] = 6
        return a[0], a[1], a[2], a[3]

    epyc_f = epyccel(f, language=stc_language)
    assert f() == epyc_f()


def test_mutable_multi_level_indexing_2(stc_language):
    def f():
        a = [1, 2, 3, 4]
        b = [a]
        c = b[0]
        c[0] = 5
        c[2] = 6
        return a[0], a[1], a[2], a[3]

    epyc_f = epyccel(f, language=stc_language)
    assert f() == epyc_f()


def test_homogenous_list_int_copy(limited_language):
    def homogeneous_list_int():
        return list([1, 2, 3, 4])

    f1 = homogeneous_list_int
    f2 = epyccel(f1, language=limited_language)

    python_out = f1()
    pyccel_out = f2()

    assert python_out == pyccel_out


def test_homogenous_list_bool_copy(limited_language):
    def homogeneous_list_bool():
        return list([True, False, True, False])

    f1 = homogeneous_list_bool
    f2 = epyccel(f1, language=limited_language)

    python_out = f1()
    pyccel_out = f2()

    assert python_out == pyccel_out


def test_homogenous_list_float_copy(limited_language):
    def homogeneous_list_float():
        return list([1.0, 2.0, 3.0, 4.0])

    f1 = homogeneous_list_float
    f2 = epyccel(f1, language=limited_language)

    python_out = f1()
    pyccel_out = f2()

    assert python_out == pyccel_out


def test_homogenous_list_int_tuple_copy(limited_language):
    def homogeneous_list_int_tuple():
        return list((1, 2, 3, 4))

    f1 = homogeneous_list_int_tuple
    f2 = epyccel(f1, language=limited_language)

    python_out = f1()
    pyccel_out = f2()

    assert python_out == pyccel_out


def test_homogenous_list_unknown_size_copy(limited_language):
    def homogeneous_list_unknown_size_copy(n: int):
        a = (3,) * n
        b = list(a)
        return b[0]

    f1 = homogeneous_list_unknown_size_copy
    f2 = epyccel(f1, language=limited_language)

    python_out = f1(5)
    pyccel_out = f2(5)

    assert python_out == pyccel_out


def test_list_contains(epyc_lists_mod):
    epyc_list_contains = epyc_lists_mod.list_contains
    pyccel_result = epyc_list_contains()
    python_result = lists.list_contains()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_list_ptr(epyc_lists_mod):
    epyc_list_ptr = epyc_lists_mod.list_ptr
    pyccel_result = epyc_list_ptr()
    python_result = lists.list_ptr()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_list_return(epyc_lists_mod):
    epyccel_func = epyc_lists_mod.list_return
    pyccel_result = epyccel_func()
    python_result = lists.list_return()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))
    assert isinstance(python_result.pop(), type(pyccel_result.pop()))


def test_list_min_max(epyc_lists_mod):
    epyccel_func = epyc_lists_mod.list_min_max
    pyccel_result = epyccel_func()
    python_result = lists.list_min_max()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))


def test_list_reverse(epyc_lists_mod):
    epyccel_func = epyc_lists_mod.list_reverse
    pyccel_result = epyccel_func()
    python_result = lists.list_reverse()
    assert python_result == pyccel_result


def test_list_str(stc_language):
    def list_str():
        a = ["hello", "world", "!"]
        return len(a)

    epyccel_func = epyccel(list_str, language=stc_language)
    pyccel_result = epyccel_func()
    python_result = list_str()
    assert python_result == pyccel_result


def test_list_const_arg(epyc_lists_mod):
    epyccel_func = epyc_lists_mod.list_const_arg
    int_arg = [1, 2, 3, 4, 5, 6, 7]
    float_arg = [1.5, 2.5, 3.5, 4.5, 6.7]
    complex_arg = [1 + 0j, 4j, 2.5 + 2j]
    for arg in (int_arg, float_arg, complex_arg):
        start = type(next(iter(arg)))(0)
        pyccel_result = epyccel_func(arg, start)
        python_result = lists.list_const_arg(arg, start)
        assert python_result == pyccel_result
        assert isinstance(pyccel_result, type(python_result))


def test_list_arg(stc_language):
    def list_arg(arg: "list[int]", n: int):
        arg.extend(range(n))

    epyccel_func = epyccel(list_arg, language=stc_language)
    arg_pyc = [7, 8, 9, 10]
    arg_pyt = arg_pyc.copy()
    n = 6
    epyccel_func(arg_pyc, n)
    list_arg(arg_pyt, n)
    assert arg_pyc == arg_pyt


def test_list_equality(epyc_lists_mod):
    epyccel_func = epyc_lists_mod.list_equality
    arg1 = [1, 2, 3, 4, 5]
    arg2 = [4, 5, 6, 7, 8]
    arg3 = [1, 2, 3]

    assert lists.list_equality(arg1, arg1) == epyccel_func(arg1, arg1)
    assert lists.list_equality(arg1, arg2) == epyccel_func(arg1, arg2)
    assert lists.list_equality(arg1, arg3) == epyccel_func(arg1, arg3)
    assert lists.list_equality(  # pylint: disable=arguments-out-of-order
        arg2, arg1
    ) == epyccel_func(
        arg2, arg1
    )  # pylint: disable=arguments-out-of-order
    assert lists.list_equality(arg3, arg1) == epyccel_func(arg3, arg1)


def test_list_equality_non_matching_types(limited_language):
    def list_equality(arg1: Final[list[int]], arg2: Final[list[float]]):
        return arg1 == arg2

    epyccel_func = epyccel(list_equality, language=limited_language)
    arg_int1 = [1, 2, 3, 4, 5]
    arg_int2 = [4, 5, 6, 7, 8]
    arg_int3 = [1, 2, 3]
    arg_float1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    arg_float2 = [4.0, 5.0, 6.0, 7.0, 8.0]
    arg_float3 = [1.0, 2.0, 3.0]

    assert list_equality(arg_int1, arg_float1) == epyccel_func(arg_int1, arg_float1)
    assert list_equality(arg_int1, arg_float2) == epyccel_func(arg_int1, arg_float2)
    assert list_equality(arg_int1, arg_float3) == epyccel_func(arg_int1, arg_float3)
    assert list_equality(arg_int2, arg_float1) == epyccel_func(arg_int2, arg_float1)
    assert list_equality(arg_int3, arg_float1) == epyccel_func(arg_int3, arg_float1)


def test_list_duplicate(epyc_lists_mod):
    epyccel_func = epyc_lists_mod.list_duplicate

    assert lists.list_duplicate(5) == epyccel_func(5)
    assert lists.list_duplicate(15) == epyccel_func(15)


def test_list_assign_slices(epyc_lists_mod):
    epyccel_func = epyc_lists_mod.list_assign_slices

    arg_int1 = [1, 2, 3, 4, 5]
    arg_int2 = [29, 23, 19, 17, 13, 11, 7]
    arg_float1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    arg_float2 = [29.0, 23.0, 19.0, 17.0, 13.0, 11.0, 7.0]
    arg_complex1 = [19 + 1j, 17 + 2j, 13 + 3j, 11 + 4j, 7 + 5j]

    assert lists.list_assign_slices(arg_int1) == epyccel_func(arg_int1)
    assert lists.list_assign_slices(arg_int2) == epyccel_func(arg_int2)
    assert lists.list_assign_slices(arg_float1) == epyccel_func(arg_float1)
    assert lists.list_assign_slices(arg_float2) == epyccel_func(arg_float2)
    assert lists.list_assign_slices(arg_complex1) == epyccel_func(arg_complex1)
