# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import Final, TypeVar
import pytest
import numpy as np
from pyccel import epyccel

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
def limited_language(request):
    return request.param

def test_pop_last_element(language) :
    def pop_last_element():
        a = [1,3,45]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = language, verbose = 2)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_list_bool(language) :
    def pop_last_element():
        a = [True, False, True]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_list_float(language) :
    def pop_last_element():
        a = [1.5 , 3.1, 4.5]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_list_of_lists(stc_language) :
    def pop_last_element():
        a = [[4.6, 3.3], [4.2, 9.1], [2.3, 6.8]]
        b = a.pop()
        return a.pop(), b
    epyc_last_element = epyccel(pop_last_element, language = stc_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_list_of_lists_var(stc_language) :
    def pop_last_element():
        a = [[4.6, 3.3], [4.2, 9.1], [2.3, 6.8]]
        b = a.pop()
        return b
    epyc_last_element = epyccel(pop_last_element, language = stc_language)
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
    epyc_last_element = epyccel(pop_last_element, language = stc_language)
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
        e = [d,b]
        return d[0] + d[1] + e[1][1]
    epyc_last_element = epyccel(pop_last_element, language = stc_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_list_of_tuples(limited_language) :
    def pop_last_element():
        a = [(4.6, 3.3), (4.2, 9.1), (2.3, 6.8)]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = limited_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_list_of_ndarrays(limited_language) :
    def pop_last_element():
        from numpy import array

        array1 = array([[1, 2, 3], [4, 5, 6]])
        array2 = array([[7, 8, 9], [10, 11, 12]])
        array3 = array([[13, 14, 15], [16, 17, 18]])
        a = [array1, array2, array3]
        r = array(a.pop())
        return r
    epyc_last_element = epyccel(pop_last_element, language = limited_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert np.array_equal(python_result, pyccel_result)

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

def test_pop_expression(language) :
    def pop_last_element():
        a = [1, 3, 45]
        return a.pop() + 3
    epyc_last_element = epyccel(pop_last_element, language = language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_as_arg(language) :
    def pop_as_arg():
        a = [1, 3, 45]
        return a.pop(a.pop(0))
    epyc_as_arg = epyccel(pop_as_arg, language = language)
    pyccel_result = epyc_as_arg()
    python_result = pop_as_arg()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_append_basic(language):
    def f():
        a = [1, 2, 3]
        a.append(4)
        return len(a), a[0], a[1], a[2], a[3]

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_multiple(language):
    def f():
        a = [1, 2, 3]
        a.append(4)
        a.append(5)
        a.append(6)
        return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_list(stc_language):
    def f():
        a = [[1, 2, 3]]
        a.append([4, 5, 6])
        return len(a)

    epyc_f = epyccel(f, language=stc_language)
    assert f() == epyc_f()

def test_append_range(language):
    def f():
        a = [1, 2, 3]
        for i in range(0, 1000):
            a.append(i)
        a.append(1000)
        return len(a), a[-1], a[-2]

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_range_list(limited_language):
    def f():
        a = [[1, 2, 3]]
        for i in range(0, 1000):
            a.append([i, i + 1])
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()

def test_append_bool(language):
    def f():
        a = [True, True, True]
        a.append(False)
        a.append(False)
        a.append(True)
        return len(a), a[3], a[4], a[5]

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_float(language):
    def f():
        a = [3.5, 2.2, 1.5]
        a.append(3.0)
        a.append(2.9)
        a.append(1.1)
        return len(a), a[3], a[4], a[5]

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_complex(language):
    def f():
        a = [1 + 2j, 3 + 4j, 5 + 6j]
        a.append(9j)
        a.append(2 + 2j)
        a.append(1j)
        return len(a), a[3], a[4], a[5]

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

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
        return len(a), a[0][0,0], a[-1][1,0]

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

def test_insert_booleans(language):
    def f():
        a = [True, False, True]
        a.insert(0, True)
        a.insert(-100, True)
        a.insert(1000, False)
        a.insert(0, False)
        a.insert(666, True)
        a.insert(-1, True)
        a.insert(-25, False)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_insert_complex(language):
    def f():
        a = [2j, 3 + 6j, 0 + 0j]
        a.insert(0, 9j)
        a.insert(-100, 1 - 1j)
        a.insert(1000, -3j)
        a.insert(0, 0j)
        a.insert(666, 1j)
        a.insert(-1, 1 + 0j)
        a.insert(-25, 0 - 0j)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_insert_float(language):
    def f():
        a = [0.0, 3.6 , 0.5]
        a.insert(0, 6.4)
        a.insert(-100, 25.12)
        a.insert(1000, 13.04)
        a.insert(0, 19.99)
        a.insert(666, 20.00)
        a.insert(-1, 3.01)
        a.insert(-25, 2.5)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

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
        return len(a), a[0][0,1], a[-1][1,2]

    epyc_f = epyccel(f, language=limited_language)
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

def test_insert_list(limited_language):
    def f():
        a = [[1, 2, 3]]
        a.insert(1, [4, 5, 6])
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()

def test_insert_range(language):
    def f():
        a = [1, 2, 3]
        for i in range(4, 1000):
            a.insert(i - 1 ,i)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

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
        a : 'list[int]' = []
        a.clear()
        return a

    epyc_clear_2 = epyccel(clear_2, language = language)
    pyccel_result = epyc_clear_2()
    python_result = clear_2()
    assert python_result == pyccel_result

def test_clear_3(limited_language):

    def clear_3():
        a = [[1, 2, 3]]
        a.clear()
        return a

    epyc_clear_3 = epyccel(clear_3, language = limited_language)
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
        from numpy import ones, int64

        a = [int64(1),int64(2),int64(3)]
        b = ones(3, dtype=int64)
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=limited_language)
    assert f() == epyc_f()

def test_extend_np_float(limited_language):
    def f():
        from numpy import ones, float64

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
        a = [1,2,3,4]
        a[0] = 5
        a[2] += 6
        return a[0], a[1], a[2], a[3]

    epyc_f = epyccel(f, language=stc_language)
    assert f() == epyc_f()

def test_mutable_multi_level_indexing(stc_language):
    def f():
        a = [1,2,3,4]
        b = [a]
        b[0][0] = 5
        b[0][2] = 6
        return a[0], a[1], a[2], a[3]

    epyc_f = epyccel(f, language=stc_language)
    assert f() == epyc_f()

def test_mutable_multi_level_indexing_2(stc_language):
    def f():
        a = [1,2,3,4]
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
    f2 = epyccel( f1 , language=limited_language)

    python_out = f1()
    pyccel_out = f2()

    assert python_out == pyccel_out

def test_homogenous_list_bool_copy(limited_language):
    def homogeneous_list_bool():
        return list([True, False, True, False])
    f1 = homogeneous_list_bool
    f2 = epyccel( f1 , language=limited_language)

    python_out = f1()
    pyccel_out = f2()

    assert python_out == pyccel_out

def test_homogenous_list_float_copy(limited_language):
    def homogeneous_list_float():
        return list([1.0, 2.0, 3.0, 4.0])
    f1 = homogeneous_list_float
    f2 = epyccel( f1 , language=limited_language)

    python_out = f1()
    pyccel_out = f2()

    assert python_out == pyccel_out

def test_homogenous_list_int_tuple_copy(limited_language):
    def homogeneous_list_int_tuple():
        return list((1, 2, 3, 4))
    f1 = homogeneous_list_int_tuple
    f2 = epyccel( f1 , language=limited_language)

    python_out = f1()
    pyccel_out = f2()

    assert python_out == pyccel_out

def test_homogenous_list_unknown_size_copy(limited_language):
    def homogeneous_list_unknown_size_copy(n : int):
        a = (3,)*n
        b = list(a)
        return b[0]
    f1 = homogeneous_list_unknown_size_copy
    f2 = epyccel( f1 , language=limited_language)

    python_out = f1(5)
    pyccel_out = f2(5)

    assert python_out == pyccel_out

def test_list_contains(language):
    def list_contains():
        a = [1, 3, 4, 7, 10, 3]
        return (1 in a), (5 in a), (3 in a)

    epyc_list_contains = epyccel(list_contains, language = language)
    pyccel_result = epyc_list_contains()
    python_result = list_contains()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_list_ptr(language):
    def list_ptr():
        a = [1, 3, 4, 7, 10, 3]
        b = a
        b.append(22)
        return len(a), len(b)

    epyc_list_ptr = epyccel(list_ptr, language = language)
    pyccel_result = epyc_list_ptr()
    python_result = list_ptr()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_list_return(language):
    def list_return():
        a = [1,2,3,4,5]
        return a

    epyccel_func = epyccel(list_return, language = language)
    pyccel_result = epyccel_func()
    python_result = list_return()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))
    assert isinstance(python_result.pop(), type(pyccel_result.pop()))


def test_list_min_max(language):
    def list_min_max():
        a_int = [1, 2, 3, 4]
        a_float = [1.1, 2.2, 3.3, 4.4]
        return min(a_int), max(a_int), min(a_float), max(a_float)
    epyccel_func = epyccel(list_min_max, language = language)
    pyccel_result = epyccel_func()
    python_result = list_min_max()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))


def test_list_reverse(language):
    def list_reverse():
        a_int = [1, 2, 3]
        a_float = [1.1, 2.2, 3.3]
        a_complex = [1j, 2-3j]
        a_single = [1]
        a_int.reverse()
        a_float.reverse()
        a_complex.reverse()
        a_single.reverse()
        return (a_int[0], a_int[-1], a_float[0], a_float[-1],
                a_single[0], a_single[-1], a_complex[0], a_complex[-1])
    epyccel_func = epyccel(list_reverse, language = language)
    pyccel_result = epyccel_func()
    python_result = list_reverse()
    assert python_result == pyccel_result


def test_list_str(stc_language):
    def list_str():
        a = ['hello', 'world', '!']
        return len(a)

    epyccel_func = epyccel(list_str, language = stc_language)
    pyccel_result = epyccel_func()
    python_result = list_str()
    assert python_result == pyccel_result

def test_list_const_arg(language):
    T = TypeVar('T', int, float, complex)

    def list_arg(arg : Final[list[T]], my_sum : T):
        for ai in arg:
            my_sum += ai
        return my_sum

    epyccel_func = epyccel(list_arg, language = language)
    int_arg = [1,2,3,4,5,6,7]
    float_arg = [1.5, 2.5, 3.5, 4.5, 6.7]
    complex_arg = [1+0j,4j,2.5+2j]
    for arg in (int_arg, float_arg, complex_arg):
        start = type(next(iter(arg)))(0)
        pyccel_result = epyccel_func(arg, start)
        python_result = list_arg(arg, start)
        assert python_result == pyccel_result
        assert isinstance(pyccel_result, type(python_result))

def test_list_arg(stc_language):
    def list_arg(arg : 'list[int]', n : int):
        arg.extend(range(n))

    epyccel_func = epyccel(list_arg, language = stc_language)
    arg_pyc = [7,8,9,10]
    arg_pyt = arg_pyc.copy()
    n = 6
    epyccel_func(arg_pyc, n)
    list_arg(arg_pyt, n)
    assert arg_pyc == arg_pyt

def test_list_equality(language):
    def list_equality(arg1 : Final[list[int]], arg2 : Final[list[int]]):
        return arg1 == arg2

    epyccel_func = epyccel(list_equality, language = language)
    arg1 = [1,2,3,4,5]
    arg2 = [4,5,6,7,8]
    arg3 = [1,2,3]

    assert list_equality(arg1, arg1) == epyccel_func(arg1, arg1)
    assert list_equality(arg1, arg2) == epyccel_func(arg1, arg2)
    assert list_equality(arg1, arg3) == epyccel_func(arg1, arg3)
    assert list_equality(arg2, arg1) == epyccel_func(arg2, arg1) #pylint: disable=arguments-out-of-order
    assert list_equality(arg3, arg1) == epyccel_func(arg3, arg1)

def test_list_equality_non_matching_types(limited_language):
    def list_equality(arg1 : Final[list[int]], arg2 : Final[list[float]]):
        return arg1 == arg2

    epyccel_func = epyccel(list_equality, language = limited_language)
    arg_int1 = [1,2,3,4,5]
    arg_int2 = [4,5,6,7,8]
    arg_int3 = [1,2,3]
    arg_float1 = [1.,2.,3.,4.,5.]
    arg_float2 = [4.,5.,6.,7.,8.]
    arg_float3 = [1.,2.,3.]

    assert list_equality(arg_int1, arg_float1) == epyccel_func(arg_int1, arg_float1)
    assert list_equality(arg_int1, arg_float2) == epyccel_func(arg_int1, arg_float2)
    assert list_equality(arg_int1, arg_float3) == epyccel_func(arg_int1, arg_float3)
    assert list_equality(arg_int2, arg_float1) == epyccel_func(arg_int2, arg_float1)
    assert list_equality(arg_int3, arg_float1) == epyccel_func(arg_int3, arg_float1)

def test_list_duplicate(language):
    def list_duplicate(n : int):
        a = [1] * n
        b = [1, 2, 3] * n
        return a, b

    epyccel_func = epyccel(list_duplicate, language = language)

    assert list_duplicate(5) == epyccel_func(5)
    assert list_duplicate(15) == epyccel_func(15)
