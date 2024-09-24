# pylint: disable=missing-function-docstring, missing-module-docstring
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
def language(request):
    return request.param

@pytest.fixture( params=[
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="lists not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ],
    scope = "module"
)
def stc_language(request):
    return request.param

def test_pop_last_element(stc_language) :
    def pop_last_element():
        a = [1,3,45]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = stc_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_list_bool(stc_language) :
    def pop_last_element():
        a = [True, False, True]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = stc_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_list_float(stc_language) :
    def pop_last_element():
        a = [1.5 , 3.1, 4.5]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = stc_language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_list_of_lists(language) :
    def pop_last_element():
        a = [[4.6, 3.3], [4.2, 9.1], [2.3, 6.8]]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_list_of_tuples(language) :
    def pop_last_element():
        a = [(4.6, 3.3), (4.2, 9.1), (2.3, 6.8)]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_list_of_ndarrays(language) :
    def pop_last_element():
        from numpy import array

        array1 = array([[1, 2, 3], [4, 5, 6]])
        array2 = array([[7, 8, 9], [10, 11, 12]])
        array3 = array([[13, 14, 15], [16, 17, 18]])
        a = [array1, array2, array3]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert np.array_equal(python_result, pyccel_result)

def test_pop_specific_index(stc_language) :
    def pop_specific_index():
        a = [1j,3j,45j]
        return a.pop(1)
    epyc_specific_index = epyccel(pop_specific_index, language = stc_language)
    python_result = pop_specific_index()
    pyccel_result = epyc_specific_index()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_negative_index(stc_language) :
    def pop_negative_index():
        a = [1j,3j,45j]
        return a.pop(-1)
    epyc_negative_index = epyccel(pop_negative_index, language = stc_language)
    python_result = pop_negative_index()
    pyccel_result = epyc_negative_index()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

def test_pop_2(stc_language) :
    def pop_2():
        a = [1.7,2.7,45.0]
        a.pop()
        return a.pop(-1)
    pop_2_epyc = epyccel(pop_2, language = stc_language)
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

def test_append_bool(language):
    def f():
        a = [True, True, True]
        a.append(False)
        a.append(False)
        a.append(True)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_float(language):
    def f():
        a = [3.5, 2.2, 1.5]
        a.append(3.0)
        a.append(2.9)
        a.append(1.1)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_complex(language):
    def f():
        a = [1 + 2j, 3 + 4j, 5 + 6j]
        a.append(9j)
        a.append(2 + 2j)
        a.append(1j)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_append_ndarrays(language):
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
        return a

    epyc_f = epyccel(f, language=language)
    assert np.array_equal(f(), epyc_f())

def test_append_user_defined_objects(language):
    import modules.list_user_defined_objs2 as mod

    modnew = epyccel(mod, language=language)
    python_list = mod.fn()
    accelerated_list = modnew.fn()
    assert len(python_list) == len(accelerated_list)
    for python_elem, accelerated_elem in zip(python_list, accelerated_list):
        assert python_elem.x == accelerated_elem.x

def test_insert_basic(language):
    def f():
        a = [1, 2, 3]
        a.insert(4, 4)
        return a

    epyc_f = epyccel(f, language=language)
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

def test_insert_ndarrays(language):
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
        return a

    epyc_f = epyccel(f, language=language)
    assert np.array_equal(f(), epyc_f())

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

def test_extend_basic(language):
    def f():
        a = [1, 2, 3]
        b = [4, 5, 6]
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_multiple(language):
    def f():
        a = [1, 2, 3]
        a.extend([4, 5])
        a.extend([6, 7, 8, 9])
        a.extend([10])
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_nested_list(language):
    def f():
        a = [[1, 2, 3]]
        a.extend([[4, 5, 6]])
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_tuple_with_list(language):
    def f():
        a = [1, 2, 3]
        b = (4, 5, 6)
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_boolean_tuple(language):
    def f():
        a = [True, False, True]
        b = (False, True, False)
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_float_tuple(language):
    def f():
        a = [3.4, 2.1, 3.9]
        b = (4.1, 5.9, 0.3)
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_complex_list(language):
    def f():
        a = [1j, 2 + 3j, 0 + 0j]
        b = [4j, 5j, 1 + 6j]
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_range(language):
    def f():
        a = [1, 2, 3]
        a.extend(range(4, 9))
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_function_return(language):
    def f():
        def g():
            a = [4, 5]
            return a

        lst = [1, 2, 3]
        lst.extend(g())

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_sort_basic(language):
    def f():
        a = [4, 0, 1, -1]
        a.sort()
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_sort_bool(language):
    def f():
        a = [True, False, False, True, False]
        a.sort()
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_sort_float(language):
    def f():
        a = [3.4, 1.0, -4.5, 0.0, 2.1]
        a.sort()
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_list_as_arg(language):
    def f():
        a = [1, 2, 3]
        a.extend([4, 5, 6])
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_tuple_as_arg(language):
    def f():
        a = [1, 2, 3]
        a.extend((4, 5, 6))
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_np_int(language):
    def f():
        from numpy import ones, int64

        a = [int64(1),int64(2),int64(3)]
        b = ones(3, dtype=int64)
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_np_float(language):
    def f():
        from numpy import ones, float64

        a = [float64(1.0), float64(2.0), float64(3.0)]
        b = ones(3, dtype=float64)
        a.extend(b)
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_user_defined_objects(language):
    import modules.list_user_defined_objs2 as mod

    modnew = epyccel(mod, language=language)
    python_list = mod.fn()
    accelerated_list = modnew.fn()
    assert len(python_list) == len(accelerated_list)
    for python_elem, accelerated_elem in zip(python_list, accelerated_list):
        assert python_elem.x == accelerated_elem.x

def test_remove_basic(language):
    def f():
        lst = [1, 2, 3, 4]
        lst.remove(3)
        return lst
    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_remove_float(language):
    def f():
        lst = [1.4, 2.3, 3.2, 4.4]
        lst.remove(3.2)
        return lst
    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_remove_complex(language):
    def f():
        lst = [1j, 3j, 8j]
        lst.remove(3j)
        return lst
    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_remove_list_from_list(language):
    def f():
        lst = [[True, False, True], [False, True]]
        lst.remove([False, True])
        return lst
    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_extend_list_class_attribute(language):
    import modules.list_class_attr as mod

    modnew = epyccel(mod, language=language)
    python_list = mod.fn()
    accelerated_list = modnew.fn()
    assert python_list == accelerated_list

def test_copy_basic(language):
    def f():
        a = [1, 2, 3]
        b = a.copy()
        return b

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_copy_nested(language):
    def f():
        a = [[1, 2], [3, 4]]
        b = a.copy()
        return b

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_copy_modify_nested_values(language):
    def f():
        a = [[1, 2], [3, 4]]
        b = a.copy()
        a[0][0] = 0
        a[0][1] = 0
        return b

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

def test_mixed_list_methods(language):
    def f():
        a = [(1, 4, 5), (33, 12, 5), (3, 5)]
        a.append((0, 1, 2))
        a.pop()
        a.clear()
        a.insert(-10, (2, 4, 3))
        a.extend(((4, 5, 6), (3, 3)))
        return a

    epyc_f = epyccel(f, language=language)
    assert f() == epyc_f()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="List return not supported in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="List return not supported in fortran"),
            pytest.mark.fortran]),
        pytest.param("python", marks = [
            pytest.mark.xfail(reason="List return not implemented, related issue #337"),
            pytest.mark.fortran]),
    ]
)
def test_extend_returned_list(language):
    def f():
        def g():
            return [4, 5, 6]
        lst = [1, 2, 3]
        lst.extend(g())
    epyc_f = epyccel(f, language=language)
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

def test_homogenous_list_int_copy(language):
    def homogeneous_list_int():
        return list([1, 2, 3, 4])
    f1 = homogeneous_list_int
    f2 = epyccel( f1 , language=language)

    python_out = f1()
    pyccel_out = f2()
    print(pyccel_out)
    print(python_out)

    assert python_out == pyccel_out

def test_homogenous_list_bool_copy(language):
    def homogeneous_list_bool():
        return list([True, False, True, False])
    f1 = homogeneous_list_bool
    f2 = epyccel( f1 , language=language)

    python_out = f1()
    pyccel_out = f2()
    print(pyccel_out)
    print(python_out)

    assert python_out == pyccel_out

def test_homogenous_list_float_copy(language):
    def homogeneous_list_float():
        return list([1.0, 2.0, 3.0, 4.0])
    f1 = homogeneous_list_float
    f2 = epyccel( f1 , language=language)

    python_out = f1()
    pyccel_out = f2()
    print(pyccel_out)
    print(python_out)

    assert python_out == pyccel_out

def test_homogenous_list_int_tuple_copy(language):
    def homogeneous_list_int_tuple():
        return list((1, 2, 3, 4))
    f1 = homogeneous_list_int_tuple
    f2 = epyccel( f1 , language=language)

    python_out = f1()
    pyccel_out = f2()
    print(pyccel_out)
    print(python_out)

    assert python_out == pyccel_out

def test_homogenous_list_unknown_size_copy(language):
    def homogeneous_list_unknown_size_copy(n : int):
        a = (3,)*n
        b = list(a)
        return b[0]
    f1 = homogeneous_list_unknown_size_copy
    f2 = epyccel( f1 , language=language)

    python_out = f1(5)
    pyccel_out = f2(5)
    print(pyccel_out)
    print(python_out)

    assert python_out == pyccel_out
