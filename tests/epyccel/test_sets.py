# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from modules import sets

from pyccel import epyccel

from epyccel_utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_sets_mod(language):
    return epyccel_module_with_fallback(sets, language)


@pytest.fixture(
    params=[
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.skip(reason="set method not implemented in fortran"),
                pytest.mark.fortran,
            ],
        ),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="set method not implemented in c"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ],
    scope="module",
)
def python_only_language(request):
    return request.param


def test_add_literal_int(epyc_sets_mod):
    add_literal_int = sets.add_literal_int
    epyc_add_literal_int = epyc_sets_mod.add_literal_int
    pyccel_result = epyc_add_literal_int()
    python_result = add_literal_int()
    assert python_result == pyccel_result


def test_add_literal_complex(epyc_sets_mod):
    add_literal_complex = sets.add_literal_complex
    epyc_add_literal_complex = epyc_sets_mod.add_literal_complex
    pyccel_result = epyc_add_literal_complex()
    python_result = add_literal_complex()
    assert python_result == pyccel_result


def test_add_variable_int(epyc_sets_mod):
    add_variable_int = sets.add_variable_int
    epyc_add_variable_int = epyc_sets_mod.add_variable_int
    pyccel_result = epyc_add_variable_int()
    python_result = add_variable_int()
    assert python_result == pyccel_result


def test_clear_int(epyc_sets_mod):
    clear_int = sets.clear_int
    epyccel_clear = epyc_sets_mod.clear_int
    pyccel_result = epyccel_clear()
    python_result = clear_int()
    assert python_result == pyccel_result


def test_clear_float(epyc_sets_mod):
    clear_float = sets.clear_float
    epyccel_clear = epyc_sets_mod.clear_float
    pyccel_result = epyccel_clear()
    python_result = clear_float()
    assert python_result == pyccel_result


def test_clear_complex(epyc_sets_mod):
    clear_complex = sets.clear_complex
    epyccel_clear = epyc_sets_mod.clear_complex
    pyccel_result = epyccel_clear()
    python_result = clear_complex()
    assert python_result == pyccel_result


def test_copy_int(epyc_sets_mod):
    copy_int = sets.copy_int
    epyccel_copy = epyc_sets_mod.copy_int
    pyccel_result = epyccel_copy()
    python_result = copy_int()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result[0] == pyccel_result[0]
    assert python_result[1] == pyccel_result[1]
    assert set(python_result[2:]) == set(pyccel_result[2:])


def test_copy_float(epyc_sets_mod):
    copy_float = sets.copy_float
    epyccel_copy = epyc_sets_mod.copy_float
    pyccel_result = epyccel_copy()
    python_result = copy_float()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result[0] == pyccel_result[0]
    assert python_result[-1] == pyccel_result[-1]
    assert set(python_result[1:-1]) == set(pyccel_result[1:-1])


def test_copy_complex(epyc_sets_mod):
    copy_complex = sets.copy_complex
    epyccel_copy = epyc_sets_mod.copy_complex
    pyccel_result = epyccel_copy()
    python_result = copy_complex()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_remove_complex(epyc_sets_mod):
    remove_complex = sets.remove_complex
    epyccel_remove = epyc_sets_mod.remove_complex
    pyccel_result = epyccel_remove()
    python_result = remove_complex()
    assert python_result == pyccel_result


def test_remove_int(epyc_sets_mod):
    remove_int = sets.remove_int
    epyccel_remove = epyc_sets_mod.remove_int
    pyccel_result = epyccel_remove()
    python_result = remove_int()
    assert python_result == pyccel_result


def test_remove_float(epyc_sets_mod):
    remove_float = sets.remove_float
    epyccel_remove = epyc_sets_mod.remove_float
    pyccel_result = epyccel_remove()
    python_result = remove_float()
    assert python_result == pyccel_result


def test_Discard_int(epyc_sets_mod):
    Discard_int = sets.Discard_int
    epyccel_remove = epyc_sets_mod.Discard_int
    pyccel_result = epyccel_remove()
    python_result = Discard_int()
    assert python_result == pyccel_result


def test_Discard_complex(epyc_sets_mod):
    Discard_complex = sets.Discard_complex
    epyccel_remove = epyc_sets_mod.Discard_complex
    pyccel_result = epyccel_remove()
    python_result = Discard_complex()
    assert python_result == pyccel_result


def test_Discard_wrong_arg(epyc_sets_mod):
    Discard_wrong_arg = sets.Discard_wrong_arg
    epyccel_remove = epyc_sets_mod.Discard_wrong_arg
    pyccel_result = epyccel_remove()
    python_result = Discard_wrong_arg()
    assert python_result == pyccel_result


def test_update_basic(epyc_sets_mod):
    update_basic = sets.update_basic
    epyccel_update = epyc_sets_mod.update_basic
    pyccel_result = epyccel_update()
    python_result = update_basic()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_update_multiple(epyc_sets_mod):
    update_multiple = sets.update_multiple
    epyccel_update = epyc_sets_mod.update_multiple
    pyccel_result = epyccel_update()
    python_result = update_multiple()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_update_multiple_args(epyc_sets_mod):
    update_multiple = sets.update_multiple_args
    epyccel_update = epyc_sets_mod.update_multiple_args
    pyccel_result = epyccel_update()
    python_result = update_multiple()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_update_boolean_tuple(epyc_sets_mod):
    update_boolean_tuple = sets.update_boolean_tuple
    epyccel_update = epyc_sets_mod.update_boolean_tuple
    pyccel_result = epyccel_update()
    python_result = update_boolean_tuple()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_update_complex_list(epyc_sets_mod):
    update_complex_list = sets.update_complex_list
    epyccel_update = epyc_sets_mod.update_complex_list
    pyccel_result = epyccel_update()
    python_result = update_complex_list()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_update_range(epyc_sets_mod):
    update_range = sets.update_range
    epyccel_update = epyc_sets_mod.update_range
    pyccel_result = epyccel_update()
    python_result = update_range()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_update_set_as_arg(epyc_sets_mod):
    update_set_as_arg = sets.update_set_as_arg
    epyccel_update = epyc_sets_mod.update_set_as_arg
    pyccel_result = epyccel_update()
    python_result = update_set_as_arg()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_update_tuple_as_arg(epyc_sets_mod):
    update_tuple_as_arg = sets.update_tuple_as_arg
    epyccel_update = epyc_sets_mod.update_tuple_as_arg
    pyccel_result = epyccel_update()
    python_result = update_tuple_as_arg()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_with_list(epyc_sets_mod):
    set_With_list = sets.set_With_list
    epyc_set_With_list = epyc_sets_mod.set_With_list
    pyccel_result = epyc_set_With_list()
    python_result = set_With_list()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_set_with_tuple(epyc_sets_mod):
    set_With_tuple = sets.set_With_tuple
    epyc_set_With_tuple = epyc_sets_mod.set_With_tuple
    pyccel_result = epyc_set_With_tuple()
    python_result = set_With_tuple()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_set_with_set(epyc_sets_mod):
    set_With_set = sets.set_With_set
    epyc_set_With_set = epyc_sets_mod.set_With_set
    pyccel_result = epyc_set_With_set()
    python_result = set_With_set()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_init_with_set(epyc_sets_mod):
    init_with_set = sets.init_with_set
    epyc_init_with_set = epyc_sets_mod.init_with_set
    pyccel_result = epyc_init_with_set()
    python_result = init_with_set()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_init_with_list(epyc_sets_mod):
    init_with_list = sets.init_with_list
    epyc_init_with_list = epyc_sets_mod.init_with_list
    pyccel_result = epyc_init_with_list()
    python_result = init_with_list()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_copy_from_arg1(python_only_language):
    def copy_from_arg1(a: "list[float]"):
        b = set(a)
        return b

    a = [2.5, 1.4, 9.2]
    epyc_copy_from_arg = epyccel(copy_from_arg1, language=python_only_language)
    pyccel_result = epyc_copy_from_arg(a)
    python_result = copy_from_arg1(a)
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_set_copy_from_arg2(epyc_sets_mod):
    copy_from_arg2 = sets.copy_from_arg2
    a = {2.5, 1.4, 9.2}
    epyc_copy_from_arg = epyc_sets_mod.copy_from_arg2
    pyccel_result = epyc_copy_from_arg(a)
    python_result = copy_from_arg2(a)
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result


def test_Pop_int(epyc_sets_mod):
    Pop_int = sets.Pop_int
    epyccel_remove = epyc_sets_mod.Pop_int
    pyccel_result = set(epyccel_remove())
    python_result = set(Pop_int())
    assert python_result == pyccel_result


def test_Pop_float(epyc_sets_mod):
    Pop_float = sets.Pop_float
    epyccel_remove = epyc_sets_mod.Pop_float
    pyccel_result = set(epyccel_remove())
    python_result = set(Pop_float())
    assert python_result == pyccel_result


def test_Pop_complex(epyc_sets_mod):
    Pop_complex = sets.Pop_complex
    epyccel_remove = epyc_sets_mod.Pop_complex
    pyccel_result = set(epyccel_remove())
    python_result = set(Pop_complex())
    assert python_result == pyccel_result


def test_set_union_int(epyc_sets_mod):
    union_int = sets.union_int
    epyccel_func = epyc_sets_mod.union_int
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result == pyccel_result


def test_set_union_no_args(epyc_sets_mod):
    union_int = sets.set_union_no_args
    epyccel_func = epyc_sets_mod.set_union_no_args
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_union_2_args(epyc_sets_mod):
    union_int = sets.set_union_2_args
    epyccel_func = epyc_sets_mod.set_union_2_args
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_union_temporaries(epyc_sets_mod):
    union_int = sets.set_union_temporaries
    epyccel_func = epyc_sets_mod.set_union_temporaries
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="Can't use a pointer to a temporary object."),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_temporary_set_union(language):
    def union_int():
        a = {1, 2}
        b = {2}
        d = a.union(b).pop()
        return d

    epyccel_func = epyccel(union_int, language=language)
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result == pyccel_result


def test_temporary_set_union_2(epyc_sets_mod):
    union_int = sets.temporary_set_union_2
    epyccel_func = epyc_sets_mod.temporary_set_union_2
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result == pyccel_result


def test_set_union_list(epyc_sets_mod):
    union_list = sets.union_list
    epyccel_func = epyc_sets_mod.union_list
    pyccel_result = epyccel_func()
    python_result = union_list()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_union_tuple(epyc_sets_mod):
    union_tuple = sets.union_tuple
    epyccel_func = epyc_sets_mod.union_tuple
    pyccel_result = epyccel_func()
    python_result = union_tuple()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_union_operator(epyc_sets_mod):
    union_int = sets.set_union_operator
    epyccel_func = epyc_sets_mod.set_union_operator
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_union_augoperator(epyc_sets_mod):
    union_int = sets.set_union_augoperator
    epyccel_func = epyc_sets_mod.set_union_augoperator
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_intersection_int(epyc_sets_mod):
    intersection_int = sets.intersection_int
    epyccel_func = epyc_sets_mod.intersection_int
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_intersection_no_args(epyc_sets_mod):
    intersection_int = sets.set_intersection_no_args
    epyccel_func = epyc_sets_mod.set_intersection_no_args
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_intersection_2_args(epyc_sets_mod):
    intersection_int = sets.set_intersection_2_args
    epyccel_func = epyc_sets_mod.set_intersection_2_args
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_intersection_int_temporaries(epyc_sets_mod):
    intersection_int = sets.set_intersection_int_temporaries
    epyccel_func = epyc_sets_mod.set_intersection_int_temporaries
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_temporary_set_intersection(epyc_sets_mod):
    intersection_int = sets.temporary_set_intersection
    epyccel_func = epyc_sets_mod.temporary_set_intersection
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result == pyccel_result


def test_set_intersection_operator(epyc_sets_mod):
    intersection_int = sets.set_intersection_operator
    epyccel_func = epyc_sets_mod.set_intersection_operator
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_intersection_operator_2(epyc_sets_mod):
    intersection_int = sets.set_intersection_operator_2
    epyccel_func = epyc_sets_mod.set_intersection_operator_2
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert set(python_result) == set(pyccel_result)


def test_set_intersection_update(epyc_sets_mod):
    intersection_int = sets.set_intersection_update
    epyccel_func = epyc_sets_mod.set_intersection_update
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_intersection_multiple_update(epyc_sets_mod):
    intersection_int = sets.set_intersection_multiple_update
    epyccel_func = epyc_sets_mod.set_intersection_multiple_update
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_intersection_augoperator(epyc_sets_mod):
    intersection_int = sets.set_intersection_augoperator
    epyccel_func = epyc_sets_mod.set_intersection_augoperator
    pyccel_result = epyccel_func()
    python_result = intersection_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_contains(epyc_sets_mod):
    union_int = sets.set_contains
    epyccel_func = epyc_sets_mod.set_contains
    pyccel_result = epyccel_func()
    python_result = union_int()
    assert python_result == pyccel_result


def test_set_ptr(epyc_sets_mod):
    set_ptr = sets.set_ptr
    epyccel_func = epyc_sets_mod.set_ptr
    pyccel_result = epyccel_func()
    python_result = set_ptr()
    assert python_result == pyccel_result


def test_set_iter(epyc_sets_mod):
    set_sum_int = sets.set_sum_int
    epyccel_func = epyc_sets_mod.set_sum_int
    pyccel_result = epyccel_func()
    python_result = set_sum_int()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))


def test_set_iter_prod(epyc_sets_mod):
    set_iter_prod = sets.set_iter_prod
    epyccel_func = epyc_sets_mod.set_iter_prod
    pyccel_result = epyccel_func()
    python_result = set_iter_prod()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))


def test_set_const_arg(epyc_sets_mod):
    set_arg = sets.set_arg
    epyccel_func = epyc_sets_mod.set_arg
    int_arg = {1, 2, 3, 4, 5, 6, 7}
    float_arg = {1.5, 2.5, 3.5, 4.5, 6.7}
    complex_arg = {1 + 0j, 4j, 2.5 + 2j}
    for arg in (int_arg, float_arg, complex_arg):
        start = type(next(iter(arg)))(0)
        pyccel_result = epyccel_func(arg, start)
        python_result = set_arg(arg, start)
        assert python_result == pyccel_result
        assert isinstance(pyccel_result, type(python_result))


def test_set_arg(stc_language):
    def set_arg(arg: "set[int]", n: int):
        arg.update(range(n))

    epyccel_func = epyccel(set_arg, language=stc_language)
    arg_pyc = {7, 8, 9, 10}
    arg_pyt = arg_pyc.copy()
    n = 6
    epyccel_func(arg_pyc, n)
    set_arg(arg_pyt, n)
    assert arg_pyc == arg_pyt


def test_set_return(epyc_sets_mod):
    set_return = sets.set_return
    epyccel_func = epyc_sets_mod.set_return
    pyccel_result = epyccel_func()
    python_result = set_return()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))
    assert isinstance(python_result.pop(), type(pyccel_result.pop()))


def test_set_min_max(epyc_sets_mod):
    set_min_max = sets.set_min_max
    epyccel_func = epyc_sets_mod.set_min_max
    pyccel_result = epyccel_func()
    python_result = set_min_max()
    assert python_result == pyccel_result
    assert isinstance(python_result, type(pyccel_result))


def test_set_is_disjoint(epyc_sets_mod):
    set_is_disjoint = sets.set_is_disjoint
    epyccel_func = epyc_sets_mod.set_is_disjoint
    example_set1 = {1, 2, 3, 4}
    example_set2 = {5, 6, 7, 8}
    example_set3 = {7, 8, 2}
    assert set_is_disjoint(example_set1, example_set2) == epyccel_func(
        example_set1, example_set2
    )
    assert set_is_disjoint(example_set1, example_set3) == epyccel_func(
        example_set1, example_set3
    )
    assert set_is_disjoint(example_set3, example_set2) == epyccel_func(
        example_set3, example_set2
    )


def test_set_difference_int(epyc_sets_mod):
    difference_int = sets.difference_int
    epyccel_func = epyc_sets_mod.difference_int
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_difference_no_args(epyc_sets_mod):
    difference_int = sets.set_difference_no_args
    epyccel_func = epyc_sets_mod.set_difference_no_args
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_difference_2_args(epyc_sets_mod):
    difference_int = sets.set_difference_2_args
    epyccel_func = epyc_sets_mod.set_difference_2_args
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_difference_int_temporaries(epyc_sets_mod):
    difference_int = sets.set_difference_int_temporaries
    epyccel_func = epyc_sets_mod.set_difference_int_temporaries
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_temporary_set_difference(epyc_sets_mod):
    difference_int = sets.temporary_set_difference
    epyccel_func = epyc_sets_mod.temporary_set_difference
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result == pyccel_result


def test_set_difference_operator(epyc_sets_mod):
    difference_int = sets.set_difference_operator
    epyccel_func = epyc_sets_mod.set_difference_operator
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_difference_update(epyc_sets_mod):
    difference_int = sets.set_difference_update
    epyccel_func = epyc_sets_mod.set_difference_update
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_difference_multiple_update(epyc_sets_mod):
    difference_int = sets.set_difference_multiple_update
    epyccel_func = epyc_sets_mod.set_difference_multiple_update
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])


def test_set_difference_augoperator(epyc_sets_mod):
    difference_int = sets.set_difference_augoperator
    epyccel_func = epyc_sets_mod.set_difference_augoperator
    pyccel_result = epyccel_func()
    python_result = difference_int()
    assert python_result[0] == pyccel_result[0]
    assert set(python_result[1:]) == set(pyccel_result[1:])
