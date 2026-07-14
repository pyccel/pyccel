# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
import pytest
from modules import tuples as tuples_module

from pyccel import epyccel
from pyccel.errors.errors import PyccelError

from utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_tuples_mod(language):
    return epyccel_module_with_fallback(tuples_module, language)

no_arg_tuple_funcs = (
    "homogeneous_tuple_int",
    "homogeneous_tuple_bool",
    "homogeneous_tuple_float",
    "homogeneous_tuple_math",
    "homogeneous_tuple_containing_var",
    "homogeneous_tuple_of_arrays",
    "inhomogeneous_tuple_1",
    "inhomogeneous_tuple_2",
    "inhomogeneous_tuple_3",
    "inhomogeneous_tuple_2_levels_1",
    "inhomogeneous_tuple_2_levels_2",
    "homogeneous_tuple_2_levels",
    "tuple_unpacking_1",
    "tuple_unpacking_2",
    "tuple_unpacking_5",
    "tuple_name_clash",
    "tuples_as_indexing_basic",
    "tuples_as_indexing_var",
    "tuple_multi_indexing_1",
    "tuple_multi_indexing_2",
    "tuple_inhomogeneous_return",
    "tuple_homogeneous_return",
    "tuple_arg_unpacking",
    "tuple_indexing_basic",
    "tuple_indexing_2d",
    "tuple_visitation_homogeneous",
    "tuples_homogeneous_have_pointers",
    "tuples_inhomogeneous_have_pointers",
    "tuples_homogeneous_copies_have_pointers",
    "tuples_inhomogeneous_copies_have_pointers",
    "tuples_mul_homogeneous",
    "tuples_mul_homogeneous2",
    "tuples_mul_homogeneous3",
    "tuples_mul_homogeneous4",
    "tuples_mul_homogeneous5",
    "tuples_mul_inhomogeneous",
    "tuples_mul_inhomogeneous2",
    "tuples_mul_homogeneous_2d",
    "tuples_mul_mixed_homogeneous_2d",
    "tuples_mul_inhomogeneous_2d",
    "tuples_add_homogeneous",
    "tuples_add_homogeneous_variables",
    "tuples_add_homogeneous_with_variables",
    "tuples_add_homogeneous_with_variables2",
    "tuples_add_inhomogeneous",
    "tuples_add_inhomogeneous_variables",
    "tuples_add_inhomogeneous_with_variables",
    "tuples_add_inhomogeneous_with_variables2",
    "tuples_add_mixed_homogeneous",
    "tuples_add_mixed_homogeneous_variables",
    "tuples_add_mixed_homogeneous_with_variables",
    "tuples_2d_sum",
    "tuples_func",
    "tuple_slice",
    "tuple_variable_index",
    "tuple_variable_slice",
    "tuple_negative_slice",
    "inhomogeneous_tuple_negative_slice",
    "inhomogeneous_tuple_var_negative_slice",
    "tuple_index",
    "tuple_homogeneous_int",
    "tuple_homogeneous_bool",
    "tuple_homogeneous_float",
    "tuple_homogeneous_math",
    "tuple_inhomogeneous_1",
    "tuple_inhomogeneous_2",
    "tuple_inhomogeneous_3",
    "tuple_homogeneous",
    "tuple_inhomogeneous",
    "tuple_multilevel_inhomogeneous",
    "annotated_tuple_inhomog_return",
    "annotated_tuple_homog_return",
    "tuple_return_unknown_length",
    "tuple_assignment",
    "return_1_elem_inhomog_tuple",
    "return_empty_tuple",
    "return_empty_int_tuple",
    "return_annotated_empty_tuple",
)


def compare_python_pyccel(p_output, f_output):
    if p_output is None:
        assert f_output is None
        return
    if not hasattr(p_output, "__len__"):
        p_output = [p_output]
    if not hasattr(f_output, "__len__"):
        f_output = [f_output]
    assert len(p_output) == len(f_output)

    for pth, pycc in zip(p_output, f_output):

        if isinstance(pth, np.ndarray):
            assert np.allclose(pth, pycc)

        elif isinstance(pth, bool):
            pycc_bool = pycc == 1
            assert pth == pycc_bool

        elif isinstance(pth, (int, str)):
            assert isinstance(pycc, type(pth))
            assert pth == pycc

        else:
            assert np.isclose(pth, pycc)


@pytest.mark.parametrize("test_func", no_arg_tuple_funcs)
def test_tuples(test_func, epyc_tuples_mod):
    f1 = getattr(tuples_module, test_func)
    f2 = getattr(epyc_tuples_mod, test_func)

    python_out = f1()
    pyccel_out = f2()
    compare_python_pyccel(python_out, pyccel_out)


@pytest.mark.parametrize(
    "test_func", ['tuple_unpacking_3', 'tuple_unpacking_4']
)
def test_tuples_with_2d_args(test_func, epyc_tuples_mod):
    f1 = getattr(tuples_module, test_func)
    f2 = getattr(epyc_tuples_mod, test_func)

    python_x = np.array(np.random.randint(100, size=(3, 4)), dtype=int)
    pyccel_x = python_x.copy()

    f1(python_x)
    f2(pyccel_x)
    np.allclose(python_x, pyccel_x)

@pytest.mark.parametrize(
    "language",
    [
        pytest.param("c", marks=[
            pytest.mark.c,
            pytest.mark.skip(
                reason="Can't save a list of strings (#459)",
                )
            ]),
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.fortran,
                pytest.mark.skip(
                reason="Can't save a list of strings (#459)",
                ),
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ],
)
def test_homogeneous_tuple_string(language):
    def homogeneous_tuple_string():
        ai = ("hello", "tuple", "world", "!!")
        i = 1
        return ai[0], ai[i], ai[2], ai[3]

    f1 = homogeneous_tuple_string
    f2 = epyccel(f1, language=language)

    python_out = f1()
    pyccel_out = f2()
    compare_python_pyccel(python_out, pyccel_out)


@pytest.mark.parametrize(
    "language",
    [
        pytest.param("c", marks=[
            pytest.mark.c,
            pytest.mark.skip(
                reason="Can't save a list of strings (#459)",
                )
            ]),
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.fortran,
                pytest.mark.skip(
                reason="Can't save a list of strings (#459)",
                ),
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ],
)
def test_tuple_homogeneous_string(language):
    def tuple_homogeneous_string():
        a = tuple(("hello", "tuple", "world", "!!"))
        i = 1
        return a[0], a[i], a[2], a[3], len(a)

    f1 = tuple_homogeneous_string
    f2 = epyccel(f1, language=language)

    python_out = f1()
    pyccel_out = f2()
    compare_python_pyccel(python_out, pyccel_out)

@pytest.mark.skip(reason="Can't iterate over an inhomogeneous tuple")
def test_tuple_visitation_inhomogeneous(language): 
    def tuple_visitation_inhomogeneous():
        ai = (1, 3.5, False)
        for a in ai:
            print(a)

    f1 = tuple_visitation_inhomogeneous
    f2 = epyccel(f1, language=language)

    python_out = f1()
    pyccel_out = f2()
    compare_python_pyccel(python_out, pyccel_out)

def test_homogeneous_tuples_of_bools_as_args(epyc_tuples_mod):
    my_tuple = tuples_module.homogeneous_tuples_of_bools_as_args
    epyc_func = epyc_tuples_mod.homogeneous_tuples_of_bools_as_args
    assert my_tuple((True, False, False)) == epyc_func((True, False, False))
    tuple_arg = (False, True, False, True, True, True)
    assert my_tuple(tuple_arg) == epyc_func(tuple_arg)


def test_homogeneous_tuples_of_ints_as_args(epyc_tuples_mod):
    my_tuple = tuples_module.homogeneous_tuples_of_ints_as_args
    epyc_func = epyc_tuples_mod.homogeneous_tuples_of_ints_as_args
    assert my_tuple((1, 2, 3)) == epyc_func((1, 2, 3))
    tuple_arg = (-1, 9, 20, -55, 23)
    assert my_tuple(tuple_arg) == epyc_func(tuple_arg)


def test_homogeneous_tuples_of_floats_as_args(epyc_tuples_mod):
    my_tuple = tuples_module.homogeneous_tuples_of_floats_as_args
    epyc_func = epyc_tuples_mod.homogeneous_tuples_of_floats_as_args
    assert my_tuple((1.0, 2.0, 3.0)) == epyc_func((1.0, 2.0, 3.0))
    tuple_arg = (-1.0, 9.0, 20.0, -55.3, 23.2)
    assert my_tuple(tuple_arg) == epyc_func(tuple_arg)


def test_homogeneous_tuples_of_complexes_as_args(epyc_tuples_mod):
    my_tuple = tuples_module.homogeneous_tuples_of_complexes_as_args
    epyc_func = epyc_tuples_mod.homogeneous_tuples_of_complexes_as_args
    assert my_tuple((1.0 + 4j, 2.0 - 2j, 3.0 + 0j)) == epyc_func(
        (1.0 + 4j, 2.0 - 2j, 3.0 + 0j)
    )
    tuple_arg = (1.0 + 4j, 2.0 - 2j, 3.0 + 0j, -23.12 - 4.4j)
    assert my_tuple(tuple_arg) == epyc_func(tuple_arg)


def test_homogeneous_tuples_of_numpy_ints_as_args(epyc_tuples_mod):
    my_tuple = tuples_module.homogeneous_tuples_of_numpy_ints_as_args
    epyc_func = epyc_tuples_mod.homogeneous_tuples_of_numpy_ints_as_args
    tuple_arg = (np.int8(1), np.int8(2), np.int8(3))
    assert my_tuple(tuple_arg) == epyc_func(tuple_arg)


def test_homogeneous_tuples_template_args(epyc_tuples_mod):
    my_tuple = tuples_module.homogeneous_tuples_template_args
    epyc_func = epyc_tuples_mod.homogeneous_tuples_template_args
    tuple_int_arg = (1, 2, 3)
    tuple_float_arg = (4.0, 5.0, 6.0)

    int_pyth = my_tuple(tuple_int_arg)
    int_epyc = epyc_func(tuple_int_arg)
    assert int_pyth == int_epyc
    assert isinstance(int_epyc[1], int)

    float_pyth = my_tuple(tuple_float_arg)
    float_epyc = epyc_func(tuple_float_arg)
    assert float_pyth == float_epyc
    assert isinstance(float_epyc[1], float)


def test_multi_level_tuple_arg(language):
    def my_tuple(a: "tuple[tuple[int,...],...]"):
        return len(a), len(a[0]), a[0][0], a[1][0], a[0][1], a[1][1]

    tuple_arg = ((1, 2), (3, 4))

    if language != "python":
        # Raises an error because tuples inside tuples may have different lengths
        # This could be removed once lists are supported as the tuples could then
        # be stored in lists instead of arrays.
        with pytest.raises(PyccelError):
            _ = epyccel(my_tuple, language=language)
    else:
        epyc_func = epyccel(my_tuple, language=language)

        assert my_tuple(tuple_arg) == epyc_func(tuple_arg)


def test_homogeneous_tuples_result(epyc_tuples_mod):
    my_tuple = tuples_module.homogeneous_tuples_result
    epyc_func = epyc_tuples_mod.homogeneous_tuples_result

    assert my_tuple() == epyc_func()
