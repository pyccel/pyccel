# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from modules import functionals
from numpy import arange, array, equal
from numpy.random import randint

from pyccel import epyccel
from utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_functionals_mod(language):
    return epyccel_module_with_fallback(functionals, language)


def compare_epyccel(f_name, epyc_functionals_mod, *args):
    f = getattr(functionals, f_name)
    f2 = getattr(epyc_functionals_mod, f_name)
    out1 = f(*args)
    out2 = f2(*args)
    assert equal(out1, out2).all()


def test_functional_for_1d_range(epyc_functionals_mod):
    compare_epyccel('functional_for_1d_range', epyc_functionals_mod)


def test_functional_for_overwrite_1d_range(epyc_functionals_mod):
    compare_epyccel('functional_for_overwrite_1d_range', epyc_functionals_mod)


def test_functional_for_1d_var(epyc_functionals_mod):
    y = array(randint(99, size=4), dtype=int)
    compare_epyccel('functional_for_1d_var', epyc_functionals_mod, y)


def test_functional_for_1d_const(epyc_functionals_mod):
    y = array(randint(99, size=4), dtype=int)
    z = randint(99)
    compare_epyccel('functional_for_1d_const', epyc_functionals_mod, y, z)


def test_functional_for_1d_const2(epyc_functionals_mod):
    compare_epyccel('functional_for_1d_const2', epyc_functionals_mod)


def test_functional_for_2d_range(epyc_functionals_mod):
    compare_epyccel('functional_for_2d_range', epyc_functionals_mod)


def test_functional_for_2d_var_range(epyc_functionals_mod):
    y = array(randint(99, size=3), dtype=int)
    compare_epyccel('functional_for_2d_var_range', epyc_functionals_mod, y)


def test_functional_for_2d_var_var(epyc_functionals_mod):
    y = array(randint(99, size=3), dtype=int)
    z = array(randint(99, size=2), dtype=int)
    compare_epyccel('functional_for_2d_var_var', epyc_functionals_mod, y, z)


def test_functional_for_2d_dependant_range(epyc_functionals_mod):
    compare_epyccel('functional_for_2d_dependant_range_1', epyc_functionals_mod)
    compare_epyccel('functional_for_2d_dependant_range_2', epyc_functionals_mod)
    compare_epyccel('functional_for_2d_dependant_range_3', epyc_functionals_mod)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.skip(reason="lists of tuples are not yes supported"),
                pytest.mark.fortran,
            ],
        ),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="lists of tuples are not yes supported"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_functional_for_2d_array_range(language):
    def functional_for_2d_array_range(idx: "int"):
        a = [(x1, y1, z1) for x1 in range(3) for y1 in range(x1, 5) for z1 in range(y1, 10)]
        return len(a), a[idx][0], a[idx][1], a[idx][2]

    idx = randint(28)
    f2 = epyccel(functional_for_2d_array_range, language=language)

    assert equal(functional_for_2d_array_range(idx), f2(idx)).all()


def test_functional_for_2d_range_const(epyc_functionals_mod):
    compare_epyccel('functional_for_2d_range_const', epyc_functionals_mod)


def test_functional_for_3d_range(epyc_functionals_mod):
    compare_epyccel('functional_for_3d_range', epyc_functionals_mod)


def test_unknown_length_functional(epyc_functionals_mod):
    y = array(randint(100, size=20), dtype=int)
    compare_epyccel('unknown_length_functional', epyc_functionals_mod, y)


def test_functional_with_enumerate(epyc_functionals_mod):
    compare_epyccel('functional_with_enumerate', epyc_functionals_mod)


def test_functional_with_enumerate_with_start(epyc_functionals_mod):
    compare_epyccel('functional_with_enumerate_with_start', epyc_functionals_mod)


def test_functional_with_condition(epyc_functionals_mod):
    compare_epyccel('functional_with_condition', epyc_functionals_mod)


def test_functional_with_zip(epyc_functionals_mod):
    compare_epyccel('functional_with_zip', epyc_functionals_mod)


def test_functional_with_multiple_zips(epyc_functionals_mod):
    compare_epyccel('functional_with_multiple_zips', epyc_functionals_mod)


def test_functional_filter_and_transform(epyc_functionals_mod):
    compare_epyccel('functional_with_condition', epyc_functionals_mod)


def test_functional_with_multiple_conditions(epyc_functionals_mod):
    compare_epyccel('functional_with_multiple_conditions', epyc_functionals_mod)


def test_functional_negative_indices(epyc_functionals_mod):
    compare_epyccel('functional_negative_indices', epyc_functionals_mod, arange(10))


def test_functional_reverse(epyc_functionals_mod):
    compare_epyccel('functional_reverse', epyc_functionals_mod, arange(4))


def test_functional_indexed_iterator(epyc_functionals_mod):
    compare_epyccel('functional_indexed_iterator', epyc_functionals_mod, arange(10))
