# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
import pytest
from modules import loops
from utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_loops_mod(language):
    return epyccel_module_with_fallback(loops, language)


# ==============================================================================


def test_sum_natural_numbers(epyc_loops_mod):
    f1 = loops.sum_natural_numbers
    f2 = epyc_loops_mod.sum_natural_numbers
    assert f1(42) == f2(42)


def test_factorial(epyc_loops_mod):
    f1 = loops.factorial
    f2 = epyc_loops_mod.factorial
    assert f1(11) == f2(11)


def test_fibonacci(epyc_loops_mod):
    f1 = loops.fibonacci
    f2 = epyc_loops_mod.fibonacci
    assert f1(42) == f2(42)


def test_sum_nat_numbers_while(epyc_loops_mod):
    f1 = loops.sum_nat_numbers_while
    f2 = epyc_loops_mod.sum_nat_numbers_while
    assert f1(42) == f2(42)


def test_factorial_while(epyc_loops_mod):
    f1 = loops.factorial_while
    f2 = epyc_loops_mod.factorial_while
    assert f1(10) == f2(10)


def test_while_not_0(epyc_loops_mod):
    f1 = loops.while_not_0
    f2 = epyc_loops_mod.while_not_0
    assert f1(42) == f2(42)


def test_double_while_sum(epyc_loops_mod):
    f1 = loops.double_while_sum
    f2 = epyc_loops_mod.double_while_sum
    assert f1(10, 10) == f2(10, 10)


def test_fibonacci_while(epyc_loops_mod):
    f1 = loops.fibonacci_while
    f2 = epyc_loops_mod.fibonacci_while
    assert f1(42) == f2(42)


def test_double_loop(epyc_loops_mod):
    f1 = loops.double_loop
    f2 = epyc_loops_mod.double_loop
    assert f1(2) == f2(2)


def test_double_loop_on_2d_array_C(epyc_loops_mod):

    f1 = loops.double_loop_on_2d_array_C
    f2 = epyc_loops_mod.double_loop_on_2d_array_C

    x = np.zeros((11, 4), dtype=int)
    y = np.ones((11, 4), dtype=int)

    f1(x)
    f2(y)
    assert np.array_equal(x, y)


def test_double_loop_on_2d_array_F(epyc_loops_mod):

    f1 = loops.double_loop_on_2d_array_F
    f2 = epyc_loops_mod.double_loop_on_2d_array_F

    x = np.zeros((11, 4), dtype=int, order="F")
    y = np.ones((11, 4), dtype=int, order="F")

    f1(x)
    f2(y)
    assert np.array_equal(x, y)


def test_product_loop_on_2d_array_C(epyc_loops_mod):

    f1 = loops.product_loop_on_2d_array_C
    f2 = epyc_loops_mod.product_loop_on_2d_array_C

    x = np.zeros((11, 4), dtype=int)
    y = np.ones((11, 4), dtype=int)

    f1(x)
    f2(y)
    assert np.array_equal(x, y)


def test_product_loop_on_2d_array_F(epyc_loops_mod):

    f1 = loops.product_loop_on_2d_array_F
    f2 = epyc_loops_mod.product_loop_on_2d_array_F

    x = np.zeros((11, 4), dtype=int, order="F")
    y = np.ones((11, 4), dtype=int, order="F")

    f1(x)
    f2(y)
    assert np.array_equal(x, y)


def test_product_loop(epyc_loops_mod):

    f1 = loops.product_loop
    f2 = epyc_loops_mod.product_loop

    x = np.zeros((44), dtype=float)
    y = np.zeros((44), dtype=float)

    f1(x, 4, 11)
    f2(y, 4, 11)
    assert np.array_equal(x, y)


def test_map_on_1d_array(epyc_loops_mod):

    f1 = loops.map_on_1d_array
    f2 = epyc_loops_mod.map_on_1d_array

    z = np.arange(7)

    assert np.array_equal(f1(z), f2(z))


def test_enumerate_on_1d_array(epyc_loops_mod):

    f1 = loops.enumerate_on_1d_array
    f2 = epyc_loops_mod.enumerate_on_1d_array

    z = np.arange(7)

    assert np.array_equal(f1(z), f2(z))


def test_enumerate_on_1d_array_with_start(epyc_loops_mod):

    f1 = loops.enumerate_on_1d_array_with_start
    f2 = epyc_loops_mod.enumerate_on_1d_array_with_start

    z = np.arange(7)

    assert np.array_equal(f1(z, 5), f2(z, 5))
    assert np.array_equal(f1(z, -2), f2(z, -2))


def test_enumerate_on_1d_array_with_tuple(epyc_loops_mod):

    f1 = loops.enumerate_on_1d_array_with_tuple
    f2 = epyc_loops_mod.enumerate_on_1d_array_with_tuple

    z = np.arange(7)

    assert np.array_equal(f1(z), f2(z))


def test_zip_prod(epyc_loops_mod):

    f1 = loops.zip_prod
    f2 = epyc_loops_mod.zip_prod

    assert np.array_equal(f1(10), f2(10))


def test_loop_on_real_array(epyc_loops_mod):

    f1 = loops.product_loop_on_real_array
    f2 = epyc_loops_mod.product_loop_on_real_array

    z1 = np.ones(11)
    out1 = np.empty_like(z1)
    z2 = z1.copy()
    out2 = out1.copy()

    f1(z1, out1)
    f2(z2, out2)

    assert np.array_equal(out1, out2)


def test_for_loops(epyc_loops_mod):
    f1 = loops.for_loop1
    g1 = epyc_loops_mod.for_loop1
    f2 = loops.for_loop2
    g2 = epyc_loops_mod.for_loop2
    f3 = loops.for_loop2
    g3 = epyc_loops_mod.for_loop2

    assert f1(1, 10, 1) == g1(1, 10, 1)
    assert f1(10, 1, -1) == g1(10, 1, -1)
    assert f1(1, 10, 2) == g1(1, 10, 2)
    assert f1(10, 1, -3) == g1(10, 1, -3)
    assert f2() == g2()
    assert f3() == g3()


def test_breaks(epyc_loops_mod):
    f1 = loops.fizzbuzz_search_with_breaks
    f2 = epyc_loops_mod.fizzbuzz_search_with_breaks

    fizz = 2
    buzz = 3
    max_val = 12

    out1 = f1(fizz, buzz, max_val)
    out2 = f2(fizz, buzz, max_val)

    assert out1 == out2


def test_continue(epyc_loops_mod):
    f1 = loops.fizzbuzz_sum_with_continue
    f2 = epyc_loops_mod.fizzbuzz_sum_with_continue

    fizz = 2
    buzz = 3
    max_val = 12

    out1 = f1(fizz, buzz, max_val)
    out2 = f2(fizz, buzz, max_val)

    assert out1 == out2


def test_temp_array_in_loop(epyc_loops_mod):
    f1 = loops.temp_array_in_loop
    f2 = epyc_loops_mod.temp_array_in_loop

    a = np.zeros(6, dtype=int)
    b = np.zeros(6, dtype=int)

    c_py, d_py = f1(a, b)

    a[:] = 0
    b[:] = 0

    c_ep, d_ep = f2(a, b)

    assert np.array_equal(c_py, c_ep)
    assert np.array_equal(d_py, d_ep)


def test_less_than_100(epyc_loops_mod):
    f1 = loops.less_than_100
    f2 = epyc_loops_mod.less_than_100

    assert f1(10) == f2(10)
    assert f1(101) == f2(101)


def test_for_expression(epyc_loops_mod):
    f1 = loops.for_expression
    f2 = epyc_loops_mod.for_expression

    assert f1() == f2()


@pytest.mark.parametrize(
    "language",
    (
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.skip(
                    reason="lists of lists not yet implemented in Fortran. Types defined in other modules"
                ),
                pytest.mark.fortran,
            ],
        ),
        pytest.param("c", marks=pytest.mark.c),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_for_lists_of_lists(language):
    def for_lists_of_lists():
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        c = [0, 0]
        for ai, bi in zip(a, b):
            for i in range(2):
                c[i] = ai[i] + bi[i]
                bi[i] = -1

        return c[0], c[1], b[0][0], b[0][1], b[1][0], b[1][1]

    f1 = for_lists_of_lists
    f2 = epyccel(f1, language=language, flags="-Werror")

    assert f1() == f2()


def test_for_unknown_index_slice(epyc_loops_mod):
    f1 = loops.for_unknown_index_slice
    f2 = epyc_loops_mod.for_unknown_index_slice

    assert f1() == f2()
