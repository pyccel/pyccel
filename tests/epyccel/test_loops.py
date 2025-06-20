# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
import numpy as np

from modules import loops

from pyccel import epyccel

#==============================================================================

def test_sum_natural_numbers(language):
    f1 = loops.sum_natural_numbers
    f2 = epyccel( f1, language = language )
    assert f1( 42 ) == f2( 42 )

def test_factorial(language):
    f1 = loops.factorial
    f2 = epyccel( f1, language = language )
    assert f1( 11 ) == f2( 11 )

def test_fibonacci(language):
    f1 = loops.fibonacci
    f2 = epyccel( f1, language = language )
    assert f1( 42 ) == f2( 42 )

def test_sum_nat_numbers_while(language):
    f1 = loops.sum_nat_numbers_while
    f2 = epyccel( f1, language = language )
    assert f1( 42 ) == f2( 42 )

def test_factorial_while(language):
    f1 = loops.factorial_while
    f2 = epyccel( f1, language = language )
    assert f1( 10 ) == f2( 10 )

def test_while_not_0(language):
    f1 = loops.while_not_0
    f2 = epyccel( f1, language = language )
    assert f1( 42 ) == f2( 42 )

def test_double_while_sum(language):
    f1 = loops.double_while_sum
    f2 = epyccel( f1, language = language )
    assert f1( 10, 10 ) == f2( 10, 10 )

def test_fibonacci_while(language):
    f1 = loops.fibonacci_while
    f2 = epyccel( f1, language = language )
    assert f1( 42 ) == f2( 42 )

def test_double_loop(language):
    f1 = loops.double_loop
    f2 = epyccel( f1, language = language )
    assert f1( 2 ) == f2( 2 )

def test_double_loop_on_2d_array_C(language):

    f1 = loops.double_loop_on_2d_array_C
    f2 = epyccel(f1, language = language)

    x = np.zeros( (11,4), dtype=int )
    y = np.ones ( (11,4), dtype=int )

    f1( x )
    f2( y )
    assert np.array_equal( x, y )

def test_double_loop_on_2d_array_F(language):

    f1 = loops.double_loop_on_2d_array_F
    f2 = epyccel(f1, language=language)

    x = np.zeros( (11,4), dtype=int, order='F' )
    y = np.ones ( (11,4), dtype=int, order='F' )

    f1( x )
    f2( y )
    assert np.array_equal( x, y )

def test_product_loop_on_2d_array_C(language):

    f1 = loops.product_loop_on_2d_array_C
    f2 = epyccel(f1, language=language)

    x = np.zeros( (11,4), dtype=int )
    y = np.ones ( (11,4), dtype=int )

    f1( x )
    f2( y )
    assert np.array_equal( x, y )

def test_product_loop_on_2d_array_F(language):

    f1 = loops.product_loop_on_2d_array_F
    f2 = epyccel(f1, language=language)

    x = np.zeros( (11,4), dtype=int, order='F' )
    y = np.ones ( (11,4), dtype=int, order='F' )

    f1( x )
    f2( y )
    assert np.array_equal( x, y )

def test_product_loop(language):

    f1 = loops.product_loop
    f2 = epyccel(f1, language=language)

    x = np.zeros( (44), dtype=float )
    y = np.zeros( (44), dtype=float )

    f1( x, 4, 11 )
    f2( y, 4, 11 )
    assert np.array_equal( x, y )

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Function in function not implemented in C", run=False),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_map_on_1d_array(language):

    f1 = loops.map_on_1d_array
    f2 = epyccel(f1, language=language)

    z = np.arange( 7 )

    assert np.array_equal( f1(z), f2(z) )

def test_enumerate_on_1d_array(language):

    f1 = loops.enumerate_on_1d_array
    f2 = epyccel(f1, language=language)

    z = np.arange( 7 )

    assert np.array_equal( f1(z), f2(z) )

def test_enumerate_on_1d_array_with_start(language):

    f1 = loops.enumerate_on_1d_array_with_start
    f2 = epyccel(f1, language=language)

    z = np.arange( 7 )

    assert np.array_equal( f1(z, 5), f2(z, 5) )
    assert np.array_equal( f1(z,-2), f2(z,-2) )

def test_enumerate_on_1d_array_with_tuple(language):

    f1 = loops.enumerate_on_1d_array_with_tuple
    f2 = epyccel(f1, language=language)

    z = np.arange( 7 )

    assert np.array_equal( f1(z), f2(z) )

def test_zip_prod(language):

    f1 = loops.zip_prod
    f2 = epyccel( f1, language = language )

    assert np.array_equal( f1(10), f2(10) )

def test_loop_on_real_array(language):

    f1 = loops.product_loop_on_real_array
    f2 = epyccel(f1, language=language)

    z1 = np.ones(11)
    out1 = np.empty_like(z1)
    z2 = z1.copy()
    out2 = out1.copy()

    f1(z1, out1)
    f2(z2, out2)

    assert np.array_equal( out1, out2 )

def test_for_loops(language):
    f1 = loops.for_loop1
    g1 = epyccel(f1, language=language)
    f2 = loops.for_loop2
    g2 = epyccel(f2, language=language)
    f3 = loops.for_loop2
    g3 = epyccel(f3, language=language)

    assert (f1(1,10,1) == g1(1,10,1))
    assert (f1(10,1,-1) == g1(10,1,-1))
    assert (f1(1, 10, 2) == g1(1, 10, 2))
    assert (f1(10, 1, -3) == g1(10, 1, -3))
    assert (f2() == g2())
    assert (f3() == g3())

def test_breaks(language):
    f1 = loops.fizzbuzz_search_with_breaks
    f2 = epyccel( f1, language = language )

    fizz = 2
    buzz = 3
    max_val = 12

    out1 = f1(fizz, buzz, max_val)
    out2 = f2(fizz, buzz, max_val)

    assert  out1 == out2

def test_continue(language):
    f1 = loops.fizzbuzz_sum_with_continue
    f2 = epyccel( f1, language = language )

    fizz = 2
    buzz = 3
    max_val = 12

    out1 = f1(fizz, buzz, max_val)
    out2 = f2(fizz, buzz, max_val)

    assert  out1 == out2

def test_temp_array_in_loop(language):
    f1 = loops.temp_array_in_loop
    f2 = epyccel( f1, language = language )

    a = np.zeros(6, dtype=int)
    b = np.zeros(6, dtype=int)

    c_py,d_py = f1(a,b)

    a[:] = 0
    b[:] = 0

    c_ep,d_ep = f2(a,b)

    assert np.array_equal(c_py, c_ep)
    assert np.array_equal(d_py, d_ep)

def test_less_than_100(language):
    f1 = loops.less_than_100
    f2 = epyccel( f1, language = language )

    assert f1(10) == f2(10)
    assert f1(101) == f2(101)

def test_for_expression(language):
    f1 = loops.for_expression
    f2 = epyccel( f1, language = language )

    assert f1() == f2()

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="lists of lists not yet implemented in Fortran. See #2210"),
            pytest.mark.fortran]
        ),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="lists of lists not yet implemented in C. See #2210"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_for_lists_of_lists(language):
    f1 = loops.for_lists_of_lists
    f2 = epyccel( f1, language = language )

    assert f1() == f2()
