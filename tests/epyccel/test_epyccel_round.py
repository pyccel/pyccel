# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
import numpy as np
from numpy.random import randint

from pyccel.decorators import types
from pyccel.epyccel import epyccel

def test_round_int(language):
    @types('float')
    def round_int(x):
        return round(x)

    f = epyccel(round_int, language=language, developer_mode=True)
    x = randint(100) / 10

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round down
    x = 3.345

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round up
    x = 3.845

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round half
    x = 6.5

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

def test_negative_round_int(language):
    @types('float')
    def round_int(x):
        return round(x)

    f = epyccel(round_int, language=language, developer_mode=True)
    x = -randint(100) / 10

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round up
    x = -3.345

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round down
    x = -3.845

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round half
    x = -6.5

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

def test_round_ndigits(language):
    @types('float','int')
    def round_ndigits(x, i):
        return round(x,i)

    f = epyccel(round_ndigits, language=language, developer_mode=True)
    x = randint(100) / 10

    f_output = f(x, 1)
    round_ndigits_output = round_ndigits(x, 1)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3.343

    f_output = f(x,2)
    round_ndigits_output = round_ndigits(x,2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3323.0

    f_output = f(x,-2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = -3390.0

    f_output = f(x,-2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

@pytest.mark.parametrize( 'language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c',       marks = [
        pytest.mark.xfail(reason="Python implements bankers' round. But only for ndigits=0"),
        pytest.mark.c]),
    pytest.param('python', marks = pytest.mark.python),
    )
)
def test_round_ndigits_half(language):
    @types('float','int')
    def round_ndigits(x, i):
        return round(x,i)

    f = epyccel(round_ndigits, language=language, developer_mode=True)
    x = randint(100) / 10

    f_output = f(x, 1)
    round_ndigits_output = round_ndigits(x, 1)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3.345

    f_output = f(x,2)
    round_ndigits_output = round_ndigits(x,2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = -3350.0

    f_output = f(x,-2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 45.0

    f_output = f(x,-1)
    round_ndigits_output = round_ndigits(x,-1)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))
