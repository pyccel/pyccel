# pylint: disable=missing-function-docstring, missing-module-docstring/
from numpy.random import randint

from pyccel.decorators import types
from pyccel.epyccel import epyccel

def test_round_int(language):
    @types('real')
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

def test_negative_round_int(language):
    @types('real')
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

#TODO: Add float tests
