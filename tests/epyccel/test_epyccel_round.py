# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types
from pyccel.epyccel import epyccel

def test_round_int(language):
    @types('real')
    def round_int(x):
        return round(x)

    f = epyccel(round_int, language=language)
    x = randint(100) / 10

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
