# pylint: disable=missing-function-docstring, missing-module-docstring

from numpy.random import randint
from numpy import isclose
import numpy as np

from pyccel.epyccel import epyccel

RTOL = 1e-12
ATOL = 1e-16

def test_default_precision_template(language):

    @template('T', ['int[:]', 'float[:]', 'complex[:]'])
    def return_array_element(array : 'T'):
        return array[0]

    test_types = ['int', 'float', 'complex']
    f1 = return_array_element
    f2 = epyccel(f1, language=language)
    for t in test_types:
        d1 = randint(1, 15)
        arr = np.ones(d1, dtype=t)
        python_result = f1(arr)
        pyccel_result = f2(arr)

        assert isinstance(pyccel_result, type(python_result))
        assert isclose(pyccel_result, python_result, rtol=RTOL, atol=ATOL)
