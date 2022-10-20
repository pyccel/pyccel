# pylint: disable=missing-function-docstring, missing-module-docstring

from numpy.random import randint
from numpy import isclose
import numpy as np

from pyccel.decorators import types
from pyccel.epyccel import epyccel


def test_default_precision_template(language):

    @types('int[:]')
    @types('float[:]')
    @types('complex[:]')
    def return_array_element(array):
        return array[0]

    test_types = [np.int64 , np.float64 , np.complex128, 'int', 'float', 'complex']
    f1 = return_array_element
    f2 = epyccel(f1, language=language)
    for t in test_types:
        d1 = randint(1, 15)
        arr = np.ones(d1, dtype=t)
        python_result = f1(arr)
        pyccel_result = f2(arr)

        assert type(pyccel_result) == type(python_result)
        assert isclose(pyccel_result, python_result)
