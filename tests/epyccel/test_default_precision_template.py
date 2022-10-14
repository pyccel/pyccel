# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy.random import rand, randint, uniform
from numpy import isclose, iinfo, finfo
import numpy as np

from pyccel.decorators import types, template
from pyccel.epyccel import epyccel

test_types = ['int', 'float', 'complex']
def test_default_precision_template(language):
    @template('T', types=['int64[:]', 'float[:]', 'complex[:]'])
    @types('T')
    def return_array_element(array):
        return array[0]
    
    f1 = return_array_element
    f2 = epyccel(f1, language=language)
    
    for t in test_types:
        d1 = randint(1, 15)
        arr = np.ones(d1).astype(t)
        python_result = f1(arr)
        pyccel_result = f2(arr)
        
        assert isclose(pyccel_result, python_result)
        
    