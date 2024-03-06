# pylint: disable=missing-function-docstring, missing-module-docstring

import pytest
import numpy as np

from pyccel.epyccel import epyccel

from pyccel.decorators import kernel


#------------------------------------------------------------------------------
@pytest.fixture(params=[
        pytest.param("cuda", marks = pytest.mark.cuda),
    ]
)
def language(request):
    return request.param

#==============================================================================

@pytest.mark.gpu
def test_kernel(language):
    @kernel
    def add_one_kernel(a: 'int[:]'):
        a[0] += 1

    def f():
        a = np.array([1], dtype=np.int32)
        add_one_kernel[1, 1](a)
        return a[0]

    epyc_f = epyccel(f, language=language)
    res = epyc_f()
    assert res == 2
