# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# coding: utf-8
""" Tests for headers. This ensures intermediate steps are tested before headers are deprecated.
Once headers are deprecated this file can be removed.
"""

import warnings
import pytest

from pyccel.epyccel import epyccel
from pyccel.errors.errors import PyccelSemanticError, Errors
from pyccel.decorators import allow_negative_index, stack_array


def test_local_type_annotation(language):
    def local_type_annotation():
        gift : int
        gift = 10
        return gift

    epyc_local_type_annotation = epyccel(local_type_annotation, language=language)
    assert epyc_local_type_annotation() == local_type_annotation()
    assert isinstance(epyc_local_type_annotation(), type(local_type_annotation()))

def test_local_wrong_type_annotation(language):
    def local_wrong_type_annotation():
        gift : float
        gift = 10
        return gift

    with pytest.raises(PyccelSemanticError):
        epyccel(local_wrong_type_annotation, language=language)

def test_allow_negative_index_annotation(language):
    @allow_negative_index('array')
    def allow_negative_index_annotation():
        import numpy as np
        array : 'int[:](order=C)'
        array = np.array([1,2,3,4,5])
        j = -3
        return array[j]

    errors = Errors()
    errors.reset()
    epyc_allow_negative_index_annotation = epyccel(allow_negative_index_annotation, language=language)
    assert errors.num_messages() == 1
    errors.reset()

    assert epyc_allow_negative_index_annotation() == allow_negative_index_annotation()
    assert isinstance(epyc_allow_negative_index_annotation(), type(allow_negative_index_annotation()))


def test_stack_array_annotation(language):
    @stack_array('array')
    def stack_array_annotation():
        import numpy as np
        array : 'int[:,:]'
        array = np.array([[1,2],[3,4],[5,6]])
        return array[2,0]

    epyc_stack_array_annotation = epyccel(stack_array_annotation, language=language)

    assert epyc_stack_array_annotation() == stack_array_annotation()
    assert isinstance(epyc_stack_array_annotation(), type(stack_array_annotation()))
