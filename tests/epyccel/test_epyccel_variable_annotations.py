# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# coding: utf-8
""" Tests for headers. This ensures intermediate steps are tested before headers are deprecated.
Once headers are deprecated this file can be removed.
"""

import pytest

from pyccel import epyccel
from pyccel.errors.errors import PyccelSemanticError
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

    epyc_allow_negative_index_annotation = epyccel(allow_negative_index_annotation, language=language)

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


def test_local_type_annotation_2(language):
    def local_type_annotation():
        gift : int = 10
        return gift

    epyc_local_type_annotation = epyccel(local_type_annotation, language=language)
    assert epyc_local_type_annotation() == local_type_annotation()
    assert isinstance(epyc_local_type_annotation(), type(local_type_annotation()))

def test_local_wrong_type_annotation_2(language):
    def local_wrong_type_annotation():
        gift : float = 10
        return gift

    with pytest.raises(PyccelSemanticError):
        epyccel(local_wrong_type_annotation, language=language)

def test_local_wrong_type_annotation_3(language):
    def local_wrong_type_annotation():
        gift : 'const float' = 10.5
        gift = 11.0
        return gift

    with pytest.raises(PyccelSemanticError):
        epyccel(local_wrong_type_annotation, language=language)

def test_allow_negative_index_annotation_2(language):
    @allow_negative_index('array')
    def allow_negative_index_annotation():
        import numpy as np
        array : 'int[:](order=C)' = np.array([1,2,3,4,5])
        j = -3
        return array[j]

    epyc_allow_negative_index_annotation = epyccel(allow_negative_index_annotation, language=language)

    assert epyc_allow_negative_index_annotation() == allow_negative_index_annotation()
    assert isinstance(epyc_allow_negative_index_annotation(), type(allow_negative_index_annotation()))


def test_stack_array_annotation_2(language):
    @stack_array('array')
    def stack_array_annotation():
        import numpy as np
        array : 'int[:,:]' = np.array([[1,2],[3,4],[5,6]])
        return array[2,0]

    epyc_stack_array_annotation = epyccel(stack_array_annotation, language=language)

    assert epyc_stack_array_annotation() == stack_array_annotation()
    assert isinstance(epyc_stack_array_annotation(), type(stack_array_annotation()))

def test_final_annotation(language):
    def final_annotation():
        from typing import Final
        a : Final[int] = 3
        a = 4
        return a

    with pytest.raises(PyccelSemanticError):
        epyccel(final_annotation, language=language)

def test_homogeneous_tuple_annotation(language):
    def homogeneous_tuple_annotation():
        # Not valid in Python 3.8
        a : tuple[int, ...] #pylint: disable=unsubscriptable-object
        a = (1,2,3)
        return a[0], a[1], a[2]

    epyc_homogeneous_tuple_annotation = epyccel(homogeneous_tuple_annotation, language=language)

    assert epyc_homogeneous_tuple_annotation() == homogeneous_tuple_annotation()
    assert isinstance(epyc_homogeneous_tuple_annotation(), type(homogeneous_tuple_annotation()))

def test_homogeneous_tuple_2_annotation(language):
    def homogeneous_tuple_annotation():
        # Not valid in Python 3.8
        a : tuple[tuple[int, ...], ...] #pylint: disable=unsubscriptable-object
        a = ((1,2,3), (4,5,6))
        return a[0][0], a[1][0], a[0][2]

    epyc_homogeneous_tuple_annotation = epyccel(homogeneous_tuple_annotation, language=language)

    assert epyc_homogeneous_tuple_annotation() == homogeneous_tuple_annotation()
    assert isinstance(epyc_homogeneous_tuple_annotation(), type(homogeneous_tuple_annotation()))

def test_homogeneous_tuple_annotation_str(language):
    def homogeneous_tuple_annotation():
        a : 'tuple[int, ...]'
        a = (1,2,3)
        return a[0], a[1], a[2]

    epyc_homogeneous_tuple_annotation = epyccel(homogeneous_tuple_annotation, language=language)

    assert epyc_homogeneous_tuple_annotation() == homogeneous_tuple_annotation()
    assert isinstance(epyc_homogeneous_tuple_annotation(), type(homogeneous_tuple_annotation()))

def test_homogeneous_tuple_2_annotation_str(language):
    def homogeneous_tuple_annotation():
        a : 'tuple[tuple[int, ...], ...]'
        a = ((1,2,3), (4,5,6))
        return a[0][0], a[1][0], a[0][2]

    epyc_homogeneous_tuple_annotation = epyccel(homogeneous_tuple_annotation, language=language)

    assert epyc_homogeneous_tuple_annotation() == homogeneous_tuple_annotation()
    assert isinstance(epyc_homogeneous_tuple_annotation(), type(homogeneous_tuple_annotation()))

def test_homogeneous_set_annotation_int(language):
    def homogeneous_set_annotation ():
        a : 'set[int]' #pylint: disable=unused-variable
        a = {1, 2, 3, 4}
        return len(a)
    epyc_homogeneous_set_annotation =  epyccel(homogeneous_set_annotation, language=language)
    assert epyc_homogeneous_set_annotation() == homogeneous_set_annotation()
    assert isinstance(epyc_homogeneous_set_annotation(), type(homogeneous_set_annotation()))

def test_homogeneous_set_without_annotation(language):
    def homogeneous_set():
        a = {1, 2, 3, 4} #pylint: disable=unused-variable
        return len(a)
    epyc_homogeneous_set =  epyccel(homogeneous_set, language=language)
    assert epyc_homogeneous_set() == homogeneous_set()
    assert isinstance(epyc_homogeneous_set(), type(homogeneous_set()))

def test_homogeneous_set_annotation_float(language):
    def homogeneous_set_annotation ():
        a : 'set[float]' #pylint: disable=unused-variable
        a = {1.5, 2.5, 3.3, 4.3}
        return len(a)
    epyc_homogeneous_set_annotation =  epyccel(homogeneous_set_annotation, language=language)
    assert epyc_homogeneous_set_annotation() == homogeneous_set_annotation()
    assert isinstance(epyc_homogeneous_set_annotation(), type(homogeneous_set_annotation()))

def test_homogeneous_set_annotation_bool(language):
    def homogeneous_set_annotation ():
        a : 'set[bool]' #pylint: disable=unused-variable
        a = {False, True, False, True} #pylint: disable=duplicate-value
        return len(a)
    epyc_homogeneous_set_annotation =  epyccel(homogeneous_set_annotation, language=language)
    assert epyc_homogeneous_set_annotation() == homogeneous_set_annotation()
    assert isinstance(epyc_homogeneous_set_annotation(), type(homogeneous_set_annotation()))

def test_homogeneous_set_annotation_complex(language):
    def homogeneous_set_annotation():
        a: 'set[complex]'  # pylint: disable=unused-variable
        a = {1+1j, 2+2j, 3+3j, 1-1j}
        return len(a)
    epyc_homogeneous_set_annotation = epyccel(homogeneous_set_annotation, language=language)
    assert epyc_homogeneous_set_annotation() == homogeneous_set_annotation()
    assert isinstance(epyc_homogeneous_set_annotation(), type(homogeneous_set_annotation()))

def test_empty_homogeneous_set_annotation_int(language):
    def homogeneous_set_annotation ():
        a : 'set[int]' #pylint: disable=unused-variable
        a = set()
        return len(a)
    epyc_homogeneous_set_annotation =  epyccel(homogeneous_set_annotation, language=language)
    assert epyc_homogeneous_set_annotation() == homogeneous_set_annotation()
    assert isinstance(epyc_homogeneous_set_annotation(), type(homogeneous_set_annotation()))

def test_homogeneous_empty_list_annotation_int(language):
    def homogeneous_list_annotation():
        a: 'list[int]'  # pylint: disable=unused-variable
        a = []
        return len(a)
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))

def test_homogeneous_empty_list_2_annotation_int(language):
    def homogeneous_list_annotation():
        a: 'list[int]'  # pylint: disable=unused-variable
        a = list() #pylint: disable=use-list-literal
        return len(a)
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))

def test_homogeneous_list_annotation_int(language):
    def homogeneous_list_annotation():
        a: 'list[int]'  # pylint: disable=unused-variable
        a = [1, 2, 3, 4]
        return a[0], a[1], a[2], a[3]
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))

def test_homogeneous_list_without_annotation(language):
    def homogeneous_list():
        a = [1, 2, 3, 4] # pylint: disable=unused-variable
        return a[0], a[1], a[2], a[3]
    epyc_homogeneous_list = epyccel(homogeneous_list, language=language)
    assert epyc_homogeneous_list() == homogeneous_list()
    assert isinstance(epyc_homogeneous_list(), type(homogeneous_list()))

def test_homogeneous_list_annotation_float(language):
    def homogeneous_list_annotation():
        a: 'list[float]'  # pylint: disable=unused-variable
        a = [1.1, 2.2, 3.3, 4.4]
        return a[0], a[1], a[2], a[3]
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))

def test_homogeneous_list_annotation_float64(language):
    def homogeneous_list_annotation():
        from numpy import float64
        a: 'list[float64]'  # pylint: disable=unused-variable
        a = [1.1, 2.2, 3.3, 4.4]
        return a[0], a[1], a[2], a[3]
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))
    assert isinstance(epyc_homogeneous_list_annotation()[0], type(homogeneous_list_annotation()[0]))

def test_homogeneous_list_annotation_bool(language):
    def homogeneous_list_annotation():
        a: 'list[bool]'  # pylint: disable=unused-variable
        a = [False, True, True, False]
        return a[0], a[1], a[2], a[3]
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))

def test_homogeneous_list_annotation_complex(language):
    def homogeneous_list_annotation():
        a: 'list[complex]'  # pylint: disable=unused-variable
        a = [1+1j, 2+2j, 3+3j, 4+4j]
        return a[0], a[1], a[2], a[3]
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))

def test_homogeneous_list_annotation_embedded_complex(stc_language):
    def homogeneous_list_annotation():
        a : 'list[complex]' = [1j, 2j]
        b = [a] # pylint: disable=unused-variable
        return b[0][0]
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=stc_language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))

def test_dict_int_float(language):
    def dict_int_float():
        # Not valid in Python 3.8
        a : dict[int, float] #pylint: disable=unsubscriptable-object,unused-variable
        a = {1:1.0, 2:2.0}
        return len(a)

    epyc_dict_int_float = epyccel(dict_int_float, language = language)
    assert epyc_dict_int_float() == dict_int_float()

def test_dict_empty_init(language):
    def dict_empty_init():
        # Not valid in Python 3.8
        a : dict[int, float] #pylint: disable=unsubscriptable-object,unused-variable
        a = {}
        return len(a)

    epyc_dict_empty_init = epyccel(dict_empty_init, language = language)
    assert epyc_dict_empty_init() == dict_empty_init()

def test_dict_complex_float(language):
    def dict_int_float():
        # Not valid in Python 3.8
        a : dict[complex, float] #pylint: disable=unsubscriptable-object,unused-variable
        a = {1j:1.0, -1j:2.0}
        return len(a)

    epyc_dict_int_float = epyccel(dict_int_float, language = language)
    assert epyc_dict_int_float() == dict_int_float()
