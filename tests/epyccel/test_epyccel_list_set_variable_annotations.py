# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# coding: utf-8
""" Tests for headers. This ensures intermediate steps are tested before headers are deprecated.
Once headers are deprecated this file can be removed.
"""

import pytest

from pyccel import epyccel

@pytest.fixture( params=[
        pytest.param("c", marks = pytest.mark.fortran),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="Variable declaration not implemented in fortran, related issues #1657 #1658"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ],
    scope = "module"
)
def language(request):
    return request.param

def test_homogeneous_empty_list_annotation_int(language):
    def homogeneous_set_annotation ():
        a : 'list[int]' #pylint: disable=unused-variable
        a = []
    epyc_homogeneous_set_annotation =  epyccel(homogeneous_set_annotation, language=language)
    assert epyc_homogeneous_set_annotation() == homogeneous_set_annotation()
    assert isinstance(epyc_homogeneous_set_annotation(), type(homogeneous_set_annotation()))

def test_homogeneous_set_annotation_int(language):
    def homogeneous_set_annotation ():
        a : 'set[int]' #pylint: disable=unused-variable
        a = {1, 2, 3, 4}
    epyc_homogeneous_set_annotation =  epyccel(homogeneous_set_annotation, language=language)
    assert epyc_homogeneous_set_annotation() == homogeneous_set_annotation()
    assert isinstance(epyc_homogeneous_set_annotation(), type(homogeneous_set_annotation()))

def test_homogeneous_set_without_annotation(language):
    def homogeneous_set():
        a = {1, 2, 3, 4} #pylint: disable=unused-variable
    epyc_homogeneous_set =  epyccel(homogeneous_set, language=language)
    assert epyc_homogeneous_set() == homogeneous_set()
    assert isinstance(epyc_homogeneous_set(), type(homogeneous_set()))

def test_homogeneous_set_annotation_float(language):
    def homogeneous_set_annotation ():
        a : 'set[float]' #pylint: disable=unused-variable
        a = {1.5, 2.5, 3.3, 4.3}
    epyc_homogeneous_set_annotation =  epyccel(homogeneous_set_annotation, language=language)
    assert epyc_homogeneous_set_annotation() == homogeneous_set_annotation()
    assert isinstance(epyc_homogeneous_set_annotation(), type(homogeneous_set_annotation()))

def test_homogeneous_set_annotation_bool(language):
    def homogeneous_set_annotation ():
        a : 'set[bool]' #pylint: disable=unused-variable
        a = {False, True, False, True} #pylint: disable=duplicate-value
    epyc_homogeneous_set_annotation =  epyccel(homogeneous_set_annotation, language=language)
    assert epyc_homogeneous_set_annotation() == homogeneous_set_annotation()
    assert isinstance(epyc_homogeneous_set_annotation(), type(homogeneous_set_annotation()))

def test_homogeneous_set_annotation_complex(language):
    def homogeneous_set_annotation():
        a: 'set[complex]'  # pylint: disable=unused-variable
        a = {1+1j, 2+2j, 3+3j, 4+4j}
    epyc_homogeneous_set_annotation = epyccel(homogeneous_set_annotation, language=language)
    assert epyc_homogeneous_set_annotation() == homogeneous_set_annotation()
    assert isinstance(epyc_homogeneous_set_annotation(), type(homogeneous_set_annotation()))

def test_homogeneous_list_annotation_int(language):
    def homogeneous_list_annotation():
        a: 'list[int]'  # pylint: disable=unused-variable
        a = [1, 2, 3, 4]
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))

def test_homogeneous_list_without_annotation(language):
    def homogeneous_list():
        a = [1, 2, 3, 4] # pylint: disable=unused-variable
    epyc_homogeneous_list = epyccel(homogeneous_list, language=language)
    assert epyc_homogeneous_list() == homogeneous_list()
    assert isinstance(epyc_homogeneous_list(), type(homogeneous_list()))

def test_homogeneous_list_annotation_float(language):
    def homogeneous_list_annotation():
        a: 'list[float]'  # pylint: disable=unused-variable
        a = [1.1, 2.2, 3.3, 4.4]
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))

def test_homogeneous_list_annotation_bool(language):
    def homogeneous_list_annotation():
        a: 'list[bool]'  # pylint: disable=unused-variable
        a = [False, True, True, False]
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))

def test_homogeneous_list_annotation_complex(language):
    def homogeneous_list_annotation():
        a: 'list[complex]'  # pylint: disable=unused-variable
        a = [1+1j, 2+2j, 3+3j, 4+4j]
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))

def test_homogeneous_list_annotation_embedded_complex(language):
    def homogeneous_list_annotation():
        a : 'list[complex]' = [1j, 2j]
        b = [a] # pylint: disable=unused-variable
    epyc_homogeneous_list_annotation = epyccel(homogeneous_list_annotation, language=language)
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation()))
