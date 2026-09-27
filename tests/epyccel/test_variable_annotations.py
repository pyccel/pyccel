# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# coding: utf-8
"""Tests for headers. This ensures intermediate steps are tested before headers are deprecated.
Once headers are deprecated this file can be removed.
"""

from typing import Annotated, Final

import pytest
from modules import variable_annotations

from pyccel import epyccel
from pyccel.errors.errors import Errors, PyccelSemanticError

from epyccel_utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_variable_annotations_mod(language):
    return epyccel_module_with_fallback(variable_annotations, language)


def test_local_type_annotation(epyc_variable_annotations_mod):
    local_type_annotation = variable_annotations.local_type_annotation
    epyc_local_type_annotation = epyc_variable_annotations_mod.local_type_annotation
    assert epyc_local_type_annotation() == local_type_annotation()
    assert isinstance(epyc_local_type_annotation(), type(local_type_annotation()))


def test_local_wrong_type_annotation(language):
    def local_wrong_type_annotation():
        gift: float
        gift = 10
        return gift

    with pytest.raises(PyccelSemanticError):
        epyccel(local_wrong_type_annotation, language=language)


def test_allow_negative_index_annotation(epyc_variable_annotations_mod):
    allow_negative_index_annotation = variable_annotations.allow_negative_index_annotation
    epyc_allow_negative_index_annotation = (
        epyc_variable_annotations_mod.allow_negative_index_annotation
    )

    assert epyc_allow_negative_index_annotation() == allow_negative_index_annotation()
    assert isinstance(
        epyc_allow_negative_index_annotation(), type(allow_negative_index_annotation())
    )


def test_stack_array_annotation(epyc_variable_annotations_mod):
    stack_array_annotation = variable_annotations.stack_array_annotation
    epyc_stack_array_annotation = epyc_variable_annotations_mod.stack_array_annotation

    assert epyc_stack_array_annotation() == stack_array_annotation()
    assert isinstance(epyc_stack_array_annotation(), type(stack_array_annotation()))


def test_local_type_annotation_2(epyc_variable_annotations_mod):
    local_type_annotation = variable_annotations.local_type_annotation_2
    epyc_local_type_annotation = epyc_variable_annotations_mod.local_type_annotation_2
    assert epyc_local_type_annotation() == local_type_annotation()
    assert isinstance(epyc_local_type_annotation(), type(local_type_annotation()))


def test_local_wrong_type_annotation_2(language):
    def local_wrong_type_annotation():
        gift: float = 10
        return gift

    with pytest.raises(PyccelSemanticError):
        epyccel(local_wrong_type_annotation, language=language)


def test_local_wrong_type_annotation_3(language):
    def local_wrong_type_annotation():
        gift: Final[float] = 10.5
        gift = 11.0
        return gift

    with pytest.raises(PyccelSemanticError):
        epyccel(local_wrong_type_annotation, language=language)


def test_allow_negative_index_annotation_2(epyc_variable_annotations_mod):
    allow_negative_index_annotation = variable_annotations.allow_negative_index_annotation_2
    epyc_allow_negative_index_annotation = (
        epyc_variable_annotations_mod.allow_negative_index_annotation_2
    )

    assert epyc_allow_negative_index_annotation() == allow_negative_index_annotation()
    assert isinstance(
        epyc_allow_negative_index_annotation(), type(allow_negative_index_annotation())
    )


def test_stack_array_annotation_2(epyc_variable_annotations_mod):
    stack_array_annotation = variable_annotations.stack_array_annotation_2
    epyc_stack_array_annotation = epyc_variable_annotations_mod.stack_array_annotation_2

    assert epyc_stack_array_annotation() == stack_array_annotation()
    assert isinstance(epyc_stack_array_annotation(), type(stack_array_annotation()))


def test_final_annotation(language):
    def final_annotation():
        a: Final[int] = 3
        a = 4
        return a

    with pytest.raises(PyccelSemanticError):
        epyccel(final_annotation, language=language)


def test_final_annotation_transmission(epyc_variable_annotations_mod):
    final_annotation = variable_annotations.final_annotation
    epyc_final_annotation = epyc_variable_annotations_mod.final_annotation
    assert final_annotation() == epyc_final_annotation()


def test_homogeneous_tuple_annotation(epyc_variable_annotations_mod):
    homogeneous_tuple_annotation = variable_annotations.homogeneous_tuple_annotation
    epyc_homogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.homogeneous_tuple_annotation
    )

    orig_output = homogeneous_tuple_annotation()
    epyc_output = epyc_homogeneous_tuple_annotation()

    assert epyc_output == orig_output
    assert isinstance(
        epyc_output, type(orig_output)
    )
    assert isinstance(
        epyc_output[0], type(orig_output[0])
    )


def test_homogeneous_tuple_2_annotation(epyc_variable_annotations_mod):
    homogeneous_tuple_annotation = variable_annotations.homogeneous_tuple_2_annotation
    epyc_homogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.homogeneous_tuple_2_annotation
    )

    orig_output = homogeneous_tuple_annotation()
    epyc_output = epyc_homogeneous_tuple_annotation()

    assert epyc_output == orig_output
    assert isinstance(
        epyc_output, type(orig_output)
    )
    assert isinstance(
        epyc_output[0], type(orig_output[0])
    )


def test_homogeneous_tuple_annotation_str(epyc_variable_annotations_mod):
    homogeneous_tuple_annotation = variable_annotations.homogeneous_tuple_annotation_str
    epyc_homogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.homogeneous_tuple_annotation_str
    )

    orig_output = homogeneous_tuple_annotation()
    epyc_output = epyc_homogeneous_tuple_annotation()

    assert epyc_output == orig_output
    assert isinstance(
        epyc_output, type(orig_output)
    )
    assert isinstance(
        epyc_output[0], type(orig_output[0])
    )


def test_homogeneous_tuple_2_annotation_str(epyc_variable_annotations_mod):
    homogeneous_tuple_annotation = (
        variable_annotations.homogeneous_tuple_2_annotation_str
    )
    epyc_homogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.homogeneous_tuple_2_annotation_str
    )

    orig_output = homogeneous_tuple_annotation()
    epyc_output = epyc_homogeneous_tuple_annotation()

    assert epyc_output == orig_output
    assert isinstance(
        epyc_output, type(orig_output)
    )
    assert isinstance(
        epyc_output[0], type(orig_output[0])
    )


def test_homogeneous_set_annotation_int(epyc_variable_annotations_mod):
    homogeneous_set_annotation = variable_annotations.homogeneous_set_annotation_int
    epyc_homogeneous_set_annotation = (
        epyc_variable_annotations_mod.homogeneous_set_annotation_int
    )

    orig_output = homogeneous_set_annotation()
    epyc_output = epyc_homogeneous_set_annotation()

    assert orig_output == epyc_output
    assert isinstance(epyc_output, type(orig_output))
    assert isinstance(epyc_output.pop(), type(orig_output.pop()))


def test_homogeneous_set_without_annotation(epyc_variable_annotations_mod):
    homogeneous_set = variable_annotations.homogeneous_set_without_annotation
    epyc_homogeneous_set = epyc_variable_annotations_mod.homogeneous_set_without_annotation

    orig_output = homogeneous_set()
    epyc_output = epyc_homogeneous_set()

    assert orig_output == epyc_output
    assert isinstance(epyc_output, type(orig_output))
    assert isinstance(epyc_output.pop(), type(orig_output.pop()))


def test_homogeneous_set_annotation_float(epyc_variable_annotations_mod):
    homogeneous_set_annotation = variable_annotations.homogeneous_set_annotation_float
    epyc_homogeneous_set_annotation = (
        epyc_variable_annotations_mod.homogeneous_set_annotation_float
    )

    orig_output = homogeneous_set_annotation()
    epyc_output = epyc_homogeneous_set_annotation()

    assert orig_output == epyc_output
    assert isinstance(epyc_output, type(orig_output))
    assert isinstance(epyc_output.pop(), type(orig_output.pop()))


def test_homogeneous_set_annotation_bool(epyc_variable_annotations_mod):
    homogeneous_set_annotation = variable_annotations.homogeneous_set_annotation_bool
    epyc_homogeneous_set_annotation = (
        epyc_variable_annotations_mod.homogeneous_set_annotation_bool
    )

    orig_output = homogeneous_set_annotation()
    epyc_output = epyc_homogeneous_set_annotation()

    assert orig_output == epyc_output
    assert isinstance(epyc_output, type(orig_output))
    assert isinstance(epyc_output.pop(), type(orig_output.pop()))


def test_homogeneous_set_annotation_complex(epyc_variable_annotations_mod):
    homogeneous_set_annotation = variable_annotations.homogeneous_set_annotation_complex
    epyc_homogeneous_set_annotation = (
        epyc_variable_annotations_mod.homogeneous_set_annotation_complex
    )

    orig_output = homogeneous_set_annotation()
    epyc_output = epyc_homogeneous_set_annotation()

    assert orig_output == epyc_output
    assert isinstance(epyc_output, type(orig_output))
    assert isinstance(epyc_output.pop(), type(orig_output.pop()))


def test_empty_homogeneous_set_annotation_int(epyc_variable_annotations_mod):
    homogeneous_set_annotation = (
        variable_annotations.empty_homogeneous_set_annotation_int
    )
    epyc_homogeneous_set_annotation = (
        epyc_variable_annotations_mod.empty_homogeneous_set_annotation_int
    )

    orig_output = homogeneous_set_annotation()
    epyc_output = epyc_homogeneous_set_annotation()

    assert orig_output == epyc_output
    assert isinstance(epyc_output, type(orig_output))


def test_homogeneous_empty_list_annotation_int(epyc_variable_annotations_mod):
    homogeneous_list_annotation = variable_annotations.homogeneous_empty_list_annotation_int
    epyc_homogeneous_list_annotation = (
        epyc_variable_annotations_mod.homogeneous_empty_list_annotation_int
    )
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(
        epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation())
    )


def test_homogeneous_empty_list_2_annotation_int(epyc_variable_annotations_mod):
    homogeneous_list_annotation = (
        variable_annotations.homogeneous_empty_list_2_annotation_int
    )
    epyc_homogeneous_list_annotation = (
        epyc_variable_annotations_mod.homogeneous_empty_list_2_annotation_int
    )
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(
        epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation())
    )


def test_homogeneous_list_annotation_int(epyc_variable_annotations_mod):
    homogeneous_list_annotation = variable_annotations.homogeneous_list_annotation_int
    epyc_homogeneous_list_annotation = (
        epyc_variable_annotations_mod.homogeneous_list_annotation_int
    )
    orig_output = homogeneous_list_annotation()
    epyc_output = epyc_homogeneous_list_annotation()

    assert epyc_output == orig_output
    assert isinstance(
        epyc_output, type(orig_output)
    )
    assert isinstance(
        epyc_output[0], type(orig_output[0])
    )


def test_homogeneous_list_without_annotation(epyc_variable_annotations_mod):
    homogeneous_list = variable_annotations.homogeneous_list
    epyc_homogeneous_list = epyc_variable_annotations_mod.homogeneous_list
    orig_output = homogeneous_list()
    epyc_output = epyc_homogeneous_list()

    assert epyc_output == orig_output
    assert isinstance(
        epyc_output, type(orig_output)
    )
    assert isinstance(
        epyc_output[0], type(orig_output[0])
    )


def test_homogeneous_list_annotation_float(epyc_variable_annotations_mod):
    homogeneous_list_annotation = variable_annotations.homogeneous_list_annotation_float
    epyc_homogeneous_list_annotation = (
        epyc_variable_annotations_mod.homogeneous_list_annotation_float
    )
    orig_output = homogeneous_list_annotation()
    epyc_output = epyc_homogeneous_list_annotation()

    assert epyc_output == orig_output
    assert isinstance(
        epyc_output, type(orig_output)
    )
    assert isinstance(
        epyc_output[0], type(orig_output[0])
    )


def test_homogeneous_list_annotation_float64(epyc_variable_annotations_mod):
    homogeneous_list_annotation = (
        variable_annotations.homogeneous_list_annotation_float64
    )
    epyc_homogeneous_list_annotation = (
        epyc_variable_annotations_mod.homogeneous_list_annotation_float64
    )
    orig_output = homogeneous_list_annotation()
    epyc_output = epyc_homogeneous_list_annotation()

    assert epyc_output == orig_output
    assert isinstance(
        epyc_output, type(orig_output)
    )
    assert isinstance(
        epyc_output[0], type(orig_output[0])
    )


def test_homogeneous_list_annotation_bool(epyc_variable_annotations_mod):
    homogeneous_list_annotation = variable_annotations.homogeneous_list_annotation_bool
    epyc_homogeneous_list_annotation = (
        epyc_variable_annotations_mod.homogeneous_list_annotation_bool
    )
    orig_output = homogeneous_list_annotation()
    epyc_output = epyc_homogeneous_list_annotation()

    assert epyc_output == orig_output
    assert isinstance(
        epyc_output, type(orig_output)
    )
    assert isinstance(
        epyc_output[0], type(orig_output[0])
    )


def test_homogeneous_list_annotation_complex(epyc_variable_annotations_mod):
    homogeneous_list_annotation = (
        variable_annotations.homogeneous_list_annotation_complex
    )
    epyc_homogeneous_list_annotation = (
        epyc_variable_annotations_mod.homogeneous_list_annotation_complex
    )
    orig_output = homogeneous_list_annotation()
    epyc_output = epyc_homogeneous_list_annotation()

    assert epyc_output == orig_output
    assert isinstance(
        epyc_output, type(orig_output)
    )
    assert isinstance(
        epyc_output[0], type(orig_output[0])
    )


def test_homogeneous_list_annotation_embedded_complex(stc_language):
    def homogeneous_list_annotation():
        a: list[complex] = [1j, 2j]
        b = [a]
        return b[0][0]

    epyc_homogeneous_list_annotation = epyccel(
        homogeneous_list_annotation, language=stc_language
    )
    assert epyc_homogeneous_list_annotation() == homogeneous_list_annotation()
    assert isinstance(
        epyc_homogeneous_list_annotation(), type(homogeneous_list_annotation())
    )


def test_dict_int_float(epyc_variable_annotations_mod):
    dict_int_float = variable_annotations.dict_int_float
    epyc_dict_int_float = epyc_variable_annotations_mod.dict_int_float

    orig_output = dict_int_float()
    epyc_output = epyc_dict_int_float()

    assert epyc_output == orig_output
    assert isinstance(epyc_output, type(orig_output))
    orig_k, orig_v = orig_output.popitem()
    epyc_k, epyc_v = epyc_output.popitem()
    assert isinstance(epyc_k, type(orig_k))
    assert isinstance(epyc_v, type(orig_v))


def test_dict_empty_init(epyc_variable_annotations_mod):
    dict_empty_init = variable_annotations.dict_empty_init
    epyc_dict_empty_init = epyc_variable_annotations_mod.dict_empty_init
    assert epyc_dict_empty_init() == dict_empty_init()


def test_dict_complex_float(epyc_variable_annotations_mod):
    dict_complex_float = variable_annotations.dict_complex_float
    epyc_dict_complex_float = epyc_variable_annotations_mod.dict_complex_float

    orig_output = dict_complex_float()
    epyc_output = epyc_dict_complex_float()

    assert epyc_output == orig_output
    assert isinstance(epyc_output, type(orig_output))
    orig_k, orig_v = orig_output.popitem()
    epyc_k, epyc_v = epyc_output.popitem()
    assert isinstance(epyc_k, type(orig_k))
    assert isinstance(epyc_v, type(orig_v))


def test_inhomogeneous_tuple_annotation_1(epyc_variable_annotations_mod):
    inhomogeneous_tuple_annotation = variable_annotations.inhomogeneous_tuple_annotation
    epyc_inhomogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.inhomogeneous_tuple_annotation
    )
    assert epyc_inhomogeneous_tuple_annotation() == inhomogeneous_tuple_annotation()


def test_inhomogeneous_tuple_annotation_2(epyc_variable_annotations_mod):
    inhomogeneous_tuple_annotation = (
        variable_annotations.inhomogeneous_tuple_annotation_2
    )
    epyc_inhomogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.inhomogeneous_tuple_annotation_2
    )
    assert epyc_inhomogeneous_tuple_annotation() == inhomogeneous_tuple_annotation()


def test_inhomogeneous_tuple_annotation_3(epyc_variable_annotations_mod):
    inhomogeneous_tuple_annotation = (
        variable_annotations.inhomogeneous_tuple_annotation_3
    )
    epyc_inhomogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.inhomogeneous_tuple_annotation_3
    )
    assert epyc_inhomogeneous_tuple_annotation() == inhomogeneous_tuple_annotation()


def test_inhomogeneous_tuple_annotation_4(epyc_variable_annotations_mod):
    inhomogeneous_tuple_annotation = (
        variable_annotations.inhomogeneous_tuple_annotation_4
    )
    epyc_inhomogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.inhomogeneous_tuple_annotation_4
    )
    assert epyc_inhomogeneous_tuple_annotation() == inhomogeneous_tuple_annotation()


def test_inhomogeneous_tuple_annotation_5(epyc_variable_annotations_mod):
    inhomogeneous_tuple_annotation = (
        variable_annotations.inhomogeneous_tuple_annotation_5
    )
    epyc_inhomogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.inhomogeneous_tuple_annotation_5
    )
    assert epyc_inhomogeneous_tuple_annotation() == inhomogeneous_tuple_annotation()


def test_inhomogeneous_tuple_annotation_6(epyc_variable_annotations_mod):
    inhomogeneous_tuple_annotation = (
        variable_annotations.inhomogeneous_tuple_annotation_6
    )
    epyc_inhomogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.inhomogeneous_tuple_annotation_6
    )
    assert epyc_inhomogeneous_tuple_annotation() == inhomogeneous_tuple_annotation()


def test_inhomogeneous_tuple_annotation_7(epyc_variable_annotations_mod):
    inhomogeneous_tuple_annotation = (
        variable_annotations.inhomogeneous_tuple_annotation_7
    )
    epyc_inhomogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.inhomogeneous_tuple_annotation_7
    )
    assert epyc_inhomogeneous_tuple_annotation() == inhomogeneous_tuple_annotation()


def test_inhomogeneous_tuple_annotation_8(epyc_variable_annotations_mod):
    inhomogeneous_tuple_annotation = (
        variable_annotations.inhomogeneous_tuple_annotation_8
    )
    epyc_inhomogeneous_tuple_annotation = (
        epyc_variable_annotations_mod.inhomogeneous_tuple_annotation_8
    )
    assert epyc_inhomogeneous_tuple_annotation() == inhomogeneous_tuple_annotation()


def test_str_declaration(epyc_variable_annotations_mod):
    str_declaration = variable_annotations.str_declaration
    epyc_str_declaration = epyc_variable_annotations_mod.str_declaration
    assert str_declaration() == epyc_str_declaration()


def test_unknown_annotation(language):
    # Initialize singleton that stores Pyccel errors
    def unknown_annotation():
        a: Annotated[int, ">10"] = 15
        return a

    errors = Errors()

    epyc_unknown_annotation = epyccel(unknown_annotation, language=language)

    # Check result of pyccelized function
    assert unknown_annotation() == epyc_unknown_annotation()

    # Check that we got exactly 1 Pyccel warning
    assert errors.has_warnings()
    assert errors.num_messages() == 1

    # Check that the warning is correct
    warning_info = [*errors.error_info_map.values()][0][0]
    assert ">10" in warning_info.symbol
