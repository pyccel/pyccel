# pylint: disable=missing-function-docstring, missing-module-docstring
import inspect

import pytest
from modules import strings_module

from epyccel_utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_strings_module(language):
    return epyccel_module_with_fallback(strings_module, language)

failing_tests = {
    "concatenate": "C does not support string concatenation",
    "concatenate_multiple": "C does not support string concatenation",
    "concatenate_expr": "C does not support string concatenation",
}


@pytest.mark.parametrize("test_func", [
    "one_quote",
    "two_quote",
    "three_quote",
    "return_literal",
    "concatenate",
    "concatenate_multiple",
    "concatenate_expr",
    "string_function_call",
    "string_function_call_on_literal",
    "string_function_return",
])
def test_strings(test_func, epyc_strings_module):
    if test_func in failing_tests and epyc_strings_module.language == "c":
        pytest.xfail(failing_tests[test_func])

    f1 = getattr(strings_module, test_func)
    f2 = getattr(epyc_strings_module, test_func)

    python_out = f1()
    pyccel_out = f2()
    print(python_out)
    print(pyccel_out)
    assert python_out == pyccel_out


def test_string_compare(epyc_strings_module):
    str_comp = strings_module.str_comp
    f = epyc_strings_module.str_comp

    assert str_comp() == f()


def test_string_argument(epyc_strings_module):
    str_option_test = strings_module.str_option_test
    f = epyc_strings_module.str_option_test

    assert str_option_test("do this") == f("do this")
    assert str_option_test("do that") == f("do that")


def test_string_argument_optional(epyc_strings_module):
    str_option_test = strings_module.string_argument_optional_str_option_test
    f = epyc_strings_module.string_argument_optional_str_option_test

    assert str_option_test("do this") == f("do this")
    assert str_option_test("do that") == f("do that")
    assert str_option_test() == f()
