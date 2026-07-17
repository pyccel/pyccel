# pylint: disable=missing-function-docstring, missing-module-docstring
import inspect

import pytest
from modules import strings, strings_module


from epyccel_utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_strings_mod(language):
    return epyccel_module_with_fallback(strings, language)


@pytest.fixture(scope="module")
def epyc_strings_module(language):
    return epyccel_module_with_fallback(strings_module, language)


string_funcs = [
    f for f in strings_module.__all__ if inspect.isfunction(getattr(strings_module, f))
]

failing_tests = {
    "concatenate": "C does not support string concatenation",
    "concatenate_multiple": "C does not support string concatenation",
    "concatenate_expr": "C does not support string concatenation",
}


@pytest.mark.parametrize("test_func", string_funcs)
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


def test_string_compare(epyc_strings_mod):
    str_comp = strings.str_comp
    f = epyc_strings_mod.str_comp

    assert str_comp() == f()


def test_string_argument(epyc_strings_mod):
    str_option_test = strings.str_option_test
    f = epyc_strings_mod.str_option_test

    assert str_option_test("do this") == f("do this")
    assert str_option_test("do that") == f("do that")


def test_string_argument_optional(epyc_strings_mod):
    str_option_test = strings.string_argument_optional_str_option_test
    f = epyc_strings_mod.string_argument_optional_str_option_test

    assert str_option_test("do this") == f("do this")
    assert str_option_test("do that") == f("do that")
    assert str_option_test() == f()
