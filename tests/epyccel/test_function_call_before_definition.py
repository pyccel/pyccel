# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
import modules.function_call_before_definition as mod1
import modules.function_call_before_definition_2 as mod2

from pyccel import epyccel

def test_fun_1(language):
    modnew = epyccel(mod1, language = language)
    assert mod1.x1 == modnew.x1
    assert mod1.x2 == modnew.x2
    assert mod1.x3 == modnew.x3

def test_fun_2(language):
    modnew = epyccel(mod2, language = language)
    assert mod2.a == modnew.a
