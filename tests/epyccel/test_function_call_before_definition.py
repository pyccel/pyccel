# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
import modules.function_call_before_definition as mod1
import modules.function_call_before_definition_2 as mod2

from pyccel import epyccel

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Functions in functions not implemented in c"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_fun_1(language):
    modnew = epyccel(mod1, language = language)
    assert mod1.x1 == modnew.x1
    assert mod1.x2 == modnew.x2
    assert mod1.x3 == modnew.x3

def test_fun_2(language):
    modnew = epyccel(mod2, language = language)
    assert mod2.a == modnew.a
