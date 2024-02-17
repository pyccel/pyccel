# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
import numpy as np
import modules.function_call_before_definition as mod
from pyccel.epyccel import epyccel

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Can't compile nested functions"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_gen_1(language):
    modnew = epyccel(mod, language = language)
    assert mod.x1 == modnew.x1
    assert mod.x2 == modnew.x2
