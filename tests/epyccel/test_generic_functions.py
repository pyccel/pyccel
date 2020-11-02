 # pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
import modules.generic_functions as mod
from pyccel.epyccel import epyccel

@pytest.fixture( params=[
        pytest.param("fortran", marks = [
            pytest.mark.fortran,
            pytest.mark.skip]),
        pytest.param("c", marks = [
            pytest.mark.c]
        )
    ]
)
def language(request):
    return request.param

@pytest.mark.parametrize("a",[(5),(5.5)])
def test_f1(language, a):
    f1 = mod.f1
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)
