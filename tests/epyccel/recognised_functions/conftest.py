# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest

@pytest.fixture( params=[
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c)
    ]
)
def language(request):
    return request.param
