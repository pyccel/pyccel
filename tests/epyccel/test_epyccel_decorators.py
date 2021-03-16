# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

import pytest
from pyccel.epyccel import epyccel
from pyccel.decorators import private

@pytest.fixture(params=[
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c)
    ]
)
def language(request):
    return request.param

def test_private(language):
    @private
    def f():
        print("hidden")

    g = epyccel(f, language=language)

    with pytest.raises(NotImplementedError):
        g()

