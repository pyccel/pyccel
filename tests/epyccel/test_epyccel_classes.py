# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import pytest
import modules.expose_classes as mod
from pyccel.epyccel import epyccel

@pytest.fixture(scope="module")
def modnew(language):
    return epyccel(mod, language = language)

def test_class_import(language):
    class A:
        def __init__(self : 'A'):
            pass

    epyc_A = epyccel(A)

    assert isinstance(epyc_A, type)


def test_class_return(modnew):
    a = modnew.get_A()
    assert isinstance(a, modnew.A)
    a_new, i = modnew.get_A_int()
    a_new2, i2 = mod.get_A_int()
    assert isinstance(a_new, modnew.A)
    assert isinstance(a_new2, modnew.A)
    assert i == i2

    b = modnew.get_B(3.0)
    assert isinstance(b, modnew.B)
