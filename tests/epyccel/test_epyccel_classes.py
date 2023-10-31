# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from pyccel.epyccel import epyccel

def test_class_import(language):
    class A:
        def __init__(self : 'A'):
            pass

    epyc_A = epyccel(A)

    assert isinstance(epyc_A, type)
