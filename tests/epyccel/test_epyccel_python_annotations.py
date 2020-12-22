# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.epyccel import epyccel

def test_module_1():
    import modules.python_annotations as mod

    modnew = epyccel(mod)
    assert modnew.fib(10) == mod.fib(10)
