# pylint: disable=missing-function-docstring, missing-module-docstring/
import numpy as np
from pyccel.epyccel import epyccel

def test_module_1():
    import modules.python_annotations as mod

    modnew = epyccel(mod)
