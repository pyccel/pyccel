import pytest
import numpy as np

from pyccel.epyccel import epyccel

#==============================================================================
def test_module_1():
    import modules.openmp as openmp

    mod = epyccel(openmp, accelerator='openmp')
    mod.set_num_threads(4)
    assert mod.get_num_threads() == 4

    assert mod.f1(0) == 0
    assert mod.f1(1) == 1
    assert mod.f1(2) == 2
    assert mod.f1(3) == 3

    assert mod.f1(5) == -1

    mod.set_num_threads(8)
    assert mod.get_num_threads() == 8

    assert mod.f1(5) == 5

