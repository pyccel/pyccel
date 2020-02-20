import pytest
import numpy as np

from pyccel import epyccel
from modules import base

def compare_epyccel(f, *args):
    f2 = epyccel(f)
    out = f(*args)
    out2 = f2(*args)
    assert np.equal(out1, out2)

def test_cmp_bool():
    compare_epyccel(base.cmp_bool, True, True)

